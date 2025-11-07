import os
import json
from datetime import datetime
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent
from langchain.agents.agent_types import AgentType
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.schema import SystemMessage, HumanMessage, AIMessage
from app.models import ConversationMemory

load_dotenv()

AI_API_KEY = os.getenv("AI_API_KEY")
DB_URL = os.getenv("DB_URL")

# ===================== LLM Initialization =====================

def get_llm(temperature=0.4):
    """Return initialized LLM with token tracking."""
    callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
    return ChatGoogleGenerativeAI(
        model="gemini-2.0-flash-exp",
        temperature=temperature,
        google_api_key=AI_API_KEY,
    )


# ===================== MEMORY HANDLING =====================

def get_conversation_summary(user) -> str:
    """Summarize last 5 conversations."""
    try:
        past_convos = ConversationMemory.objects.filter(user=user).order_by('-created_at')[:5]

        if not past_convos.exists():
            return "No previous conversation history."

        history_text = "\n".join([
            f"User: {conv.user_input}\nAI: {conv.ai_response}"
            for conv in reversed(past_convos)
        ])

        llm = get_llm(temperature=0.2)
        prompt = PromptTemplate(
            input_variables=["history"],
            template="""
Summarize this chat in 2-3 sentences (Urdu/English mix allowed).
Focus on what the user was generally looking for or discussing.

{history}

Summary:"""
        )
        summary_chain = LLMChain(llm=llm, prompt=prompt)
        summary = summary_chain.run({"history": history_text})
        return summary.strip()

    except Exception as e:
        print(f"Error getting summary: {e}")
        return "No previous conversation history."


# ===================== QUERY DETECTION =====================

def detect_query_type(user_query: str) -> str:
    """Detect whether query is about DB or chat/advice."""
    database_keywords = [
        # English
        'find', 'search', 'show', 'list', 'get', 'profiles', 'users',
        'female', 'male', 'girl', 'boy', 'location', 'lahore', 'karachi', 'age',
        # Roman Urdu
        'rishta', 'user dikhao', 'mujhe chahiye', 'ladki', 'larki', 'ladka',
        'profile dikhao', 'show rishta', 'search karo', 'filter karo',
        'umar', 'city', 'shahar', 'umar walay', 'users dikhaiye'
    ]

    advice_keywords = [
        'advice', 'suggestion', 'relationship', 'love', 'trust',
        'larka kesa hona chahiye', 'ladki kaisi', 'breakup', 'shaadi advice',
        'relationship tips', 'rishta advice', 'feelings', 'mashwara'
    ]

    query_lower = user_query.lower()

    if any(keyword in query_lower for keyword in database_keywords):
        return "database"
    elif any(keyword in query_lower for keyword in advice_keywords):
        return "advice"
    return "chat"


# ===================== DATABASE QUERY =====================

def query_database(user_query: str, context: str = "") -> str:
    """Query the DB via natural language (English + Roman Urdu)."""
    try:
        db = SQLDatabase.from_uri(DB_URL)
        llm = get_llm(temperature=0)
        sql_agent = create_sql_agent(
            llm=llm,
            db=db,
            agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=5
        )
        full_query = f"Context: {context}\n\nQuery: {user_query}"
        result = sql_agent.invoke({"input": full_query})
        return result.get("output", "No results found.")
    except Exception as e:
        return f"Database query failed: {str(e)}"


# ===================== RESPONSE GENERATION =====================

def generate_contextual_response(user_query: str, summary: str, db_result: str = None, query_type: str = "chat") -> str:
    """Generate response with personality and bilingual tone."""
    try:
        llm = get_llm(temperature=0.7)

        # Base personality
        base_prompt = """
You are **Asaan Rishta AI**, a warm, intelligent matchmaking assistant who speaks in a friendly mix of English and Roman Urdu.

Previous Chat Summary:
{summary}

User Message: {query}

{db_context}

Instructions:
- If query_type == "advice", give short relationship guidance (in friendly Roman Urdu + English mix).
- If database results exist, summarize them naturally.
- If just chatting, keep it casual and human-like.
- Never sound robotic or repetitive.
- Keep replies 2–4 sentences long.
- Be empathetic, positive, and slightly humorous if suitable.

Response:
"""

        db_context = f"Database Results:\n{db_result}" if db_result else ""
        prompt = PromptTemplate(
            input_variables=["summary", "query", "db_context"],
            template=base_prompt
        )

        chain = LLMChain(llm=llm, prompt=prompt)
        response = chain.run({
            "summary": summary,
            "query": user_query,
            "db_context": db_context
        })

        return response.strip()

    except Exception as e:
        return f"Sorry, I'm having trouble responding right now. ({str(e)})"


# ===================== MAIN EXECUTION =====================

def execute_query(user, user_query: str) -> str:
    """Core entry point: detect query, handle memory, generate reply."""
    try:
        summary = get_conversation_summary(user)
        query_type = detect_query_type(user_query)

        print(f"Query Type: {query_type}")
        print(f"Summary: {summary}")

        db_result = None
        if query_type == "database":
            db_result = query_database(user_query, context=summary)

        response = generate_contextual_response(
            user_query=user_query,
            summary=summary,
            db_result=db_result,
            query_type=query_type
        )

        # Save memory
        ConversationMemory.objects.create(
            user=user,
            user_input=user_query,
            ai_response=response
        )

        return response

    except Exception as e:
        print(f"Error in execute_query: {str(e)}")
        error_msg = "Sorry, something went wrong. Please try again."
        try:
            ConversationMemory.objects.create(
                user=user,
                user_input=user_query,
                ai_response=error_msg
            )
        except:
            pass
        return error_msg

import os
import json
from datetime import datetime
from dotenv import load_dotenv
from typing import List
from langchain.agents import Tool, initialize_agent, AgentType
from langchain.memory import ConversationBufferMemory
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.utilities import SQLDatabase
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from app.models import ConversationMemory  # Your Django models

load_dotenv()

AI_API_KEY = os.getenv("AI_API_KEY")
DB_URL = os.getenv("DB_URL")

# === Tool 1: Filter / Query Database ===
def filter_database(query: str) -> str:
    """
    Natural language database query.
    Example: 'Find all female users aged 25 in Lahore'
    """
    try:
        db = SQLDatabase.from_uri(DB_URL)
        toolkit = SQLDatabaseToolkit(db=db)
        tools = toolkit.get_tools()

        query_tool = next((t for t in tools if t.name == "query_sql_db"), None)
        if not query_tool:
            return "SQL query tool not found."

        sql_result = query_tool.run(query)
        if isinstance(sql_result, list):
            return json.dumps(sql_result, indent=2, ensure_ascii=False)
        return str(sql_result)
    except Exception as e:
        return f"Database query failed: {str(e)}"

# === Tool 2: Generate Contextual Reply ===
def generate_reply(history: str, message: str) -> str:
    """
    Generates a context-aware reply based on past conversation.
    """
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=AI_API_KEY)
    prompt = f"""
    You are a professional but friendly assistant.
    Here's the past chat with the user:

    {history}

    User's new message:
    {message}

    If the user was searching something before, remind them naturally.
    Example: "Last time you were finding profiles aged 25 from Lahore. Do you still want to continue that search?"
    Respond in a warm, human-like tone (2–3 lines).
    """
    return llm.predict(prompt)

# === Tool 3: Get Current Time (optional helper) ===
def get_current_time(_: str) -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# === TOOL REGISTRATION ===
tools = [
    Tool(
        name="FilterDatabase",
        func=filter_database,
        description="Use this tool to search or filter users from the database using natural language."
    ),
    Tool(
        name="GenerateReply",
        func=generate_reply,
        description="Use this tool to reply to users naturally based on their previous conversations."
    ),
    Tool(
        name="GetTime",
        func=get_current_time,
        description="Use this to tell current date and time."
    ),
]

# === MAIN EXECUTION LOGIC ===
def execute_query(user, user_query: str) -> str:
    """
    Executes a user query through the agent, maintaining memory and summarizing past 5 chats.
    """
    # Load user’s past 5 messages
    past_convos = ConversationMemory.objects.filter(user=user).order_by('-created_at')[:5]
    history_text = "\n".join(
        [f"User: {conv.user_input}\nAI: {conv.ai_response}" for conv in reversed(past_convos)]
    )

    # Summarize history
    if history_text:
        llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=AI_API_KEY)
        prompt = PromptTemplate(
            input_variables=["history"],
            template="Summarize the user's recent conversation in 3 sentences:\n\n{history}\n\nSummary:"
        )
        summary_chain = LLMChain(llm=llm, prompt=prompt)
        summary = summary_chain.run({"history": history_text})
    else:
        summary = "No previous chat history."

    # Memory setup
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    for conv in reversed(past_convos):
        memory.save_context({"input": conv.user_input}, {"output": conv.ai_response})

    # LLM
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        temperature=0.3,
        google_api_key=AI_API_KEY
    )

    # Initialize agent
    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
        memory=memory,
        verbose=True
    )

    # Add system context for personalization
    system_prompt = f"""
    You are Asaan Rishta AI Assistant.
    You remember user conversations.
    Summary of past interactions: {summary}

    When replying, if the user seems to continue a previous topic,
    remind them politely and ask if they want to continue.
    """

    full_prompt = f"{system_prompt}\n\nUser: {user_query}"

    # Run agent
    try:
        response = agent.invoke({"input": full_prompt})
        output = response.get("output", "No response generated.")
    except Exception as e:
        output = f"Error processing your request: {e}"

    # Save conversation in DB
    ConversationMemory.objects.create(user=user, user_input=user_query, ai_response=output)

    return output

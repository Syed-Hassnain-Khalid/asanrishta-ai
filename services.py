import os
from dotenv import load_dotenv
from langchain.agents import initialize_agent, AgentType
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.utilities import SQLDatabase
from langchain.memory import ConversationBufferMemory
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from app.models import ConversationMemory
from typing import Dict, Any

# === Load Environment ===
load_dotenv()
AI_API_KEY = os.getenv("AI_API_KEY")
DB_URL = os.getenv("DB_URL")

# === Setup LLM (Google Gemini) ===
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0.3,
    verbose=True,
    google_api_key=AI_API_KEY,
)

# === Setup SQL Database ===
db = SQLDatabase.from_uri(DB_URL)
toolkit = SQLDatabaseToolkit(db=db, llm=llm)

# === User-specific memory storage ===
user_memories: Dict[int, Dict[str, Any]] = {}

# === Summarization Prompt ===
summary_prompt = PromptTemplate(
    input_variables=["history"],
    template="""Summarize the following conversation concisely, focusing on:
- Key user preferences and information
- Important topics discussed
- Any database queries or results
- User's relationship goals or requirements

Conversation:
{history}

Summary:"""
)

# === FAQ Content for Rishta App ===
RISHTA_FAQ = """
Common Questions about Our Rishta Platform:

**Profile & Registration:**
- Create detailed profiles with photos, bio, education, profession, and preferences
- Verify your profile for increased trust
- Privacy controls to show/hide information

**Matching & Search:**
- Advanced filters: age, location, education, profession, religion, sect
- AI-powered recommendations based on compatibility
- Save favorite profiles and send interest

**Communication:**
- Send interest requests to profiles you like
- Chat with matched profiles after mutual acceptance
- Video call feature for verified members

**Safety & Privacy:**
- All profiles are moderated
- Report and block features available
- Your information is encrypted and secure
- Control who can see your profile

**Subscription:**
- Free basic membership with limited features
- Premium plans for unlimited messaging and advanced search
- Special family packages available
"""

# === System Prompts ===
SQL_PROMPT = """You are a helpful AI assistant for a Rishta (marriage/matchmaking) platform.
You can access the database to help users with:
- Finding matches based on criteria
- Viewing profile statistics
- Checking their received/sent interests
- Analyzing compatibility data

When answering database queries:
1. Be respectful and maintain privacy
2. Explain results in a friendly, conversational way
3. Suggest relevant next steps

Current conversation context: {summary}

User query: {input}"""

# CHAT_PROMPT = """You are QuizHippo AI, a friendly and knowledgeable assistant for a Rishta (matchmaking) platform.

# Your role:
# - Answer FAQs about the platform features
# - Provide relationship advice and tips
# - Help with profile creation guidance
# - Explain how matching algorithms work
# - Offer cultural sensitivity and respect
# - Be warm, supportive, and professional

# Available FAQ Information:
# {faq_content}

# Conversation Summary: {summary}

# Recent Conversation:
# {history}

# User: {user_input}
# AI:"""
CHAT_PROMPT = """
You are RishtaMate AI — a caring, respectful, and intelligent assistant for a Rishta (matchmaking) platform.

Your role:
- Help users find compatible matches and guide them through the matchmaking process
- Answer FAQs about profiles, interests, verification, and communication features
- Explain how AI recommendations and matching algorithms work
- Give relationship and compatibility guidance with empathy and cultural sensitivity
- Support users in creating honest and appealing profiles
- Maintain privacy, respect, and professionalism at all times

Tone and Language:
- Friendly, warm, and supportive
- Respond **in the same language as the user**, including Roman Urdu or English
- Respect cultural and family values common in South Asian matchmaking contexts

Available Information:
{faq_content}

Conversation Summary:
{summary}

Recent Conversation:
{history}

User: {user_input}
AI:"""


# === Routing Decision Prompt ===
# ROUTING_PROMPT = """Analyze this user query and determine if it requires database access or can be answered conversationally.

# Query: "{query}"

# Database access is needed for:
# - Searching/filtering profiles (e.g., "find matches in Lahore", "show me engineers")
# - Getting statistics (e.g., "how many profiles", "my received interests")
# - Specific data retrieval (e.g., "show my matches", "list pending requests")
# - CRUD operations on user data

# Conversational response is sufficient for:
# - FAQ questions about platform features
# - Relationship advice or tips
# - How-to questions about using the app
# - General conversation
# - Profile creation guidance

# Respond with ONLY one word: DATABASE or CHAT"""

ROUTING_PROMPT = """
Analyze this user query (it can be in English, Roman Urdu, or mixed) and determine if it requires database access or can be answered conversationally.

Database access is needed for:
- Searching/filtering profiles (e.g., "find matches in Lahore", "Lahore mein engineers dikhao")
- Getting statistics (e.g., "how many profiles", "mera received interests kya hain")
- Specific data retrieval (e.g., "show my matches", "list pending requests")
- CRUD operations on user data

Conversational response is sufficient for:
- FAQ questions about platform features
- Relationship advice or tips
- How-to questions about using the app
- General conversation
- Profile creation guidance

Respond with ONLY one word: DATABASE or CHAT
"""



def get_or_create_memory(user_id: int) -> Dict[str, Any]:
    """Get or create memory objects for a specific user."""
    if user_id not in user_memories:
        user_memories[user_id] = {
            'buffer': ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True,
                input_key="input",
                output_key="output"
            ),
            'summary': ""
        }
    return user_memories[user_id]


def load_user_memory(user):
    """
    Loads previous conversation history from DB and creates a summary.
    """
    memory_obj = get_or_create_memory(user.id)
    buffer_memory = memory_obj['buffer']
    
    # Load past conversations
    history = ConversationMemory.objects.filter(user=user).order_by("-created_at")[:50]  # Last 50 messages
    history = reversed(history)  # Chronological order
    
    chat_text = ""
    for item in history:
        buffer_memory.chat_memory.add_user_message(item.user_input)
        buffer_memory.chat_memory.add_ai_message(item.ai_response)
        chat_text += f"User: {item.user_input}\nAI: {item.ai_response}\n\n"
    
    # Generate summary if there's history
    if chat_text.strip():
        summary_chain = LLMChain(llm=llm, prompt=summary_prompt)
        summary = summary_chain.run({"history": chat_text})
        memory_obj['summary'] = summary
    
    return memory_obj


def save_conversation(user, query: str, response: str):
    """Save user query and AI response in the database."""
    ConversationMemory.objects.create(
        user=user,
        user_input=query,
        ai_response=response,
    )


# def decide_routing(query: str) -> str:
#     """Use LLM to intelligently decide routing."""
#     routing_chain = LLMChain(
#         llm=llm,
#         prompt=PromptTemplate(input_variables=["query"], template=ROUTING_PROMPT)
#     )
    
#     try:
#         decision = routing_chain.run({"query": query}).strip().upper()
#         return "database" if "DATABASE" in decision else "chat"
#     except:
#         # Fallback to keyword matching
#         db_keywords = [
#             "find", "search", "show", "list", "filter", "match", "profile",
#             "how many", "count", "statistics", "database", "query",
#             "interest", "request", "sent", "received", "pending"
#         ]
#         return "database" if any(kw in query.lower() for kw in db_keywords) else "chat"

def decide_routing(query: str) -> str:
    """Use LLM to intelligently decide routing (Roman Urdu supported)."""
    routing_chain = LLMChain(
        llm=llm,
        prompt=PromptTemplate(input_variables=["query"], template=ROUTING_PROMPT)
    )
    
    try:
        decision = routing_chain.run({"query": query}).strip().upper()
        return "database" if "DATABASE" in decision else "chat"
    except Exception as e:
        print(f"Routing error: {str(e)}")
        # If LLM fails, default to chat rather than risking wrong DB access
        return "chat"



def create_sql_agent(memory):
    """Create SQL agent with memory."""
    return initialize_agent(
        tools=toolkit.get_tools(),
        llm=llm,
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        memory=memory,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=5
    )


def ai_router(user, query: str) -> str:
    """
    Main routing function with intelligent decision making and memory.
    """
    try:
        # Load user's memory (includes past conversations and summary)
        memory_obj = load_user_memory(user)
        buffer_memory = memory_obj['buffer']
        conversation_summary = memory_obj['summary']
        
        # Decide routing using LLM
        route = decide_routing(query)
        
        if route == "database":
            # === SQL Agent Mode ===
            sql_agent = create_sql_agent(buffer_memory)
            
            # Add context to the query
            context_prompt = SQL_PROMPT.format(
                summary=conversation_summary or "No previous context",
                input=query
            )
            
            response = sql_agent.run(context_prompt)
            
        else:
            # === Chat Agent Mode ===
            # Get recent conversation history
            recent_messages = buffer_memory.chat_memory.messages[-10:]  # Last 10 messages
            history_text = "\n".join([
                f"{'User' if i % 2 == 0 else 'AI'}: {msg.content}" 
                for i, msg in enumerate(recent_messages)
            ])
            
            # Create chat agent with full context
            chat_agent = LLMChain(
                llm=llm,
                prompt=PromptTemplate(
                    input_variables=["faq_content", "summary", "history", "user_input"],
                    template=CHAT_PROMPT
                ),
                verbose=True
            )
            
            response = chat_agent.run({
                "faq_content": RISHTA_FAQ,
                "summary": conversation_summary or "New conversation",
                "history": history_text or "No recent messages",
                "user_input": query
            })
        
        # Save to memory
        buffer_memory.save_context({"input": query}, {"output": response})
        save_conversation(user, query, response)
        
        # Auto-summarize every 20 messages
        message_count = len(buffer_memory.chat_memory.messages)
        if message_count > 0 and message_count % 20 == 0:
            full_history = "\n".join([
                f"{'User' if i % 2 == 0 else 'AI'}: {msg.content}"
                for i, msg in enumerate(buffer_memory.chat_memory.messages)
            ])
            
            summary_chain = LLMChain(llm=llm, prompt=summary_prompt)
            new_summary = summary_chain.run({"history": full_history})
            memory_obj['summary'] = new_summary
        
        return response.strip()
    
    except Exception as e:
        print(f"Error in ai_router: {str(e)}")
        return f"I apologize, but I encountered an error processing your request. Please try rephrasing your question or contact support if the issue persists."


def clear_user_memory(user_id: int):
    """Clear memory for a specific user (useful for logout/cleanup)."""
    if user_id in user_memories:
        del user_memories[user_id]
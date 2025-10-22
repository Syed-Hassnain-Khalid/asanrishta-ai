import os
from dotenv import load_dotenv
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain_community.agent_toolkits.sql.base import create_sql_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage
from app.models import ConversationMemory

load_dotenv()

# === ENV VARIABLES ===
AI_API_KEY = os.getenv("AI_API_KEY")
DB_URL = os.getenv("DB_URL")

# === LLM SETUP ===
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash-lite",
    temperature=0,
    verbose=True,
    google_api_key=AI_API_KEY
)

# === DATABASE TOOLKIT SETUP ===
db = SQLDatabase.from_uri(DB_URL)
toolkit = SQLDatabaseToolkit(db=db, llm=llm)

# === CREATE SQL AGENT ===
agent = create_sql_agent(
    llm=llm,
    toolkit=toolkit,
    verbose=True
)

# === SYSTEM PROMPT ===
system_prompt = SystemMessage(
    content="You are a helpful AI assistant that helps users query and summarize data from their database. "
            "Always explain your answers clearly and concisely."
)

# === LOAD MEMORY FOR USER ===
def load_user_memory(user):
    """Retrieve and format past conversation memory for a specific user."""
    history = ConversationMemory.objects.filter(user=user).order_by('created_at')
    formatted = "\n".join([
        f"User: {h.user_input}\nAI: {h.ai_response}"
        for h in history
    ])
    return formatted


# === MAIN AGENT FUNCTION ===
def ask_database(user, query: str) -> str:
    """
    Executes the user's query using LangChain SQL Agent with user-specific memory.
    Stores each query-response pair in the ConversationMemory table.
    """
    try:
        # Load past conversation context
        memory_context = load_user_memory(user)
        full_input = (
            f"{system_prompt.content}\n\n"
            f"Conversation history:\n{memory_context}\n\n"
            f"User query: {query}"
        )

        # Get agent response
        response = agent.invoke({"input": full_input})
        output_text = response.get("output", "No response received.")

        # Save conversation to persistent memory
        ConversationMemory.objects.create(
            user=user,
            user_input=query,
            ai_response=output_text
        )

        return output_text

    except Exception as e:
        return f"Error: {str(e)}"

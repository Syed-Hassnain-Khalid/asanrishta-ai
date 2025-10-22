import os
from dotenv import load_dotenv
from langchain.agents.agent_toolkits import create_sql_agent, SQLDatabaseToolkit
from langchain.utilities import SQLDatabase
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import SystemMessage
from app.models import ConversationMemory

# === Load environment variables ===
load_dotenv()

AI_API_KEY = os.getenv("AI_API_KEY")
DB_URL = os.getenv("DB_URL")

# === LLM (Google Gemini) ===
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",  # or gemini-pro if you prefer
    temperature=0.2,
    verbose=True,
    google_api_key=AI_API_KEY,
)

# === Setup SQL Database ===
db = SQLDatabase.from_uri(DB_URL)

# === Setup SQL Toolkit ===
toolkit = SQLDatabaseToolkit(db=db, llm=llm)

# === Create SQL Agent ===
agent = create_sql_agent(
    llm=llm,
    toolkit=toolkit,
    verbose=True,
)

# === System Prompt ===
system_prompt = SystemMessage(
    content=(
        "You are a helpful AI assistant that helps users query and summarize data from their database. "
        "Always explain your answers clearly and concisely. "
        "If the user asks something outside the database, politely say you can only assist with data queries."
    )
)

# === Helper: Load User Memory ===
def load_user_memory(user):
    """
    Load previous chat history for a given user from ConversationMemory table.
    """
    history = ConversationMemory.objects.filter(user=user).order_by("created_at")
    formatted = "\n".join([
        f"User: {item.user_input}\nAI: {item.ai_response}"
        for item in history
    ])
    return formatted


# === Main Function: Ask Database ===
def ask_database(user, query: str) -> str:
    """
    Executes the user's SQL-related query using the LangChain SQL agent.
    Stores query-response pairs per user for persistent conversation memory.
    """
    try:
        # Get user chat history
        memory_context = load_user_memory(user)

        # Combine context + system prompt + new query
        full_prompt = (
            f"{system_prompt.content}\n\n"
            f"Conversation history:\n{memory_context}\n\n"
            f"User query: {query}"
        )

        # Run the agent
        response = agent.run(full_prompt)

        # Save chat in DB
        ConversationMemory.objects.create(
            user=user,
            user_input=query,
            ai_response=response,
        )

        return response

    except Exception as e:
        return f"Error: {str(e)}"

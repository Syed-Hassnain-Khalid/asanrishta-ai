import os
import re
import json
from dotenv import load_dotenv
from typing import Dict, Any
from langchain.agents import initialize_agent, AgentType
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.utilities import SQLDatabase
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from app.models import ConversationMemory  # your Django model

# === ENVIRONMENT ===
load_dotenv()
AI_API_KEY = os.getenv("AI_API_KEY")
DB_URL = os.getenv("DB_URL")

# === LLM (Gemini) ===
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0.3,
    google_api_key=AI_API_KEY,
)

# === SQL AGENT (Toolkit + cached globally) ===
db = SQLDatabase.from_uri(DB_URL)
toolkit = SQLDatabaseToolkit(db=db, llm=llm)
sql_agent = initialize_agent(
    tools=toolkit.get_tools(),
    llm=llm,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=False,
    handle_parsing_errors=True,
    max_iterations=8,
    max_execution_time=45,
)

# === UNIFIED PROMPT — Dynamic intent routing ===
UNIFIED_PROMPT = """
You are **RishtaMate AI**, a matchmaking assistant with both relationship knowledge and direct access to a live user database.

**Database Information**
Table: `users`
Columns: id, first_name, last_name, gender, age, city, cast, height, education, occupation, marital_status, is_active
Only use `is_active = 1` users.

**Your Tasks**
1. Decide if the query needs **database information** (profiles, matches, filters, user data)
   or if it’s a **general/FAQ** question (advice, relationship tips, small talk).
   - If the query clearly asks about people, profiles, searches, or filters → use the SQL tool automatically.
   - If not, answer conversationally.
2. Never fabricate results; only use DB when required.
3. Respond in user’s language ({language}).
4. Be concise, friendly, and clear.
5. Respect conversation context and memory summary below.

**Conversation Context**
{context_summary}

**User Query**
{query}

Now intelligently decide whether to use DB or just respond naturally.
Return your final response (not your reasoning).
"""

# === SUMMARY PROMPT ===
SUMMARY_PROMPT = """
Summarize this chat briefly:
- Key preferences (city, cast, height, etc.)
- User’s tone (English / Roman Urdu)
- Main goal

Messages:
{messages}

Short summary:
"""

# === MEMORY MANAGER ===
class EfficientMemoryManager:
    def __init__(self, user_id: int):
        self.user_id = user_id
        self.recent_messages = []
        self.summary = ""
        self.language = "english"
        self.preferences = {}

    def add_exchange(self, user_input: str, ai_response: str):
        self.recent_messages.append({"user": user_input, "ai": ai_response})
        self.recent_messages = self.recent_messages[-10:]
        self.language = self._detect_language(user_input)
        self._extract_preferences(user_input)

    def _detect_language(self, text: str) -> str:
        roman_urdu_keywords = ['aap', 'hai', 'mein', 'ka', 'ki', 'dikhao', 'rishta', 'batao', 'chahiye']
        matches = sum(word in text.lower() for word in roman_urdu_keywords)
        return "roman_urdu" if matches >= 2 else "english"

    def _extract_preferences(self, text: str):
        text = text.lower()
        casts = ['syed', 'sheikh', 'rajput', 'awan', 'arain', 'jat', 'gujjar']
        cities = ['lahore', 'karachi', 'islamabad', 'rawalpindi', 'faisalabad']
        for cast in casts:
            if cast in text:
                self.preferences['cast'] = cast.capitalize()
        for city in cities:
            if city in text:
                self.preferences['city'] = city.capitalize()
        height = re.search(r'(\d\.\d+)', text)
        if height:
            self.preferences['height'] = height.group(1)

    def get_summary(self, llm) -> str:
        if self.summary and len(self.recent_messages) < 10:
            return self.summary
        try:
            msg_text = "\n".join([f"User: {m['user']}\nAI: {m['ai']}" for m in self.recent_messages])
            chain = LLMChain(
                llm=llm,
                prompt=PromptTemplate(input_variables=["messages"], template=SUMMARY_PROMPT),
            )
            self.summary = chain.invoke({"messages": msg_text})["text"]
        except Exception as e:
            print("Summary error:", e)
            self.summary = "Summary unavailable."
        return self.summary

    def get_context(self) -> str:
        parts = []
        if self.summary:
            parts.append(f"Summary: {self.summary}")
        if self.preferences:
            prefs = ", ".join(f"{k}={v}" for k, v in self.preferences.items())
            parts.append(f"Prefs: {prefs}")
        if self.recent_messages:
            last = " | ".join(f"U:{m['user'][:50]}" for m in self.recent_messages[-3:])
            parts.append(f"Recent: {last}")
        return " • ".join(parts) or "New user"


# === MEMORY CACHE ===
user_memories: Dict[int, EfficientMemoryManager] = {}

def get_memory(user_id: int) -> EfficientMemoryManager:
    if user_id not in user_memories:
        user_memories[user_id] = EfficientMemoryManager(user_id)
    return user_memories[user_id]


# === MAIN EXECUTOR (Dynamic routing, single AI call) ===
def execute_query(user, query: str) -> str:
    try:
        memory = get_memory(user.id)
        context = memory.get_context()
        language = memory.language

        final_prompt = UNIFIED_PROMPT.format(context_summary=context, query=query, language=language)

        print(f"\n--- Routing ---\nUser: {user.id} | Lang: {language}\nPrompt built.\n")

        # ✅ ONE AI CALL — Gemini internally decides whether to use DB or not
        result = sql_agent.invoke({"input": final_prompt})

        # Handle structured / plain outputs
        response = ""
        if isinstance(result, dict):
            response = result.get("output") or result.get("text") or str(result)
        else:
            response = str(result)

        # === Update memory + save ===
        memory.add_exchange(query, response)
        ConversationMemory.objects.create(user=user, user_input=query, ai_response=response)

        # === Periodic summarization ===
        if len(memory.recent_messages) >= 10:
            memory.get_summary(llm)

        return response.strip()

    except Exception as e:
        print("Execution error:", e)
        if any(w in query.lower() for w in ['aap', 'hai', 'mein', 'kya']):
            return "Maaf karna, kuch masla aagaya. Dobara koshish karein?"
        return "Sorry, something went wrong. Please try again."


# === UTILITIES ===
def clear_user_memory(user_id: int):
    user_memories.pop(user_id, None)
    print(f"🧹 Cleared memory for user {user_id}")

def get_memory_info(user_id: int) -> Dict[str, Any]:
    if user_id not in user_memories:
        return {"error": "No memory found"}
    m = user_memories[user_id]
    return {
        "messages": len(m.recent_messages),
        "language": m.language,
        "preferences": m.preferences,
        "has_summary": bool(m.summary),
    }

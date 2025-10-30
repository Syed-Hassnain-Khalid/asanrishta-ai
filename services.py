import os
import json
from dotenv import load_dotenv
from typing import Dict, Any
from langchain.agents import initialize_agent, AgentType
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.utilities import SQLDatabase
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from app.models import ConversationMemory  # Django model for chat logs

# === ENVIRONMENT ===
load_dotenv()
AI_API_KEY = os.getenv("AI_API_KEY")
DB_URL = os.getenv("DB_URL")

# === LLM (Gemini) ===
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0.7,
    google_api_key=AI_API_KEY,
)

# === SQL TOOLKIT + AGENT ===
db = SQLDatabase.from_uri(DB_URL)
toolkit = SQLDatabaseToolkit(db=db, llm=llm)
sql_agent = initialize_agent(
    tools=toolkit.get_tools(),
    llm=llm,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=False,
    handle_parsing_errors=True,
    max_iterations=6,
    max_execution_time=30,
)

# === PROMPTS ===
UNIFIED_PROMPT = """
You are **RishtaMate AI**, a friendly Islamic matchmaking buddy on AsanRishta 💕—guiding with the light of deen and sunnah, like a wise confidant in faith and love.
Always weave in Islamic warmth: Start or end with a gentle dua, reminder of sabr/taqwa, or nod to nikkah's barakah. Speak of partners through Quran/Sunnah lens—deen first, akhlaq shining, family harmony blessed by Allah.
Talk cheerfully like a dost: Mix English + Roman Urdu naturally, keep it halal, heartfelt, and concise.

💞 Roles (Infuse Islam everywhere):
- If user seeks rishtas (city, age, gender, cast, etc.): 
  → Query SQL database (table 'users', columns: id, first_name, last_name, gender, age, city, cast, height, education, occupation, marital_status, is_active). Show only is_active=1.
  → Suggest 3–5 profiles briefly: "Ali, 27, Lahore — Engineer, BSc, masjid-goer with strong imaan."
  → Follow with: "InshaAllah, ye barakah laayein? Deen-compatible filters add karein? Allah aapko hifazat de."
- If casual chat, advice, or romantic vibes:
  → Short (under 3 lines), playful yet pious—e.g., "Sabr se dil kholo, Allah behtar dega! 💚"
  → For partner qualities (e.g., "I'm a developer—qualities in soulmate?"): Root in Islam: Prioritize taqwa, salaat-punctual, kind akhlaq, shared deen goals, family respect. Like: "Developer bhai, seek one with imaan strong jaise Surah An-Nur kehti—prayerful, honest, your deen partner in jannah journey. Kya khayal hai?"

Stay real, concise, and blessed—remember their journey with rahmah!
🧠 Context: {context_summary}
🗣️ User: {query}
Reply naturally, lively, and Islamically infused—short and from the heart:
"""

SUMMARY_PROMPT = """
Summarize the user's messages briefly:
- What kind of matches or goals they mentioned
- Any age, city, or cast preferences
- Their vibe (English or Roman Urdu)
Keep under 40 words.

User messages:
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

    def add_exchange(self, user_input: str, ai_response: str):
        self.recent_messages.append({"user": user_input, "ai": ai_response})
        self.recent_messages = self.recent_messages[-3:]  # keep last 3
        self.language = self._detect_language(user_input)

    def _detect_language(self, text: str) -> str:
        roman_urdu_keywords = ['aap', 'hai', 'mein', 'ka', 'ki', 'rishta', 'batao', 'chahiye']
        hits = sum(w in text.lower() for w in roman_urdu_keywords)
        return "roman_urdu" if hits >= 2 else "english"

    def get_summary(self, llm) -> str:
        """Generate or reuse summary."""
        if self.summary and len(self.recent_messages) < 5:
            return self.summary
        try:
            msgs = [m["user"] for m in self.recent_messages[-5:]]
            text = "\n".join([f"User: {m}" for m in msgs])
            chain = LLMChain(llm=llm, prompt=PromptTemplate(
                input_variables=["messages"], template=SUMMARY_PROMPT))
            result = chain.invoke({"messages": text})
            self.summary = result["text"].strip()
        except Exception as e:
            print("Summary error:", e)
            self.summary = "Fresh start — new chat!"
        return self.summary

    def get_context(self) -> str:
        """Return compact summary + last messages."""
        parts = []
        if self.summary:
            parts.append(f"Summary: {self.summary}")
        if self.recent_messages:
            last_users = [m["user"][:60] for m in self.recent_messages[-2:]]
            last = " | ".join(f"U:{lu}" for lu in last_users)
            parts.append(f"Recent: {last}")
        return " • ".join(parts) or "New chat spark!"

# === MEMORY CACHE (RAM + DB restore) ===
user_memories: Dict[int, EfficientMemoryManager] = {}

def get_memory(user_id: int) -> EfficientMemoryManager:
    """Load memory from cache or DB."""
    if user_id in user_memories:
        return user_memories[user_id]

    memory = EfficientMemoryManager(user_id)
    # load last 5 messages from DB
    past = ConversationMemory.objects.filter(user_id=user_id).order_by("-id")[:5]
    for item in reversed(past):
        memory.recent_messages.append({"user": item.user_input, "ai": item.ai_response})
    user_memories[user_id] = memory
    return memory

# === MAIN EXECUTION ===
def execute_query(user, query: str) -> str:
    try:
        memory = get_memory(user.id)
        context = memory.get_context()
        prompt = UNIFIED_PROMPT.format(context_summary=context, query=query)

        # Optional: print estimated token size
        print(f"⚙️ Token estimate: {len(prompt.split())} words")

        print(f"\n--- Routing ---\nUser: {user.id} | Lang: {memory.language}\nPrompt prepared.\n")

        result = sql_agent.invoke({"input": prompt})

        if isinstance(result, dict):
            response = result.get("output") or result.get("text") or str(result)
        else:
            response = str(result)

        # Fallback for empty reply
        if not response.strip():
            response = "Hmm... samajh nahi aaya 😅. Dobara batao?"

        # Update memory
        memory.add_exchange(query, response)
        ConversationMemory.objects.create(user=user, user_input=query, ai_response=response)

        # Occasionally update summary
        if len(memory.recent_messages) >= 3:
            memory.get_summary(llm)

        # Add light personality to responses
        if len(response) < 25:
            response += " 😊"

        return response.strip()

    except Exception as e:
        print("Execution error:", e)
        if any(w in query.lower() for w in ['aap', 'hai', 'mein', 'kya']):
            return "Oops, thoda hang ho gaya 😅. Dobara boliye?"
        return "System thoda busy hai, please try again."

# === UTILITIES ===
def clear_user_memory(user_id: int):
    user_memories.pop(user_id, None)
    print(f"🧹 Memory cleared for user {user_id}")

def get_memory_info(user_id: int) -> Dict[str, Any]:
    if user_id not in user_memories:
        return {"error": "No memory found"}
    m = user_memories[user_id]
    return {
        "messages": len(m.recent_messages),
        "language": m.language,
        "has_summary": bool(m.summary),
    }

import os
from dotenv import load_dotenv
from langchain.agents import initialize_agent, AgentType
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.utilities import SQLDatabase
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from app.models import ConversationMemory
from typing import Dict, Any
import json
import re

# === Load Environment ===
load_dotenv()
AI_API_KEY = os.getenv("AI_API_KEY")
DB_URL = os.getenv("DB_URL")

# === Setup LLM (Google Gemini) ===
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0.3,
    google_api_key=AI_API_KEY,
)

# === Setup SQL Database ===
db = SQLDatabase.from_uri(DB_URL)
toolkit = SQLDatabaseToolkit(db=db, llm=llm)

# === In-memory user summaries (lightweight) ===
user_summaries: Dict[int, Dict] = {}

# === SINGLE UNIFIED PROMPT (All-in-one) ===
UNIFIED_AI_PROMPT = """You are RishtaMate AI - a smart matchmaking assistant that can search database AND answer questions.

**DATABASE INFO:**
Table: users
Columns: id, first_name, last_name, email, gender, date_of_birth, religion, cast (Syed, Sheikh, Rajput, Awan, etc.), 
height (string: "5.5", "5.9", "6.0"), education, occupation, city, age, marital_status, userKaTaruf (bio), is_active
- Always filter: is_active=1
- Height comparison: Use LIKE or BETWEEN for ranges

**FAQ KNOWLEDGE:**
- Profile: Create with photos, bio, education, profession. Verify for trust.
- Matching: AI compatibility by age, cast, height, education, occupation, city.
- Communication: Send interest → both accept → chat/call (premium).
- Safety: Moderated profiles, report/block features, encrypted data.
- Plans: Free (basic), Premium (unlimited messaging, filters), Family packages.

**CONVERSATION CONTEXT:**
{context_summary}

**USER QUERY:** {query}

**INSTRUCTIONS:**
1. **Detect Intent:** Does user want DATABASE search (show/find profiles, cast/height/education filters) OR CONVERSATION (advice, FAQ, how-to)?
2. **Language Match:** User speaks {language} → respond in SAME language (English or Roman Urdu)
3. **Database Search Format:**
   ```
   Found X matches:
   1. Name - Age: X, Occupation: X, City: X, Height: X ft, Education: X
   
   💡 Tip: [helpful suggestion]
   ```
4. **Conversational Format:** Natural 2-3 sentences, helpful and warm
5. **Be Concise:** No unnecessary verbosity

**Your Response:**"""

# === SUMMARY GENERATOR (Compress 10 messages) ===
SUMMARY_PROMPT = """Summarize this conversation in 2-3 SHORT sentences. Focus on:
- User preferences (cast, height, education, location)
- Language style (English/Roman Urdu)
- Key requests made

Recent messages:
{messages}

Summary (2-3 sentences max):"""


class EfficientMemoryManager:
    """Ultra-lightweight memory manager that summarizes aggressively."""
    
    def __init__(self, user_id: int):
        self.user_id = user_id
        self.summary = ""
        self.recent_messages = []  # Keep last 10 only
        self.language = "english"
        self.preferences = {}  # Extract: cast, height, city preferences
    
    def add_exchange(self, user_input: str, ai_response: str):
        """Add exchange and keep only last 10 messages."""
        self.recent_messages.append({"user": user_input, "ai": ai_response})
        
        # Keep only last 10 messages
        if len(self.recent_messages) > 10:
            self.recent_messages = self.recent_messages[-10:]
        
        # Detect language from input
        self.language = self._detect_language(user_input)
        
        # Extract preferences
        self._extract_preferences(user_input)
    
    def _detect_language(self, text: str) -> str:
        """Quick language detection."""
        roman_urdu_markers = ['aap', 'hai', 'mein', 'ka', 'ki', 'dikhao', 'batao', 
                              'chahiye', 'kya', 'haan', 'nahi', 'rishta']
        text_lower = text.lower()
        
        matches = sum(1 for word in roman_urdu_markers if word in text_lower)
        return "roman_urdu" if matches >= 2 else "english"
    
    def _extract_preferences(self, text: str):
        """Extract key preferences from text."""
        text_lower = text.lower()
        
        # Extract cast mentions
        casts = ['syed', 'sheikh', 'rajput', 'awan', 'arain', 'jat', 'gujjar']
        for cast in casts:
            if cast in text_lower:
                self.preferences['cast'] = cast.capitalize()
        
        # Extract height mentions
        height_match = re.search(r'(\d\.\d+)', text)
        if height_match:
            self.preferences['height'] = height_match.group(1)
        
        # Extract city mentions
        cities = ['lahore', 'karachi', 'islamabad', 'rawalpindi', 'faisalabad']
        for city in cities:
            if city in text_lower:
                self.preferences['city'] = city.capitalize()
    
    def get_summary(self) -> str:
        """Get or create summary of last 10 messages."""
        if not self.recent_messages:
            return "New conversation started."
        
        # If we have existing summary and less than 5 new messages, use cached
        if self.summary and len(self.recent_messages) < 5:
            return self.summary
        
        # Create new summary
        try:
            messages_text = "\n".join([
                f"User: {m['user']}\nAI: {m['ai']}" 
                for m in self.recent_messages[-10:]  # Last 10 only
            ])
            
            summary_chain = LLMChain(
                llm=llm,
                prompt=PromptTemplate(
                    input_variables=["messages"],
                    template=SUMMARY_PROMPT
                )
            )
            
            self.summary = summary_chain.run({"messages": messages_text})
            return self.summary
            
        except Exception as e:
            print(f"❌ Summary error: {e}")
            return "Ongoing matchmaking conversation."
    
    def get_context_for_ai(self) -> str:
        """Get compact context string for AI prompt."""
        parts = []
        
        # Add summary if exists
        if self.summary:
            parts.append(f"Summary: {self.summary}")
        
        # Add last 3 exchanges only (ultra-compact)
        if len(self.recent_messages) >= 2:
            recent = self.recent_messages[-3:]
            recent_str = " | ".join([f"U:{m['user'][:50]}" for m in recent])
            parts.append(f"Recent: {recent_str}")
        
        # Add extracted preferences
        if self.preferences:
            pref_str = ", ".join([f"{k}={v}" for k, v in self.preferences.items()])
            parts.append(f"Preferences: {pref_str}")
        
        return " • ".join(parts) if parts else "New user"


def get_or_create_memory(user_id: int) -> EfficientMemoryManager:
    """Get or create memory manager."""
    if user_id not in user_summaries:
        user_summaries[user_id] = EfficientMemoryManager(user_id)
    return user_summaries[user_id]


def load_user_history(user) -> EfficientMemoryManager:
    """Load last 10 messages from DB and create summary."""
    memory = get_or_create_memory(user.id)
    
    # Load only last 10 messages from DB
    history = ConversationMemory.objects.filter(user=user).order_by("-created_at")[:10]
    history = list(reversed(history))
    
    for item in history:
        memory.add_exchange(item.user_input, item.ai_response)
    
    # Generate summary once
    if history:
        memory.get_summary()
    
    return memory


def is_database_query(query: str) -> bool:
    """Quick keyword-based detection (no AI call needed)."""
    query_lower = query.lower()
    
    # Database triggers
    db_keywords = [
        'show', 'find', 'search', 'dikhao', 'dhoondhna', 'dekhao',
        'profile', 'match', 'rishta', 'rishte',
        'cast', 'caste', 'syed', 'sheikh', 'rajput',
        'height', 'education', 'occupation', 'job',
        'city', 'lahore', 'karachi', 'islamabad',
        'my profile', 'mera profile', 'my name', 'mera naam',
        'list', 'kitne', 'how many', 'count'
    ]
    
    return any(kw in query_lower for kw in db_keywords)


def execute_unified_query(user, query: str, memory: EfficientMemoryManager) -> str:
    """Execute query with single unified prompt."""
    try:
        # Get compact context
        context = memory.get_context_for_ai()
        language = memory.language
        
        # Decide route (quick, no AI call)
        needs_database = is_database_query(query)
        
        print(f"\n{'='*50}")
        print(f"🎯 Query Type: {'DATABASE' if needs_database else 'CONVERSATIONAL'}")
        print(f"🌐 Language: {language.upper()}")
        print(f"📝 Context: {context[:100]}...")
        print(f"{'='*50}\n")
        
        if needs_database:
            # Use SQL agent with unified prompt
            agent = initialize_agent(
                tools=toolkit.get_tools(),
                llm=llm,
                agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                verbose=True,
                handle_parsing_errors=True,
                max_iterations=10,
                max_execution_time=60,
            )
            
            # Build final prompt
            final_prompt = UNIFIED_AI_PROMPT.format(
                context_summary=context,
                query=query,
                language=language.title()
            )
            
            response = agent.run(final_prompt)
        else:
            # Direct LLM call for conversational
            chain = LLMChain(
                llm=llm,
                prompt=PromptTemplate(
                    input_variables=["context_summary", "query", "language"],
                    template=UNIFIED_AI_PROMPT
                )
            )
            
            response = chain.run({
                "context_summary": context,
                "query": query,
                "language": language.title()
            })
        
        return response.strip()
        
    except Exception as e:
        print(f"❌ Query execution error: {str(e)}")
        
        if memory.language == "roman_urdu":
            return "Maaf karna, kuch issue aa gaya. Aap apni query dobara try karein?"
        else:
            return "Sorry, I encountered an issue. Could you please rephrase your query?"


def ai_router(user, query: str) -> str:
    """
    Optimized AI Router - Minimal token usage with maximum efficiency.
    
    Token-saving features:
    - Summarizes last 10 messages into 2-3 sentences
    - Single unified prompt (no multiple AI calls for routing)
    - Keyword-based routing (no AI needed)
    - Keeps only last 10 messages in memory
    - Extracts preferences automatically
    """
    try:
        # Load/create user memory
        memory = load_user_history(user)
        
        # Add current query to memory
        # (response will be added after generation)
        
        # Execute with unified prompt
        response = execute_unified_query(user, query, memory)
        
        # Update memory with exchange
        memory.add_exchange(query, response)
        
        # Save to database
        ConversationMemory.objects.create(
            user=user,
            user_input=query,
            ai_response=response,
        )
        
        # Auto-summarize if we hit 10 messages
        if len(memory.recent_messages) >= 10:
            print(f"📝 Auto-summarizing (reached 10 messages)")
            memory.get_summary()
        
        return response
        
    except Exception as e:
        print(f"❌ Router error: {str(e)}")
        
        # Quick language detection for fallback
        is_urdu = any(word in query.lower() for word in ['aap', 'mein', 'hai', 'kya'])
        
        if is_urdu:
            return "Maaf karna, kuch technical issue ho gaya. Kya aap apni query dobara bata sakte hain?"
        else:
            return "I apologize for the technical issue. Could you please try asking again?"


def clear_user_memory(user_id: int):
    """Clear memory for a user."""
    if user_id in user_summaries:
        del user_summaries[user_id]
        print(f"🗑️  Cleared memory for user {user_id}")


def get_memory_stats(user_id: int) -> Dict:
    """Get memory statistics."""
    if user_id in user_summaries:
        memory = user_summaries[user_id]
        return {
            "user_id": user_id,
            "message_count": len(memory.recent_messages),
            "language": memory.language,
            "has_summary": bool(memory.summary),
            "preferences": memory.preferences,
            "summary_preview": memory.summary[:100] if memory.summary else None
        }
    return {"error": "No memory found"}


def regenerate_summary(user_id: int) -> str:
    """Force regenerate summary for debugging."""
    if user_id in user_summaries:
        memory = user_summaries[user_id]
        memory.summary = ""  # Clear cache
        return memory.get_summary()
    return "No memory found"
# import os
# from dotenv import load_dotenv
# from langchain.agents import initialize_agent, AgentType
# from langchain.agents.agent_toolkits import SQLDatabaseToolkit
# from langchain.utilities import SQLDatabase
# from langchain.memory import ConversationBufferMemory, ConversationSummaryMemory
# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain.chains import LLMChain
# from langchain.prompts import PromptTemplate
# from app.models import ConversationMemory
# from typing import Dict, Any, List
# import json

# # === Load Environment ===
# load_dotenv()
# AI_API_KEY = os.getenv("AI_API_KEY")
# DB_URL = os.getenv("DB_URL")

# # === Setup LLM (Google Gemini) ===
# llm = ChatGoogleGenerativeAI(
#     model="gemini-2.0-flash",
#     temperature=0.4,
#     verbose=True,
#     google_api_key=AI_API_KEY,
# )

# # === Setup SQL Database ===
# db = SQLDatabase.from_uri(DB_URL)
# toolkit = SQLDatabaseToolkit(db=db, llm=llm)

# # === User-specific memory storage ===
# user_memories: Dict[int, Dict[str, Any]] = {}

# # === DYNAMIC ROUTING PROMPT (AI Decides Everything) ===
# INTELLIGENT_ROUTER_PROMPT = """You are an intelligent router for a Rishta (matchmaking) platform AI assistant.

# Analyze the user's query and decide whether it requires DATABASE access or can be answered with CONVERSATIONAL knowledge.

# **Choose DATABASE if:**
# - User wants to find, search, show, list, or filter profiles/matches/rishte
# - User asks about specific people, profiles, or their own profile details (name, age, etc.)
# - Query mentions specific attributes: cast/caste, height, education, occupation, city, age range
# - User asks for statistics, counts, or data analysis
# - Query contains: "show me", "find me", "search for", "how many", "list all", "my profile", "my name"
# - Examples: "syed developer lahore", "show rishte", "my profile details", "engineers in karachi"

# **Choose CONVERSATIONAL if:**
# - General questions about platform features, how it works
# - Relationship advice, compatibility tips, cultural guidance
# - Profile creation help, tips for writing bio
# - FAQs about messaging, verification, privacy
# - Greetings, casual chat, general discussion
# - Examples: "how does matching work", "tips for good profile", "hello", "what is this app"

# **Language Detection:**
# - Detect if user is speaking English, Roman Urdu, or mixed
# - Note the language style for the response

# **User Query:** "{query}"

# **Recent Context:** {context}

# Respond in JSON format:
# {{
#   "route": "database" or "conversational",
#   "confidence": 0-100,
#   "detected_language": "english" or "roman_urdu" or "mixed",
#   "reasoning": "brief explanation",
#   "key_entities": ["list", "of", "important", "keywords"]
# }}"""

# # === CONVERSATION SUMMARIZER ===
# CONVERSATION_SUMMARIZER_PROMPT = """Summarize this conversation concisely in 3-4 sentences. Focus on:
# 1. User's preferences (cast, height, education, occupation, location)
# 2. Language preference (English/Roman Urdu)
# 3. Key requests made and information provided
# 4. User's current state in the conversation

# Conversation:
# {history}

# Concise Summary (3-4 sentences):"""

# # === FAQ KNOWLEDGE BASE ===
# RISHTA_FAQ = """
# **Rishta Platform Knowledge Base:**

# **Profile Management:**
# - Create profiles with photos, bio, education, profession, family details
# - Profile verification increases trust and visibility
# - Privacy settings control who sees your information
# - Edit profile anytime to update information

# **Matching System:**
# - AI-powered compatibility matching based on preferences
# - Filter by: age, cast, height, education, occupation, city, marital status
# - Save favorite profiles for later review
# - Get daily match recommendations

# **Communication:**
# - Send interest requests to profiles you like
# - Chat after both parties accept interest
# - Video/voice calls for premium members
# - Respect and privacy in all communications

# **Safety & Privacy:**
# - All profiles are moderated by team
# - Report inappropriate behavior instantly
# - Block unwanted contacts
# - Your data is encrypted and secure

# **Subscription Plans:**
# - Free: Basic profile, limited searches
# - Premium: Unlimited messaging, advanced filters, priority support
# - Family packages available for multiple profiles
# """

# # === SQL AGENT CUSTOM PROMPT ===
# SQL_AGENT_SYSTEM_PROMPT = """You are a helpful database assistant for a Rishta matchmaking platform.

# **Your Capabilities:**
# - Search profiles by cast, height, education, occupation, city, age
# - Retrieve user's own profile information
# - Provide match statistics and insights
# - Answer queries about database content

# **Database Schema:**
# Table: users
# Columns: id, first_name, last_name, email, gender, date_of_birth, religion, cast, height (string like "5.5", "5.9"), 
# education, occupation, city (integer), age, marital_status, userKaTaruf (bio), status

# **Important Rules:**
# 1. Height is stored as string (e.g., "5.5", "5.9", "6.0") - use LIKE or exact matching
# 2. For height ranges, use: WHERE height BETWEEN '5.4' AND '5.6' OR height LIKE '5.5%'
# 3. Cast names: Syed, Sheikh, Rajput, Awan, etc. (case-sensitive, use exact match)
# 4. Always filter by is_active=1 to show only active profiles
# 5. Respect privacy - don't expose emails or phone numbers without context

# **Response Format:**
# Always format results as:
# ```
# Found [X] matches:

# 1. **Name** - Age: X, Occupation: X, City: X, Height: X ft, Education: X
# 2. **Name** - Age: X, Occupation: X, City: X, Height: X ft, Education: X

# 💡 Suggestions: [helpful next steps or advice]
# ```

# **Language Adaptation:**
# - If context indicates Roman Urdu, use Roman Urdu in your response
# - If English, use English
# - Be natural and conversational

# **Context:** {context}
# **User Query:** {query}

# Proceed with database search:"""

# # === CONVERSATIONAL AGENT PROMPT ===
# CONVERSATIONAL_AGENT_PROMPT = """You are RishtaMate AI - a warm, intelligent assistant for a Rishta matchmaking platform.

# **Your Personality:**
# - Friendly, respectful, and culturally aware
# - Empathetic and supportive in relationship matters
# - Professional yet approachable
# - Knowledgeable about South Asian matchmaking culture

# **Your Role:**
# - Answer FAQs about the platform
# - Provide relationship and profile advice
# - Guide users through matchmaking process
# - Offer compatibility insights
# - Help with profile creation tips

# **CRITICAL LANGUAGE RULE:**
# - User's detected language: {language}
# - If Roman Urdu: Respond ONLY in Roman Urdu (e.g., "Aap ka profile bohat acha hai")
# - If English: Respond ONLY in English
# - If Mixed: Follow user's dominant language style
# - NEVER mix languages unless user does

# **Knowledge Base:**
# {faq_content}

# **Conversation Summary:**
# {summary}

# **Recent Messages:**
# {recent_history}

# **Current User Query:** {query}

# **Instructions:**
# 1. Answer naturally in the user's language
# 2. Be concise but helpful (2-4 sentences for simple queries)
# 3. If you sense user wants database search, suggest they ask specifically
# 4. Don't make assumptions about database content
# 5. Stay in character as a matchmaking platform assistant

# Your Response:"""


# class IntelligentMemoryManager:
#     """Manages conversation memory with smart summarization."""
    
#     def __init__(self, user_id: int):
#         self.user_id = user_id
#         self.buffer = ConversationBufferMemory(
#             memory_key="chat_history",
#             return_messages=True,
#             input_key="input",
#             output_key="output"
#         )
#         self.summary = ""
#         self.message_count = 0
#         self.detected_language = "english"
    
#     def add_exchange(self, user_input: str, ai_response: str):
#         """Add a conversation exchange to memory."""
#         self.buffer.save_context({"input": user_input}, {"output": ai_response})
#         self.message_count += 1
    
#     def get_recent_history(self, n: int = 10) -> str:
#         """Get last N messages as formatted string."""
#         messages = self.buffer.chat_memory.messages[-n:]
#         history = []
#         for i, msg in enumerate(messages):
#             role = "User" if i % 2 == 0 else "AI"
#             history.append(f"{role}: {msg.content}")
#         return "\n".join(history)
    
#     def should_summarize(self) -> bool:
#         """Check if we should create a new summary."""
#         return self.message_count % 15 == 0 and self.message_count > 0
    
#     def create_summary(self) -> str:
#         """Create a summary of recent conversation."""
#         recent = self.get_recent_history(15)
#         if not recent.strip():
#             return "New conversation started."
        
#         try:
#             summary_chain = LLMChain(
#                 llm=llm,
#                 prompt=PromptTemplate(
#                     input_variables=["history"],
#                     template=CONVERSATION_SUMMARIZER_PROMPT
#                 )
#             )
#             self.summary = summary_chain.run({"history": recent})
#             return self.summary
#         except Exception as e:
#             print(f"❌ Summary creation failed: {str(e)}")
#             return self.summary or "Ongoing conversation about matchmaking."
    
#     def get_context_window(self) -> Dict[str, str]:
#         """Get optimized context for AI (summary + recent messages)."""
#         if self.should_summarize():
#             self.create_summary()
        
#         return {
#             "summary": self.summary or "New conversation.",
#             "recent_history": self.get_recent_history(6),  # Last 6 messages
#             "message_count": self.message_count,
#             "language": self.detected_language
#         }


# def get_or_create_memory_manager(user_id: int) -> IntelligentMemoryManager:
#     """Get or create memory manager for a user."""
#     if user_id not in user_memories:
#         user_memories[user_id] = IntelligentMemoryManager(user_id)
#     return user_memories[user_id]


# def load_user_history_from_db(user) -> IntelligentMemoryManager:
#     """Load conversation history from database and initialize memory."""
#     memory_manager = get_or_create_memory_manager(user.id)
    
#     # Load last 30 messages from DB
#     history = ConversationMemory.objects.filter(user=user).order_by("-created_at")[:30]
#     history = list(reversed(history))
    
#     for item in history:
#         memory_manager.add_exchange(item.user_input, item.ai_response)
    
#     # Create initial summary if there's history
#     if history:
#         memory_manager.create_summary()
    
#     return memory_manager


# def detect_language_style(text: str) -> str:
#     """Simple language detection."""
#     # Roman Urdu indicators
#     roman_urdu_words = ['aap', 'hai', 'hain', 'ka', 'ki', 'ko', 'mein', 'se', 
#                         'kya', 'nahi', 'haan', 'chahiye', 'dijiye', 'karo', 
#                         'dikhao', 'batao', 'mera', 'mere', 'rishta', 'rishte']
    
#     text_lower = text.lower()
#     roman_count = sum(1 for word in roman_urdu_words if word in text_lower)
    
#     if roman_count >= 2:
#         return "roman_urdu"
#     return "english"


# def intelligent_route_decision(query: str, context: Dict) -> Dict:
#     """AI-powered routing decision with full context awareness."""
#     try:
#         router_chain = LLMChain(
#             llm=llm,
#             prompt=PromptTemplate(
#                 input_variables=["query", "context"],
#                 template=INTELLIGENT_ROUTER_PROMPT
#             )
#         )
        
#         context_str = f"Summary: {context.get('summary', 'None')}\nRecent: {context.get('recent_history', 'None')[:200]}"
        
#         result = router_chain.run({"query": query, "context": context_str})
        
#         # Parse JSON response
#         # Clean up result if it has markdown code blocks
#         if "```json" in result:
#             result = result.split("```json")[1].split("```")[0].strip()
#         elif "```" in result:
#             result = result.split("```")[1].split("```")[0].strip()
        
#         decision = json.loads(result.strip())
        
#         print(f"\n{'='*60}")
#         print(f"🤖 INTELLIGENT ROUTING DECISION")
#         print(f"{'='*60}")
#         print(f"Query: {query}")
#         print(f"Route: {decision['route'].upper()}")
#         print(f"Confidence: {decision['confidence']}%")
#         print(f"Language: {decision['detected_language']}")
#         print(f"Reasoning: {decision['reasoning']}")
#         print(f"Key Entities: {decision['key_entities']}")
#         print(f"{'='*60}\n")
        
#         return decision
        
#     except Exception as e:
#         print(f"❌ Routing error: {str(e)}")
#         # Fallback to simple keyword detection
#         query_lower = query.lower()
#         db_keywords = ['show', 'find', 'search', 'profile', 'match', 'rishta', 
#                        'dikhao', 'cast', 'height', 'education', 'my name']
        
#         is_db = any(kw in query_lower for kw in db_keywords)
        
#         return {
#             "route": "database" if is_db else "conversational",
#             "confidence": 60,
#             "detected_language": detect_language_style(query),
#             "reasoning": "Fallback keyword matching",
#             "key_entities": []
#         }


# def execute_database_query(user, query: str, context: Dict, language: str) -> str:
#     """Execute database query with SQL agent."""
#     try:
#         print(f"📊 Executing DATABASE query...")
        
#         # Create agent with memory
#         memory_manager = get_or_create_memory_manager(user.id)
#         agent = initialize_agent(
#             tools=toolkit.get_tools(),
#             llm=llm,
#             agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
#             memory=memory_manager.buffer,
#             verbose=True,
#             handle_parsing_errors=True,
#             max_iterations=15,
#             max_execution_time=120,
#         )
        
#         # Build context-aware prompt
#         context_info = f"""
# Language Mode: {language}
# Conversation Summary: {context.get('summary', 'New conversation')}
# User ID: {user.id}
# """
        
#         enhanced_prompt = SQL_AGENT_SYSTEM_PROMPT.format(
#             context=context_info,
#             query=query
#         )
        
#         response = agent.run(enhanced_prompt)
        
#         # Language adaptation post-processing
#         if language == "roman_urdu" and not any(word in response.lower() for word in ['aap', 'hai', 'mein']):
#             # If response is in English but should be Roman Urdu, add a note
#             response = f"{response}\n\n💬 Aap Roman Urdu mein bhi puch sakte hain!"
        
#         return response.strip()
        
#     except Exception as e:
#         print(f"❌ Database query error: {str(e)}")
        
#         if language == "roman_urdu":
#             return "Database mein thori issue hai. Kya aap apni query phir se specific tareeke se bata sakte hain? Jaise: cast, height, education, occupation."
#         else:
#             return "I encountered an issue accessing the database. Could you rephrase your query with specific criteria like cast, height, education, or occupation?"


# def execute_conversational_response(query: str, context: Dict, language: str) -> str:
#     """Generate conversational response using FAQ knowledge."""
#     try:
#         print(f"💬 Generating CONVERSATIONAL response...")
        
#         conversational_chain = LLMChain(
#             llm=llm,
#             prompt=PromptTemplate(
#                 input_variables=["query", "faq_content", "summary", "recent_history", "language"],
#                 template=CONVERSATIONAL_AGENT_PROMPT
#             )
#         )
        
#         response = conversational_chain.run({
#             "query": query,
#             "faq_content": RISHTA_FAQ,
#             "summary": context.get('summary', 'New conversation'),
#             "recent_history": context.get('recent_history', 'No prior messages'),
#             "language": language
#         })
        
#         return response.strip()
        
#     except Exception as e:
#         print(f"❌ Conversational error: {str(e)}")
        
#         if language == "roman_urdu":
#             return "Main yahan aap ki madad karne ke liye hoon! Aap mujhse platform ke features, profile tips, ya match search ke baare mein puch sakte hain."
#         else:
#             return "I'm here to help! You can ask me about platform features, profile tips, or search for matches. What would you like to know?"


# def ai_router(user, query: str) -> str:
#     """
#     Intelligent AI Router with dynamic decision making and memory management.
    
#     Features:
#     - AI decides routing (database vs conversational)
#     - Maintains conversation memory with smart summarization
#     - Adapts to user's language (English/Roman Urdu)
#     - Optimizes token usage with context windows
#     """
#     try:
#         # Load user's conversation history and memory
#         memory_manager = load_user_history_from_db(user)
        
#         # Get optimized context window
#         context = memory_manager.get_context_window()
        
#         # Detect language style
#         detected_language = detect_language_style(query)
#         memory_manager.detected_language = detected_language
        
#         # AI-powered routing decision
#         routing_decision = intelligent_route_decision(query, context)
        
#         route = routing_decision['route']
#         language = routing_decision['detected_language']
        
#         # Execute based on route
#         if route == "database":
#             response = execute_database_query(user, query, context, language)
#         else:
#             response = execute_conversational_response(query, context, language)
        
#         # Update memory
#         memory_manager.add_exchange(query, response)
        
#         # Save to database
#         save_conversation(user, query, response)
        
#         # Auto-summarize if needed
#         if memory_manager.should_summarize():
#             print(f"📝 Auto-summarizing conversation (message #{memory_manager.message_count})")
#             memory_manager.create_summary()
        
#         return response
        
#     except Exception as e:
#         error_msg = f"Critical error in ai_router: {str(e)}"
#         print(f"❌ {error_msg}")
        
#         # Intelligent fallback based on query
#         detected_lang = detect_language_style(query)
        
#         if detected_lang == "roman_urdu":
#             fallback = "Maaf karna, kuch technical issue ho gaya. Main dobara try kar raha hoon. Aap mujhe batayein aap kya dhoondhna chahte hain?"
#         else:
#             fallback = "I apologize for the technical issue. I'm here to help you find matches or answer questions. What can I assist you with?"
        
#         # Still save the conversation
#         try:
#             save_conversation(user, query, fallback)
#         except:
#             pass
        
#         return fallback


# def save_conversation(user, query: str, response: str):
#     """Save conversation to database."""
#     try:
#         ConversationMemory.objects.create(
#             user=user,
#             user_input=query,
#             ai_response=response,
#         )
#     except Exception as e:
#         print(f"❌ Failed to save conversation: {str(e)}")


# def clear_user_memory(user_id: int):
#     """Clear memory for a specific user."""
#     if user_id in user_memories:
#         del user_memories[user_id]
#         print(f"🗑️  Cleared memory for user {user_id}")


# def get_memory_stats(user_id: int) -> Dict:
#     """Get memory statistics for debugging."""
#     if user_id in user_memories:
#         manager = user_memories[user_id]
#         return {
#             "user_id": user_id,
#             "message_count": manager.message_count,
#             "has_summary": bool(manager.summary),
#             "detected_language": manager.detected_language,
#             "buffer_size": len(manager.buffer.chat_memory.messages)
#         }
#     return {"error": "No memory found for user"}
# import os
# from dotenv import load_dotenv
# from langchain.agents import initialize_agent, AgentType
# from langchain.agents.agent_toolkits import SQLDatabaseToolkit
# from langchain.utilities import SQLDatabase
# from langchain.memory import ConversationBufferMemory
# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain.chains import LLMChain
# from langchain.prompts import PromptTemplate
# from app.models import ConversationMemory
# from typing import Dict, Any
# import re

# # === Load Environment ===
# load_dotenv()
# AI_API_KEY = os.getenv("AI_API_KEY")
# DB_URL = os.getenv("DB_URL")

# # === Setup LLM (Google Gemini) ===
# llm = ChatGoogleGenerativeAI(
#     model="gemini-2.0-flash",
#     temperature=0.3,
#     verbose=True,
#     google_api_key=AI_API_KEY,
#     timeout=60,  # Increased timeout for LLM calls
# )

# # === Setup SQL Database ===
# db = SQLDatabase.from_uri(DB_URL)
# toolkit = SQLDatabaseToolkit(db=db, llm=llm)

# # === User-specific memory storage ===
# user_memories: Dict[int, Dict[str, Any]] = {}

# # === Summarization Prompt ===
# summary_prompt = PromptTemplate(
#     input_variables=["history"],
#     template="""Summarize the following conversation concisely, focusing on:
# - Key user preferences and information
# - Important topics discussed
# - Any database queries or results
# - User's relationship goals or requirements

# Conversation:
# {history}

# Summary:"""
# )

# # === FAQ Content for Rishta App ===
# RISHTA_FAQ = """
# Common Questions about Our Rishta Platform:

# **Profile & Registration:**
# - Create detailed profiles with photos, bio, education, profession, and preferences
# - Verify your profile for increased trust
# - Privacy controls to show/hide information

# **Matching & Search:**
# - Advanced filters: age, location, education, profession, religion, sect
# - AI-powered recommendations based on compatibility
# - Save favorite profiles and send interest

# **Communication:**
# - Send interest requests to profiles you like
# - Chat with matched profiles after mutual acceptance
# - Video call feature for verified members

# **Safety & Privacy:**
# - All profiles are moderated
# - Report and block features available
# - Your information is encrypted and secure
# - Control who can see your profile

# **Subscription:**
# - Free basic membership with limited features
# - Premium plans for unlimited messaging and advanced search
# - Special family packages available
# """

# # === System Prompts ===
# SQL_PROMPT = """You are a helpful AI assistant for a Rishta (marriage/matchmaking) platform.
# You can access the database to help users with:
# - Finding matches based on criteria
# - Viewing profile statistics
# - Checking their received/sent interests
# - Analyzing compatibility data

# IMPORTANT INSTRUCTIONS:
# 1. For height queries, understand that heights like "5.5" mean 5 feet 5 inches, stored as strings like "5.5", "5.6", "5.9" etc.
# 2. Use LIKE or BETWEEN comparisons for height ranges
# 3. Format ALL results as numbered lists with: Name, Age, Occupation, City
# 4. ALWAYS complete your response - never stop mid-thought
# 5. If no exact matches, show close matches and explain differences

# Current conversation context: {summary}

# User query: {input}"""

# CHAT_PROMPT = """
# You are RishtaMate AI — a caring, respectful, and intelligent assistant for a Rishta (matchmaking) platform.

# Your role:
# - Help users find compatible matches and guide them through the matchmaking process
# - Answer FAQs about profiles, interests, verification, and communication features
# - Explain how AI recommendations and matching algorithms work
# - Give relationship and compatibility guidance with empathy and cultural sensitivity
# - Support users in creating honest and appealing profiles
# - Maintain privacy, respect, and professionalism at all times

# IMPORTANT: 
# - If user asks for matches/rishte, tell them I can search the database for them
# - If user asks personal questions (name, profile), tell them I can look up their profile in the database
# - Respond in the same language as the user (English, Roman Urdu, or Urdu)

# Available Information:
# {faq_content}

# Conversation Summary:
# {summary}

# Recent Conversation:
# {history}

# User: {user_input}
# AI:"""

# # === Enhanced Routing Decision Prompt ===
# ROUTING_PROMPT = """
# Analyze this user query and determine if it requires DATABASE access or CHAT response.

# DATABASE access needed for:
# - Finding/showing/listing profiles, matches, rishte (e.g., "show me rishta", "syed developer", "mera match", "my profile", "my name")
# - Asking about user's own data (name, age, profile details, interests)
# - Statistics or counts (e.g., "how many profiles", "received interests")
# - Filtering by criteria (cast, height, education, city, occupation)
# - ANY query with specific attributes like "5.5 height", "Lahore", "engineer"

# CHAT response sufficient for:
# - General FAQs about platform features
# - Relationship advice
# - How to use the app
# - Greetings without specific requests

# Query: "{query}"

# Respond with ONLY: DATABASE or CHAT"""

# def get_or_create_memory(user_id: int) -> Dict[str, Any]:
#     """Get or create memory objects for a specific user."""
#     if user_id not in user_memories:
#         user_memories[user_id] = {
#             'buffer': ConversationBufferMemory(
#                 memory_key="chat_history",
#                 return_messages=True,
#                 input_key="input",
#                 output_key="output"
#             ),
#             'summary': ""
#         }
#     return user_memories[user_id]


# def load_user_memory(user):
#     """
#     Loads previous conversation history from DB and creates a summary.
#     """
#     memory_obj = get_or_create_memory(user.id)
#     buffer_memory = memory_obj['buffer']
    
#     # Load past conversations
#     history = ConversationMemory.objects.filter(user=user).order_by("-created_at")[:50]
#     history = reversed(history)
    
#     chat_text = ""
#     for item in history:
#         buffer_memory.chat_memory.add_user_message(item.user_input)
#         buffer_memory.chat_memory.add_ai_message(item.ai_response)
#         chat_text += f"User: {item.user_input}\nAI: {item.ai_response}\n\n"
    
#     # Generate summary if there's history
#     if chat_text.strip():
#         try:
#             summary_chain = LLMChain(llm=llm, prompt=summary_prompt)
#             summary = summary_chain.run({"history": chat_text})
#             memory_obj['summary'] = summary
#         except Exception as e:
#             print(f"Summary generation error: {str(e)}")
    
#     return memory_obj


# def save_conversation(user, query: str, response: str):
#     """Save user query and AI response in the database."""
#     ConversationMemory.objects.create(
#         user=user,
#         user_input=query,
#         ai_response=response,
#     )


# def decide_routing(query: str) -> str:
#     """Use LLM to intelligently decide routing with enhanced keyword fallback."""
    
#     # First, check for strong database indicators
#     db_strong_keywords = [
#         # Profile/match requests
#         r'\b(show|find|search|list|display|get)\b.*\b(profile|match|rishta|rishte)\b',
#         # Personal info requests
#         r'\b(my|mera|mere)\b.*\b(name|profile|age|details|match|interest)\b',
#         # Specific attributes
#         r'\b(syed|height|education|occupation|developer|engineer|doctor)\b',
#         r'\b\d+\.?\d*\s*(feet|ft|height)\b',
#         # Database actions
#         r'\b(database|query|sql|table)\b',
#         # Location + attributes
#         r'\b(lahore|karachi|islamabad|city)\b.*\b(engineer|developer|doctor)\b'
#     ]
    
#     query_lower = query.lower()
#     for pattern in db_strong_keywords:
#         if re.search(pattern, query_lower):
#             print(f"✓ DATABASE route (regex match): {pattern}")
#             return "database"
    
#     # Use LLM as secondary check
#     try:
#         routing_chain = LLMChain(
#             llm=llm,
#             prompt=PromptTemplate(input_variables=["query"], template=ROUTING_PROMPT)
#         )
#         decision = routing_chain.run({"query": query}).strip().upper()
#         print(f"LLM routing decision: {decision}")
#         if "DATABASE" in decision:
#             return "database"
#     except Exception as e:
#         print(f"Routing LLM error: {str(e)}")
    
#     # Fallback keywords
#     db_keywords = [
#         "find", "search", "show", "list", "filter", "match", "profile", "rishta", "rishte",
#         "dikhao", "dikhaye", "dhoondho", "how many", "count", "my name", "mera naam",
#         "height", "education", "cast", "occupation", "city", "age"
#     ]
    
#     if any(kw in query_lower for kw in db_keywords):
#         print(f"✓ DATABASE route (keyword fallback)")
#         return "database"
    
#     print(f"✓ CHAT route")
#     return "chat"


# def create_sql_agent(memory):
#     """Create SQL agent with optimized settings."""
#     return initialize_agent(
#         tools=toolkit.get_tools(),
#         llm=llm,
#         agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
#         memory=memory,
#         verbose=True,
#         handle_parsing_errors=True,
#         max_iterations=15,  # Increased iterations
#         max_execution_time=90,  # 90 seconds max
#         early_stopping_method="generate",
#     )


# def ai_router(user, query: str) -> str:
#     """
#     Main routing function with intelligent decision making and memory.
#     """
#     try:
#         print(f"\n{'='*60}")
#         print(f"User {user.id} Query: {query}")
#         print(f"{'='*60}")
        
#         # Load user's memory
#         memory_obj = load_user_memory(user)
#         buffer_memory = memory_obj['buffer']
#         conversation_summary = memory_obj['summary']
        
#         # Decide routing
#         route = decide_routing(query)
#         print(f"→ Route Selected: {route.upper()}")
        
#         if route == "database":
#             # === SQL Agent Mode ===
#             sql_agent = create_sql_agent(buffer_memory)
            
#             # Enhanced prompt for better SQL generation
#             enhanced_sql_prompt = f"""
# You are querying a matchmaking database to help find compatible rishta profiles.

# DATABASE SCHEMA:
# - Table: users
# - Key columns: first_name, age, cast, height (string like "5.5", "5.9"), education, occupation, city, gender

# USER'S QUERY: {query}

# CONTEXT: {conversation_summary or "New conversation"}

# INSTRUCTIONS:
# 1. Parse the query carefully - "5.5 height" means height = "5.5" or LIKE "5.5%"
# 2. For height ranges, use: height BETWEEN "5.4" AND "5.6" OR height LIKE "5.5%"
# 3. Always check education IS NOT NULL if "educated" is mentioned
# 4. Format results as:
   
#    Found X matching profiles:
#    1. [Name] - Age: [age], Occupation: [occupation], City: [city], Height: [height]
#    2. [Name] - Age: [age], Occupation: [occupation], City: [city], Height: [height]
   
#    [Brief suggestion or next steps]

# 5. If no exact matches, show closest matches and explain the difference
# 6. NEVER stop mid-response - always complete the formatted list

# Execute the query now and format results properly.
# """
            
#             try:
#                 print("→ Running SQL Agent...")
#                 response = sql_agent.run(enhanced_sql_prompt)
#                 print(f"→ SQL Agent Response:\n{response}")
#             except Exception as e:
#                 print(f"SQL Agent Error: {str(e)}")
#                 # Fallback with helpful message
#                 response = f"I tried to search the database but encountered an issue. Let me try a simpler search. Could you specify: cast (e.g., Syed), height range (e.g., 5.4-5.6), education level, and occupation?"
        
#         else:
#             # === Chat Agent Mode ===
#             recent_messages = buffer_memory.chat_memory.messages[-10:]
#             history_text = "\n".join([
#                 f"{'User' if i % 2 == 0 else 'AI'}: {msg.content}" 
#                 for i, msg in enumerate(recent_messages)
#             ])
            
#             chat_prompt = PromptTemplate(
#                 input_variables=["faq_content", "summary", "history", "user_input"],
#                 template=CHAT_PROMPT
#             )
            
#             chat_agent = LLMChain(llm=llm, prompt=chat_prompt, verbose=True)
            
#             try:
#                 print("→ Running Chat Agent...")
#                 response = chat_agent.run({
#                     "faq_content": RISHTA_FAQ,
#                     "summary": conversation_summary or "New conversation",
#                     "history": history_text or "No recent messages",
#                     "user_input": query
#                 })
#                 print(f"→ Chat Agent Response:\n{response}")
#             except Exception as e:
#                 print(f"Chat Agent Error: {str(e)}")
#                 response = "I'm here to help you find the perfect match! You can ask me to search for profiles by cast, height, education, occupation, or city. What would you like to know?"
        
#         # Save to memory
#         buffer_memory.save_context({"input": query}, {"output": response})
#         save_conversation(user, query, response)
        
#         # Auto-summarize every 20 messages
#         message_count = len(buffer_memory.chat_memory.messages)
#         if message_count > 0 and message_count % 20 == 0:
#             full_history = "\n".join([
#                 f"{'User' if i % 2 == 0 else 'AI'}: {msg.content}"
#                 for i, msg in enumerate(buffer_memory.chat_memory.messages)
#             ])
            
#             try:
#                 summary_chain = LLMChain(llm=llm, prompt=summary_prompt)
#                 new_summary = summary_chain.run({"history": full_history})
#                 memory_obj['summary'] = new_summary
#             except Exception as e:
#                 print(f"Summary update error: {str(e)}")
        
#         return response.strip()
    
#     except Exception as e:
#         error_msg = f"Error in ai_router: {str(e)}"
#         print(error_msg)
        
#         # Smart fallback based on query
#         if any(word in query.lower() for word in ["find", "show", "rishta", "match", "profile"]):
#             fallback = "I can help you search for matches! Please provide specific criteria like:\n• Cast (e.g., Syed, Sheikh)\n• Height (e.g., 5.5, 5.6)\n• Education (e.g., BS, Masters)\n• Occupation (e.g., Developer, Engineer)\n• City\n\nWhat are you looking for?"
#         else:
#             fallback = "I'm RishtaMate AI, here to help you find compatible matches. You can ask me to search profiles or answer questions about the platform. How can I assist you?"
        
#         save_conversation(user, query, fallback)
#         return fallback


# def clear_user_memory(user_id: int):
#     """Clear memory for a specific user (useful for logout/cleanup)."""
#     if user_id in user_memories:
#         del user_memories[user_id]

# import os
# from dotenv import load_dotenv
# from langchain.agents import initialize_agent, AgentType
# from langchain.agents.agent_toolkits import SQLDatabaseToolkit
# from langchain.utilities import SQLDatabase
# from langchain.memory import ConversationBufferMemory
# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain.chains import LLMChain
# from langchain.prompts import PromptTemplate
# from app.models import ConversationMemory
# from typing import Dict, Any

# # === Load Environment ===
# load_dotenv()
# AI_API_KEY = os.getenv("AI_API_KEY")
# DB_URL = os.getenv("DB_URL")

# # === Setup LLM (Google Gemini) ===
# llm = ChatGoogleGenerativeAI(
#     model="gemini-2.0-flash",
#     temperature=0.3,
#     verbose=True,
#     google_api_key=AI_API_KEY,
# )

# # === Setup SQL Database ===
# db = SQLDatabase.from_uri(DB_URL)
# toolkit = SQLDatabaseToolkit(db=db, llm=llm)

# # === User-specific memory storage ===
# user_memories: Dict[int, Dict[str, Any]] = {}

# # === Summarization Prompt ===
# summary_prompt = PromptTemplate(
#     input_variables=["history"],
#     template="""Summarize the following conversation concisely, focusing on:
# - Key user preferences and information
# - Important topics discussed
# - Any database queries or results
# - User's relationship goals or requirements

# Conversation:
# {history}

# Summary:"""
# )

# # === FAQ Content for Rishta App ===
# RISHTA_FAQ = """
# Common Questions about Our Rishta Platform:

# **Profile & Registration:**
# - Create detailed profiles with photos, bio, education, profession, and preferences
# - Verify your profile for increased trust
# - Privacy controls to show/hide information

# **Matching & Search:**
# - Advanced filters: age, location, education, profession, religion, sect
# - AI-powered recommendations based on compatibility
# - Save favorite profiles and send interest

# **Communication:**
# - Send interest requests to profiles you like
# - Chat with matched profiles after mutual acceptance
# - Video call feature for verified members

# **Safety & Privacy:**
# - All profiles are moderated
# - Report and block features available
# - Your information is encrypted and secure
# - Control who can see your profile

# **Subscription:**
# - Free basic membership with limited features
# - Premium plans for unlimited messaging and advanced search
# - Special family packages available
# """

# # === System Prompts ===
# SQL_PROMPT = """You are a helpful AI assistant for a Rishta (marriage/matchmaking) platform.
# You can access the database to help users with:
# - Finding matches based on criteria
# - Viewing profile statistics
# - Checking their received/sent interests
# - Analyzing compatibility data

# When answering database queries:
# 1. Be respectful and maintain privacy
# 2. Explain results in a friendly, conversational way
# 3. Suggest relevant next steps

# Current conversation context: {summary}

# User query: {input}"""

# # CHAT_PROMPT = """You are QuizHippo AI, a friendly and knowledgeable assistant for a Rishta (matchmaking) platform.

# # Your role:
# # - Answer FAQs about the platform features
# # - Provide relationship advice and tips
# # - Help with profile creation guidance
# # - Explain how matching algorithms work
# # - Offer cultural sensitivity and respect
# # - Be warm, supportive, and professional

# # Available FAQ Information:
# # {faq_content}

# # Conversation Summary: {summary}

# # Recent Conversation:
# # {history}

# # User: {user_input}
# # AI:"""
# CHAT_PROMPT = """
# You are RishtaMate AI — a caring, respectful, and intelligent assistant for a Rishta (matchmaking) platform.

# Your role:
# - Help users find compatible matches and guide them through the matchmaking process
# - Answer FAQs about profiles, interests, verification, and communication features
# - Explain how AI recommendations and matching algorithms work
# - Give relationship and compatibility guidance with empathy and cultural sensitivity
# - Support users in creating honest and appealing profiles
# - Maintain privacy, respect, and professionalism at all times

# Tone and Language:
# - Friendly, warm, and supportive
# - Respond **in the same language as the user**, including Roman Urdu or English
# - Respect cultural and family values common in South Asian matchmaking contexts

# Available Information:
# {faq_content}

# Conversation Summary:
# {summary}

# Recent Conversation:
# {history}

# User: {user_input}
# AI:"""


# # === Routing Decision Prompt ===
# # ROUTING_PROMPT = """Analyze this user query and determine if it requires database access or can be answered conversationally.

# # Query: "{query}"

# # Database access is needed for:
# # - Searching/filtering profiles (e.g., "find matches in Lahore", "show me engineers")
# # - Getting statistics (e.g., "how many profiles", "my received interests")
# # - Specific data retrieval (e.g., "show my matches", "list pending requests")
# # - CRUD operations on user data

# # Conversational response is sufficient for:
# # - FAQ questions about platform features
# # - Relationship advice or tips
# # - How-to questions about using the app
# # - General conversation
# # - Profile creation guidance

# # Respond with ONLY one word: DATABASE or CHAT"""

# ROUTING_PROMPT = """
# Analyze this user query (it can be in English, Roman Urdu, or mixed) and determine if it requires database access or can be answered conversationally.

# Database access is needed for:
# - Searching/filtering profiles (e.g., "find matches in Lahore", "Lahore mein engineers dikhao")
# - Getting statistics (e.g., "how many profiles", "mera received interests kya hain")
# - Specific data retrieval (e.g., "show my matches", "list pending requests")
# - CRUD operations on user data

# Conversational response is sufficient for:
# - FAQ questions about platform features
# - Relationship advice or tips
# - How-to questions about using the app
# - General conversation
# - Profile creation guidance

# Respond with ONLY one word: DATABASE or CHAT
# """



# def get_or_create_memory(user_id: int) -> Dict[str, Any]:
#     """Get or create memory objects for a specific user."""
#     if user_id not in user_memories:
#         user_memories[user_id] = {
#             'buffer': ConversationBufferMemory(
#                 memory_key="chat_history",
#                 return_messages=True,
#                 input_key="input",
#                 output_key="output"
#             ),
#             'summary': ""
#         }
#     return user_memories[user_id]


# def load_user_memory(user):
#     """
#     Loads previous conversation history from DB and creates a summary.
#     """
#     memory_obj = get_or_create_memory(user.id)
#     buffer_memory = memory_obj['buffer']
    
#     # Load past conversations
#     history = ConversationMemory.objects.filter(user=user).order_by("-created_at")[:50]  # Last 50 messages
#     history = reversed(history)  # Chronological order
    
#     chat_text = ""
#     for item in history:
#         buffer_memory.chat_memory.add_user_message(item.user_input)
#         buffer_memory.chat_memory.add_ai_message(item.ai_response)
#         chat_text += f"User: {item.user_input}\nAI: {item.ai_response}\n\n"
    
#     # Generate summary if there's history
#     if chat_text.strip():
#         summary_chain = LLMChain(llm=llm, prompt=summary_prompt)
#         summary = summary_chain.run({"history": chat_text})
#         memory_obj['summary'] = summary
    
#     return memory_obj


# def save_conversation(user, query: str, response: str):
#     """Save user query and AI response in the database."""
#     ConversationMemory.objects.create(
#         user=user,
#         user_input=query,
#         ai_response=response,
#     )


# # def decide_routing(query: str) -> str:
# #     """Use LLM to intelligently decide routing."""
# #     routing_chain = LLMChain(
# #         llm=llm,
# #         prompt=PromptTemplate(input_variables=["query"], template=ROUTING_PROMPT)
# #     )
    
# #     try:
# #         decision = routing_chain.run({"query": query}).strip().upper()
# #         return "database" if "DATABASE" in decision else "chat"
# #     except:
# #         # Fallback to keyword matching
# #         db_keywords = [
# #             "find", "search", "show", "list", "filter", "match", "profile",
# #             "how many", "count", "statistics", "database", "query",
# #             "interest", "request", "sent", "received", "pending"
# #         ]
# #         return "database" if any(kw in query.lower() for kw in db_keywords) else "chat"

# def decide_routing(query: str) -> str:
#     """Use LLM to intelligently decide routing (Roman Urdu supported)."""
#     routing_chain = LLMChain(
#         llm=llm,
#         prompt=PromptTemplate(input_variables=["query"], template=ROUTING_PROMPT)
#     )
    
#     try:
#         decision = routing_chain.run({"query": query}).strip().upper()
#         return "database" if "DATABASE" in decision else "chat"
#     except Exception as e:
#         print(f"Routing error: {str(e)}")
#         # If LLM fails, default to chat rather than risking wrong DB access
#         return "chat"



# def create_sql_agent(memory):
#     """Create SQL agent with memory."""
#     return initialize_agent(
#         tools=toolkit.get_tools(),
#         llm=llm,
#         agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
#         memory=memory,
#         verbose=True,
#         handle_parsing_errors=True,
#         max_iterations=5
#     )


# def ai_router(user, query: str) -> str:
#     """
#     Main routing function with intelligent decision making and memory.
#     """
#     try:
#         # Load user's memory (includes past conversations and summary)
#         memory_obj = load_user_memory(user)
#         buffer_memory = memory_obj['buffer']
#         conversation_summary = memory_obj['summary']
        
#         # Decide routing using LLM
#         route = decide_routing(query)
        
#         if route == "database":
#             # === SQL Agent Mode ===
#             sql_agent = create_sql_agent(buffer_memory)
            
#             # Add context to the query
#             context_prompt = SQL_PROMPT.format(
#                 summary=conversation_summary or "No previous context",
#                 input=query
#             )
            
#             response = sql_agent.run(context_prompt)
            
#         else:
#             # === Chat Agent Mode ===
#             # Get recent conversation history
#             recent_messages = buffer_memory.chat_memory.messages[-10:]  # Last 10 messages
#             history_text = "\n".join([
#                 f"{'User' if i % 2 == 0 else 'AI'}: {msg.content}" 
#                 for i, msg in enumerate(recent_messages)
#             ])
            
#             # Create chat agent with full context
#             chat_agent = LLMChain(
#                 llm=llm,
#                 prompt=PromptTemplate(
#                     input_variables=["faq_content", "summary", "history", "user_input"],
#                     template=CHAT_PROMPT
#                 ),
#                 verbose=True
#             )
            
#             response = chat_agent.run({
#                 "faq_content": RISHTA_FAQ,
#                 "summary": conversation_summary or "New conversation",
#                 "history": history_text or "No recent messages",
#                 "user_input": query
#             })
        
#         # Save to memory
#         buffer_memory.save_context({"input": query}, {"output": response})
#         save_conversation(user, query, response)
        
#         # Auto-summarize every 20 messages
#         message_count = len(buffer_memory.chat_memory.messages)
#         if message_count > 0 and message_count % 20 == 0:
#             full_history = "\n".join([
#                 f"{'User' if i % 2 == 0 else 'AI'}: {msg.content}"
#                 for i, msg in enumerate(buffer_memory.chat_memory.messages)
#             ])
            
#             summary_chain = LLMChain(llm=llm, prompt=summary_prompt)
#             new_summary = summary_chain.run({"history": full_history})
#             memory_obj['summary'] = new_summary
        
#         return response.strip()
    
#     except Exception as e:
#         print(f"Error in ai_router: {str(e)}")
#         return f"I apologize, but I encountered an error processing your request. Please try rephrasing your question or contact support if the issue persists."


# def clear_user_memory(user_id: int):
#     """Clear memory for a specific user (useful for logout/cleanup)."""
#     if user_id in user_memories:
#         del user_memories[user_id]
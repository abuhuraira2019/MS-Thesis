# --- 1. STREAMLIT CLOUD SQLITE PATCH ---
# This MUST be at the very top of the file to prevent ChromaDB from crashing on the cloud.
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import os
import streamlit as st
import chromadb
from chromadb import Documents, EmbeddingFunction, Embeddings
import re
from google import genai
from google.genai import types
from dotenv import load_dotenv

# --- 2. Load Environment Variables ---
try:
    api_key = st.secrets["GEMINI_API_KEY"]
except KeyError:
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    st.error("🚨 GEMINI_API_KEY not found. Please add it to your .env file or Streamlit Cloud Secrets.")
    st.stop()

# --- 3. Custom Embedding Function ---
class GeminiEmbeddingFunction(EmbeddingFunction):
    def __init__(self, key: str):
        self.embed_client = genai.Client(api_key=key)
        
    def __call__(self, input: Documents) -> Embeddings:
        response = self.embed_client.models.embed_content(
            model='gemini-embedding-001',
            contents=input,
        )
        return [e.values for e in response.embeddings]

# --- 4. UI Configuration ---
st.set_page_config(page_title="PsyScreen-RAG", page_icon="🧠", layout="wide")
st.title("PsyScreen-RAG Clinical Assistant")
st.markdown("MS Thesis Project | Developed by Abu Huraira")

# --- 5. Initialize RAG Database ---
@st.cache_resource(show_spinner="Connecting to Vector Database...")
def get_chroma_collection(api_key_val):
    db_client = chromadb.PersistentClient(path="./chroma_db")
    embedder = GeminiEmbeddingFunction(key=api_key_val)
    return db_client.get_collection(name="clinical_manuals", embedding_function=embedder)

try:
    collection = get_chroma_collection(api_key)
except Exception as e:
    st.error(f"ChromaDB Error. Did you run build_db.py locally and push the folder to GitHub? Details: {e}")
    st.stop()

# --- 6. System Instructions ---
system_prompt = """
System Identity & Purpose
You are "PsyScreen-RAG", an interactive, empathetic, and highly competent clinical assistant developed for an MS Thesis.
You administer standardized mental health screenings (PHQ-9, GAD-7, etc.) based STRICTLY on the provided RAG context.
You must fluently understand and respond in the user's preferred language: English, Urdu, or Roman Urdu (Minglish).
You must ask patient name in first question with the language and test they should take, first examine the test he or shee neddds then start that.

Core Operating Instructions - INTERACTIVE EXAMINER MODE:
1. Be Conversational & Validating: Before asking the next question, briefly validate the user's previous answer.
2. Ask Exactly ONE Question: After validating, smoothly transition into asking ONLY the exact next questionnaire item. Wait for the user's response.
3. Diagnostic Logic: When a test is finished, calculate the score based on the RAG rubric.
4. Always include this disclaimer upon scoring: "This is a screening tool, not a formal medical diagnosis."
5. Provide Rationale: When generating the final report, explicitly cite the clinical reasoning based on the RAG data.
"""

# --- 7. Session State Initialization ---
if "genai_client" not in st.session_state:
    st.session_state.genai_client = genai.Client(api_key=api_key)

if "chat_session" not in st.session_state:
    st.session_state.chat_session = st.session_state.genai_client.chats.create(
        model="gemini-3-flash-preview",
        config=types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=0.3,
            thinking_config=types.ThinkingConfig(
                thinking_level="HIGH"
            )
        )
    )

if "messages" not in st.session_state:
    st.session_state.messages = []
    try:
        initial_response = st.session_state.chat_session.send_message(
            "Warmly greet the user, ask which language they prefer (English, Urdu, or Roman Urdu), explain the purpose of the screening, and ask if they are ready to begin."
        )
        st.session_state.messages.append({"role": "assistant", "content": initial_response.text})
    except Exception as e:
        st.error(f"API Error during initialization: {e}")

# --- 8. Crisis Detection Failsafe ---
def check_for_crisis(user_text):
    crisis_keywords = [
        r"\bkill myself\b", r"\bsuicide\b", r"\bend it all\b", r"\bwant to die\b", 
        r"\bkhudkushi\b", r"\bmar jana chahta\b", r"\bjaan dena\b", r"\bzindagi khatam\b",
        r"\bjeene ka koi faida nahi\b", r"\bmarne ka dil\b"
    ]
    user_text_lower = user_text.lower()
    for pattern in crisis_keywords:
        if re.search(pattern, user_text_lower):
            return True
    return False

def get_crisis_response():
    return """
    **🚨 CRISIS ALERT / ہنگامی صورتحال**
    You are not alone, and help is available right now. / آپس اکیلے نہیں ہیں، مدد موجود ہے۔
    
    **In Pakistan:**
    * **Umang Pakistan:** Call 0311-7786264
    * **Rozan Counseling:** Call 0304-111-1741
    * **Emergency Services:** Call 1122
    
    *Your session has been paused for your safety.*
    """

# --- 9. Sidebar & Session Management ---
st.sidebar.header("Patient Reporting")

# NEW: Clear Chat / Start New Test Button
if st.sidebar.button("🔄 Start New Test / Clear Chat"):
    if "messages" in st.session_state:
        del st.session_state["messages"]
    if "chat_session" in st.session_state:
        del st.session_state["chat_session"]
    st.rerun()

st.sidebar.markdown("---")

# Final Report Button
if st.sidebar.button("📄 Generate Final Clinical Report"):
    with st.spinner("Compiling structured report with clinical rationale..."):
        report_prompt = """
        The screening is complete. Generate a comprehensive final clinical report. 
        Format nicely with Markdown tables and bullet points.
        It MUST include:
        1. Tests Administered.
        2. Calculated Scores & Severity Categories.
        3. A brief paragraph citing the clinical rationale for this severity category based on the RAG data.
        4. Recommended CBT Techniques, explicitly explaining WHY these techniques were chosen based on the user's specific answers during the test.
        """
        response = st.session_state.chat_session.send_message(report_prompt)
        st.session_state.messages.append({"role": "assistant", "content": response.text})
        st.rerun()

# --- 10. Render Chat UI ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --- 11. Handle User Input & RAG ---
user_input = st.chat_input("Type your response here...")

if user_input:
    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    if check_for_crisis(user_input):
        crisis_msg = get_crisis_response()
        with st.chat_message("assistant"):
            st.error(crisis_msg)
        st.session_state.messages.append({"role": "assistant", "content": crisis_msg})
        st.stop() 

    with st.spinner("Retrieving clinical data from manuals..."):
        results = collection.query(query_texts=[user_input], n_results=3)
        retrieved_context = "\n\n".join(results["documents"][0]) if results["documents"] else "No specific clinical context found."

    enriched_prompt = f"""
    [Clinical Context from Database]:
    {retrieved_context}
    
    [User Input]:
    {user_input}
    
    Using the clinical context above, determine the next step. If you are administering a test, briefly validate the user's input, then ask ONLY the exact next question. Do not skip ahead.
    """

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = st.session_state.chat_session.send_message(enriched_prompt)
            st.markdown(response.text)
            st.session_state.messages.append({"role": "assistant", "content": response.text})


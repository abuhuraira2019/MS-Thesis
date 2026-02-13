import os
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import streamlit as st
import chromadb
from chromadb import Documents, EmbeddingFunction, Embeddings
import re
from google import genai
from google.genai import types
from dotenv import load_dotenv

# --- 1. Load Environment Variables (No more Sidebar Input) ---
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    st.error("🚨 GEMINI_API_KEY not found. Please ensure it is in your .env file.")
    st.stop()

# --- 2. Custom Embedding Function ---
class GeminiEmbeddingFunction(EmbeddingFunction):
    def __init__(self, key: str):
        self.client = genai.Client(api_key=key)
        
    def __call__(self, input: Documents) -> Embeddings:
        response = self.client.models.embed_content(
            model='gemini-embedding-001',
            contents=input,
        )
        return [e.values for e in response.embeddings]

# --- 3. UI Configuration ---
st.set_page_config(page_title="PsyScreen-RAG", page_icon="🧠", layout="wide")
st.title("PsyScreen-RAG Clinical Assistant")
st.markdown("MS Thesis Project | Developed by Abu Huraira")

# --- 4. Initialize Client & RAG Database ---
client = genai.Client(api_key=api_key)

@st.cache_resource(show_spinner="Connecting to Local Vector Database...")
def get_chroma_collection(api_key_val):
    db_client = chromadb.PersistentClient(path="./chroma_db")
    embedder = GeminiEmbeddingFunction(key=api_key_val)
    return db_client.get_collection(name="clinical_manuals", embedding_function=embedder)

try:
    collection = get_chroma_collection(api_key)
except Exception as e:
    st.error(f"ChromaDB Error. Did you run build_db.py? Details: {e}")
    st.stop()

# --- 5. System Instructions ---
system_prompt = """
System Identity & Purpose
"PsyScreen-RAG" is a clinical assistant. You are administering standardized mental health screenings.
You must fluently understand and respond in the user's preferred language: English, Urdu, or Roman Urdu.

Core Operating Instructions - STRICT EXAMINER MODE:
1. Ask exactly ONE question at a time. Do not list multiple questions.
2. Wait for the user to answer the current question before moving to the next.
3. Use the "Clinical Context" provided to formulate the exact questionnaire items (PHQ-9, GAD-7, etc.).
4. When a test is finished, calculate the score based on the RAG rubric.
5. Always include this disclaimer upon scoring: "This is a screening tool, not a formal medical diagnosis."
"""

# --- 6. Session State Initialization (Using Gemini 3 Flash Preview) ---
if "chat_session" not in st.session_state:
    st.session_state.chat_session = client.chats.create(
        model="gemini-3-flash-preview",
        config=types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=0.1, 
            thinking_config=types.ThinkingConfig(
                thinking_level="HIGH"
            )
        )
    )

if "messages" not in st.session_state:
    st.session_state.messages = []
    try:
        initial_response = st.session_state.chat_session.send_message(
            "Greet the user, ask which language they prefer (English, Urdu, or Roman Urdu), and ask if they are ready to begin the screening."
        )
        st.session_state.messages.append({"role": "assistant", "content": initial_response.text})
    except Exception as e:
        st.error(f"API Error: {e}")

# --- 7. Crisis Detection Failsafe ---
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

# --- 8. Sidebar & Report Generation ---
st.sidebar.header("Patient Reporting")
if st.sidebar.button("📄 Generate Final Clinical Report"):
    with st.spinner("Compiling structured report..."):
        report_prompt = "The screening is complete. Generate a final clinical report including Tests Administered, Calculated Scores, and CBT Recommendations based strictly on the RAG data. Use Markdown tables."
        response = st.session_state.chat_session.send_message(report_prompt)
        st.session_state.messages.append({"role": "assistant", "content": response.text})
        st.rerun()

# --- 9. Render Chat UI ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --- 10. Handle User Input & RAG ---
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

    # Query the database
    with st.spinner("Retrieving clinical data from PDFs..."):
        results = collection.query(query_texts=[user_input], n_results=3)
        retrieved_context = "\n\n".join(results["documents"][0]) if results["documents"] else "No context found."

    enriched_prompt = f"""
    [Clinical Context from Database]:
    {retrieved_context}
    
    [User Input]:
    {user_input}
    
    Using the clinical context above, determine the next step. If administering a test, ask ONLY the next question.
    """

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = st.session_state.chat_session.send_message(enriched_prompt)
            st.markdown(response.text)

            st.session_state.messages.append({"role": "assistant", "content": response.text})


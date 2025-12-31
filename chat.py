import streamlit as st
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from google import genai

# -------------------
# Page config
# -------------------
st.set_page_config(page_title="Chat AI Toheed Rehman", layout="centered")

# -------------------
# SIDEBAR
# -------------------
with st.sidebar:
    st.markdown("##  Chat AI")
    st.markdown("---")

    st.markdown("### 👤 Developer")
    st.write("**Toheed Rehman**")

    st.markdown("### ⚙️ Model Info")
    st.write("Gemini 2.5 Flash Lite")

    st.markdown("### 📚 RAG Source")
    st.write("Iqra University (Pakistan)")

    st.markdown("---")

    if st.button("🗑 Clear Chat"):
        st.session_state.chat_history = []
        st.rerun()

    st.markdown("---")
    st.caption("© 2025 Toheed Rehman")

# -------------------
# CSS
# -------------------
st.markdown("""
<style>
body {
    background: linear-gradient(135deg,#f0f4ff,#e6f0ff);
}

/* HEADER */
.chat-header {
    position: sticky;
    top: 0;
    z-index: 100;
    text-align: center;
    font-size: 1.6em;
    font-weight: 700;
    background: linear-gradient(90deg,#0b5ed7,#198754);
    color: white;
    padding: 15px;
    border-radius: 12px 12px 0 0;
}

/* CHAT BOX */
.chat-box {
    max-width: 750px;
    margin: auto;
    height: 60vh;
    overflow-y: auto;
    padding: 15px;
    background: #ffffff;
    border-radius: 0 0 12px 12px;
    box-shadow: 0 10px 25px rgba(0,0,0,0.1);
}

/* MESSAGES */
.user {
    background: #d0e3ff;
    color: #0b5ed7;
    padding: 12px 15px;
    border-radius: 12px;
    max-width: 80%;
    margin-left: auto;
    margin-bottom: 10px;
}

.ai {
    background: #d3f8e2;
    color: #198754;
    padding: 12px 15px;
    border-radius: 12px;
    max-width: 80%;
    margin-right: auto;
    margin-bottom: 10px;
}

.error {
    background: #f8d7da !important;
    color: #842029 !important;
}

/* INPUT AREA */
.input-area {
    position: sticky;
    bottom: 0;
    background: #ffffff;
    padding: 12px;
    display: flex;
    justify-content: center;
    box-shadow: 0 -2px 10px rgba(0,0,0,0.05);
    border-top: 1px solid #ddd;
    border-radius: 0 0 12px 12px;
}

.input-box {
    display: flex;
    width: 50%;
    max-width: 700px;
}

.input-box input[type="text"] {
    flex: 1;
    padding: 14px 16px;
    border-radius: 8px 0 0 8px;
    border: 1px solid #ccc;
    font-size: 1em;
    outline: none;
    transition: 0.2s;
}

.input-box input[type="text"]:focus {
    border-color: #0b5ed7;
    box-shadow: 0 0 5px rgba(11,94,215,0.3);
}

.input-box button {
    padding: 14px 24px;
    border: none;
    background: #0b5ed7;
    color: white;
    font-weight: 600;
    font-size: 1em;
    border-radius: 0 8px 8px 0;
    cursor: pointer;
    transition: 0.2s;
}

.input-box button:hover {
    background: #084298;
}

div[data-testid="stFormSubmitButton"] button {
    margin-top: 25px !important;
    height: 36px !important;
    padding: 0 14px;
    font-weight: 600;
}

div[data-baseweb="input"] input {
    height: 36px !important;
}

@media (max-width: 767px) {
    .input-box {
        flex-direction: row !important;
        gap: 6px;
    }

    div[data-baseweb="input"] input {
        width: auto !important;
        flex: 1;
    }

    div[data-testid="stFormSubmitButton"] button {
        width: auto !important;
        margin-top: 0 !important;
    }
}

/* FOOTER */
.footer {
    text-align: center;
    color: #084298;
    padding: 15px;
    font-size: 0.9em;
}
</style>

<script>
setTimeout(() => {
  const box = document.querySelector('.chat-box');
  if (box) box.scrollTop = box.scrollHeight;
}, 100);
</script>
""", unsafe_allow_html=True)

# -------------------
# NAVBAR
# -------------------
st.markdown("""
<div style="
    width:100%;
    background: linear-gradient(90deg,#0b5ed7,#198754);
    padding: 12px 25px;
    color: white;
    font-size: 1.1em;
    font-weight: 600;
    display: flex;
    justify-content: space-between;
    align-items: center;
    border-radius: 10px;
    margin-bottom: 15px;
">
    <div>🤖 Chat AI</div>
    <div style="font-size:0.9em;">Personal and Professional AI Assistant</div>
</div>
""", unsafe_allow_html=True)

# -------------------
# Gemini API
# -------------------
API_KEY = "AIzaSyBL2lIWVd2ZxJS3sponEwWddYl2GTZe1pI"  # YOUR KEY
MODEL_NAME = "models/gemini-2.5-flash-lite"
client = genai.Client(api_key=API_KEY)

# -------------------
# RAG DATA
# -------------------
documents = [
    {
        "university": "Iqra University (IU)",
        "founded": 1998,
        "type": "Private, Chartered University",
        "campuses": ["Karachi", "Islamabad", "Lahore", "Faisalabad", "Hyderabad"],
        "admissions": {
            "undergraduate": {
                "eligibility": "Intermediate (12 years education) or equivalent with minimum marks depending on the program",
                "semesters": ["Fall", "Spring", "Summer"],
                "application_process": "Online application, submission of documents, entry test for some programs, interview if required",
                "scholarships": ["Merit-based", "Need-based", "Sports Scholarships"]
            },
            "graduate": {
                "eligibility": "Bachelor’s degree in relevant field",
                "semesters": ["Fall", "Spring", "Summer"],
                "application_process": "Online application, submission of transcripts, interview, program-specific requirements",
                "scholarships": ["Research Assistantship", "Merit-based"]
            }
        },
        "departments": [
            {"name": "Business Administration", "programs": ["BBA", "MBA", "Executive MBA"], "faculty": "Faculty of Business Administration"},
            {"name": "Computer Science", "programs": ["BS CS", "MS CS", "PhD CS"], "faculty": "Faculty of Computing"},
            {"name": "Software Engineering", "programs": ["BS SE", "MS Software Engineering"], "faculty": "Faculty of Computing"},
            {"name": "Electrical Engineering", "programs": ["BS EE", "MS EE"], "faculty": "Faculty of Engineering"},
            {"name": "Mass Communication", "programs": ["BA Mass Communication", "MA Mass Communication"], "faculty": "Faculty of Social Sciences"},
            {"name": "Law", "programs": ["LLB", "LLM"], "faculty": "Faculty of Law"},
            {"name": "Psychology", "programs": ["BS Psychology", "MS Psychology"], "faculty": "Faculty of Social Sciences"},
            {"name": "Economics", "programs": ["BA Economics", "MA Economics"], "faculty": "Faculty of Arts & Humanities"},
            {"name": "English", "programs": ["BA English", "MA English"], "faculty": "Faculty of Arts & Humanities"}
        ],
        "faculties": [
            "Faculty of Business Administration", "Faculty of Computing", "Faculty of Engineering",
            "Faculty of Social Sciences", "Faculty of Arts & Humanities", "Faculty of Law"
        ],
        "notable_alumni": [
            "Shahid Afridi – Cricketer", "Saba Qamar – Actress", "Ali Rehman Khan – Actor",
            "Fawad Khan – Actor", "Ali Zafar – Singer/Actor", "Hina Rabbani Khar – Politician",
            "Samiya Mumtaz – Actress"
        ]
    }
]

# -------------------
# LOAD FAISS INDEX
# -------------------
@st.cache_resource
def load_index():
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    # Flatten documents to text strings for embeddings
    doc_texts = []
    for doc in documents:
        text = f"{doc['university']} ({doc['founded']})\nType: {doc['type']}\nCampuses: {', '.join(doc['campuses'])}\n"
        text += f"Departments: {', '.join([d['name'] for d in doc['departments']])}\n"
        text += f"Faculties: {', '.join(doc['faculties'])}\n"
        text += f"Alumni: {', '.join(doc['notable_alumni'])}\n"
        doc_texts.append(text)
    emb = embedder.encode(doc_texts)
    index = faiss.IndexFlatL2(emb.shape[1])
    index.add(np.array(emb))
    return embedder, index, doc_texts

embedder, index, doc_texts = load_index()

# -------------------
# RAG QUERY FUNCTION
# -------------------
def ask_gemini_rag(q, k=2):
    q_emb = embedder.encode([q])
    D, I = index.search(np.array(q_emb), k)
    
    # Gather top-k relevant documents
    retrieved_texts = []
    for i in I[0]:
        if i < len(doc_texts):
            retrieved_texts.append(doc_texts[i])
    
    # Fallback if nothing retrieved
    if not retrieved_texts:
        retrieved_texts = doc_texts
    
    context = "\n\n".join(retrieved_texts)
    
    prompt = f"""
You are a personal and professional AI assistant.
Answer the user's question based ONLY on the following context:

CONTEXT:
{context}

QUESTION:
{q}

Provide a clear, professional answer using only the context.
"""
    try:
        r = client.models.generate_content(model=MODEL_NAME, contents=prompt)
        return r.text
    except Exception as e:
        return f"Error: {e}"

# -------------------
# Session State
# -------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# -------------------
# UI
# -------------------
st.markdown('<div class="chat-header">Chat AI </div>', unsafe_allow_html=True)

msgs = ""
for c in st.session_state.chat_history:
    msgs += f'<div class="user">You: {c["user"]}</div>'
    cls = "ai error" if "Error:" in c["ai"] else "ai"
    msgs += f'<div class="{cls}">AI: {c["ai"]}</div>'

st.markdown(f'<div class="chat-box">{msgs}</div>', unsafe_allow_html=True)

# Input
st.markdown('<div class="input-area">', unsafe_allow_html=True)
with st.form("chat", clear_on_submit=True):
    col1, col2 = st.columns([5,1])
    with col1:
        q = st.text_input("", placeholder="Type your question...")
    with col2:
        send = st.form_submit_button("Send")

    if send and q:
        st.session_state.chat_history.append({
            "user": q,
            "ai": ask_gemini_rag(q)
        })
        st.rerun()
st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown('<div class="footer">© 2025 Toheed Rehman. All Rights Reserved.</div>', unsafe_allow_html=True)

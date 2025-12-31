import streamlit as st
from google import genai
import json

# -------------------
# Page config
# -------------------
st.set_page_config(page_title="Chat AI Toheed Rehman", layout="centered")

# -------------------
# Sidebar
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
    border-radius: 25px 25px 0 0;
}

/* CHAT BOX */
.chat-box {
    max-width: 750px;
    margin: auto;
    height: 50vh;
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

.stMainBlockContainer {
    width: 50% !important;
    max-width: 750px !important;
    margin: auto !important;
    padding: 1rem !important;
    box-sizing: border-box;
    margin-top: 20px !important;
}
            
            .st-emotion-cache-5qfegl {
    margin-top: 27px !important;  /* moves the element down */
}

/* FOOTER */
.footer {
    text-align: center;
    color: #084298;
    padding: 15px;
    font-size: 0.9em;
}
            

 Mobile Responsive
-------------------- */
@media (max-width: 767px) {

    .stMainBlockContainer {
        width: 90% !important;
        margin-top: 10px !important;
        padding: 0.5rem !important;
    }

    .chat-box {
        height: 40vh;
        padding: 10px;
    }

    .input-box {
        width: 90% !important;
        flex-direction: column !important;
        gap: 8px;
    }

    .input-box input[type="text"] {
        width: 100% !important;
        border-radius: 8px !important;
    }

    .input-box button {
        width: 100% !important;
        border-radius: 8px !important;
    }

    .st-emotion-cache-5qfegl {
        margin-top: 20px !important; /* smaller spacing on mobile */
    }
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
# Gemini API
# -------------------
API_KEY = "AIzaSyAE38eZ5HGvH6Ln-AVhu2bhMQMxljeW_xI"  # <-- YOUR API KEY
MODEL_NAME = "models/gemini-2.5-flash-lite"
client = genai.Client(api_key=API_KEY)

# -------------------
# Structured RAG DATA
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
            "Ahmed Khan – Entrepreneur",
            "Fatima Ali – Researcher",
            "Hassan Sheikh – Engineer",
            "Aisha Siddiqui – Scientist",
            "Omar Rehman – Lawyer",
            "Sara Malik – Academic",
            "Yusuf Qureshi – Technologist"
        ],
        "example_policies": {
            "admissions_policy": "All applicants must submit certified transcripts and meet minimum program requirements. Late applications may be considered only if seats are available.",
            "academic_policy": "Students must maintain a minimum GPA of 2.5 for undergraduate programs and 3.0 for graduate programs. Academic dishonesty will result in disciplinary action.",
            "code_of_conduct": "All students are expected to follow university rules regarding respectful behavior, attendance, and use of campus facilities. Harassment or misconduct is strictly prohibited.",
            "scholarship_policy": "Scholarships are awarded based on merit, financial need, or special achievements. Students must maintain required academic performance to retain scholarships.",
            "leave_policy": "Students may request leave for medical or personal reasons. Leaves exceeding two weeks require formal approval from the department head.",
            "grading_policy": "Grades are awarded according to the official grading scale. Appeals must be submitted within two weeks of receiving the grade.",
            "attendance_policy": "Students are required to attend at least 75% of lectures and labs per course. Attendance will be recorded and reported to the department. Missing classes without valid reason may affect grades.",
            "exam_policy": "Students must arrive at least 15 minutes before exams. No electronic devices or unauthorized materials are allowed. Cheating or academic dishonesty will result in a zero grade and possible disciplinary action.",
            "library_policy": "Library resources must be handled with care. Books must be returned by the due date. Silence must be maintained. Overdue fines will be applied for late returns.",
            "internship_policy": "All students in applicable programs must complete internships approved by the department. Students must submit reports and evaluations. Failure to complete the internship may delay graduation.",
            "example_rules": [
                "No use of mobile phones during lectures or exams.",
                "Proper university ID must be displayed on campus at all times.",
                "Smoking and alcohol are strictly prohibited on campus.",
                "Students must respect faculty, staff, and fellow students.",
                "Late submission of assignments will incur penalties unless prior approval is obtained.",
                "Unauthorized access to university facilities or labs is forbidden.",
                "Cheating, plagiarism, or academic dishonesty will result in disciplinary action.",
                "Students must follow safety guidelines in labs and workshops.",
                "Campus facilities must be used responsibly and kept clean.",
                "All official communications must be via university email accounts."
            ]
        }
    }
]

# -------------------
# Gemini LLM query function
# -------------------
def ask_gemini_llm(q):
    """
    Ask the Gemini LLM directly using structured documents as context.
    """
    context_json = json.dumps(documents, indent=2)
    prompt = f"""
You are a personal and professional AI assistant.
Answer the user's question based ONLY on the following JSON context about Iqra University:

CONTEXT:
{context_json}

QUESTION:
{q}

Provide a clear and concise answer. 
If the answer is not in the context, respond: "I am sorry, but the provided context does not contain information about that."
"""
    try:
        r = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt
        )
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

# Input area
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
            "ai": ask_gemini_llm(q)
        })
        st.rerun()
st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown('<div class="footer">© 2025 Toheed Rehman. All Rights Reserved.</div>', unsafe_allow_html=True)

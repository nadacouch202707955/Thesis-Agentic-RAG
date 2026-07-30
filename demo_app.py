"""
demo_app.py
Nada Ali Yaqoob · 202507955 · Polytechnic of Bahrain

Live demo interface for the Agentic RAG Academic Advisor.
Wraps orchestrator_agent.run_agentic_query() in a simple Streamlit web UI
for viva demonstration — clean chat-style display of the final answer only.

SETUP (one-time):
    pip install streamlit

RUN:
    py -m streamlit run demo_app.py

This opens a browser tab at http://localhost:8501 automatically.
"""

import streamlit as st
from orchestrator_agent import run_agentic_query

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Academic Advisor — Agentic RAG",
    page_icon="🎓",
    layout="centered"
)

st.title("🎓 Academic Advisor")
st.caption("Agentic RAG · Bahrain Polytechnic · MSc AI Thesis Demo")

# ─────────────────────────────────────────────
# SESSION STATE — keeps chat history while the app is open
# ─────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []  # list of (role, text) tuples

# ─────────────────────────────────────────────
# SIDEBAR — Student ID input
# ─────────────────────────────────────────────
with st.sidebar:
    st.header("Student Info")
    student_id = st.text_input(
        "Student ID (optional)",
        placeholder="e.g. S12345",
        help="Enter a valid Student ID to personalise responses using academic profile data (GPA, credits, standing)."
    )
    st.markdown("---")
    st.caption(
        "Leave blank for a general policy question with no personalisation."
    )
    if st.button("Clear conversation"):
        st.session_state.history = []
        st.rerun()

# ─────────────────────────────────────────────
# DISPLAY CHAT HISTORY
# ─────────────────────────────────────────────
for role, text in st.session_state.history:
    with st.chat_message(role):
        st.markdown(text)

# ─────────────────────────────────────────────
# CHAT INPUT
# ─────────────────────────────────────────────
question = st.chat_input("Ask a question about academic policies, registration, GPA, deadlines...")

if question:
    st.session_state.history.append(("user", question))
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            sid = student_id.strip() if student_id and student_id.strip() else None
            result = run_agentic_query(
                query=question,
                student_id=sid,
                conversation_history=[],
                verbose=False
            )
            answer = result["final_response"]

            if result["notification"]["escalated"]:
                answer += "\n\n*This question has been flagged for advisor review.*"

        st.markdown(answer)

    st.session_state.history.append(("assistant", answer))
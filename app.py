import streamlit as st
import tempfile
import os
import io
import uuid
import shutil
from datetime import datetime
from dotenv import load_dotenv
from ingest import ingest_multiple_files
from qa_chain import build_qa_chain, get_answer
from agents import AgentOrchestrator
from pathlib import Path
import html

load_dotenv()

import re as _re

def _render_message(raw: str) -> str:
    """Convert LLM markdown (bullets, bold) to safe inline HTML for chat bubbles."""
    parts = []
    in_list = False
    for line in raw.split("\n"):
        s = line.strip()
        ordered_match = _re.match(r"^(\d+)\.\s+(.*)", s)
        is_unordered = s.startswith("- ") or s.startswith("* ")
        is_bullet = is_unordered or bool(ordered_match)

        if is_bullet:
            if not in_list:
                parts.append("<ul style='margin:6px 0 6px 0;padding-left:20px;list-style:disc;'>")
                in_list = True
            if ordered_match:
                num, text_raw = ordered_match.group(1), ordered_match.group(2)
                text = _re.sub(r"\*\*(.*?)\*\*", r"<strong>\1</strong>", html.escape(text_raw))
                parts.append(f"<li style='margin-bottom:5px;'>{num}. {text}</li>")
            else:
                text_raw = _re.sub(r"^[-*]\s", "", s)
                text = _re.sub(r"\*\*(.*?)\*\*", r"<strong>\1</strong>", html.escape(text_raw))
                parts.append(f"<li style='margin-bottom:5px;'>{text}</li>")
        else:
            if in_list:
                parts.append("</ul>")
                in_list = False
            if s:
                text = _re.sub(r"\*\*(.*?)\*\*", r"<strong>\1</strong>", html.escape(s))
                parts.append(f"<p style='margin:3px 0;'>{text}</p>")
    if in_list:
        parts.append("</ul>")
    return "".join(parts)


st.set_page_config(
    page_title="MediChat — Your Medical Assistant",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

st._config.set_option('theme.base', 'light')
st._config.set_option('theme.backgroundColor', '#f0f4ff')
st._config.set_option('theme.primaryColor', '#2451b3')
st._config.set_option('theme.textColor', '#0d2b6e')
st._config.set_option('theme.secondaryBackgroundColor', '#e0e8ff')

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Nunito:wght@400;500;600;700;800&family=Playfair+Display:wght@600;700&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] { font-family: 'Nunito', sans-serif; background-color: #f0f4ff; }
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 0 !important; max-width: 100% !important; }
.stApp { background: #efeae2; min-height: 100vh; }

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d2b6e 0%, #163580 40%, #1a3d8f 100%) !important;
    border-right: none !important;
    box-shadow: 4px 0 24px rgba(13,43,110,0.18);
}
[data-testid="stSidebar"] * { color: #e8eeff !important; }
[data-testid="stFileUploader"] *:not(button) { color: #ffffff !important; opacity: 1 !important; -webkit-text-fill-color: #ffffff !important; }
[data-testid="stSidebar"] .stMarkdown p,
[data-testid="stSidebar"] .stMarkdown li { color: #b8c8f0 !important; font-size: 14px; line-height: 1.7; }
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 { color: #ffffff !important; font-family: 'Playfair Display', serif !important; font-size: 20px !important; letter-spacing: 0.3px; }

.sidebar-logo { text-align: center; padding: 28px 16px 20px; border-bottom: 1px solid rgba(255,255,255,0.12); margin-bottom: 20px; }
.sidebar-logo .logo-icon { font-size: 48px; display: block; margin-bottom: 8px; filter: drop-shadow(0 4px 12px rgba(100,160,255,0.4)); }
.sidebar-logo .logo-title { font-family: 'Playfair Display', serif; font-size: 22px; font-weight: 700; color: #ffffff !important; letter-spacing: 0.5px; }
.sidebar-logo .logo-subtitle { font-size: 12px; color: #8aaee0 !important; margin-top: 2px; letter-spacing: 1px; text-transform: uppercase; }

[data-testid="stFileUploader"] {
    background: rgba(255,255,255,0.07) !important; border: 2px dashed rgba(255,255,255,0.25) !important;
    border-radius: 16px !important; padding: 12px !important; transition: all 0.3s ease;
}
[data-testid="stFileUploader"]:hover { border-color: rgba(100,160,255,0.6) !important; background: rgba(255,255,255,0.12) !important; }
[data-testid="stFileUploader"] button {
    background: rgba(100,160,255,0.2) !important; border: 1px solid rgba(100,160,255,0.4) !important;
    color: #c8daff !important; border-radius: 10px !important; font-family: 'Nunito', sans-serif !important;
    font-weight: 600 !important; font-size: 13px !important;
}

[data-testid="stSidebar"] .stButton > button {
    width: 100%; background: linear-gradient(135deg, #4a90d9 0%, #2563c7 100%) !important;
    color: white !important; border: none !important; border-radius: 14px !important;
    padding: 12px 20px !important; font-family: 'Nunito', sans-serif !important;
    font-weight: 700 !important; font-size: 15px !important; letter-spacing: 0.3px;
    box-shadow: 0 4px 16px rgba(37,99,199,0.35) !important; transition: all 0.25s ease !important; margin-top: 8px;
}
[data-testid="stSidebar"] .stButton > button:hover { transform: translateY(-2px) !important; box-shadow: 0 8px 24px rgba(37,99,199,0.45) !important; }
[data-testid="stSidebar"] hr { border-color: rgba(255,255,255,0.12) !important; margin: 16px 0 !important; }

.doc-pill {
    background: rgba(74,144,217,0.2); border: 1px solid rgba(74,144,217,0.35);
    border-radius: 20px; padding: 5px 12px; font-size: 12px; color: #c8daff !important;
    margin: 4px 0; display: flex; align-items: center; gap: 6px; word-break: break-all;
}
.page-title { text-align: center; padding: 36px 0 24px; }
.page-title h1 { font-family: 'Playfair Display', serif; font-size: 38px; font-weight: 700; color: #0d2b6e; margin: 0; line-height: 1.2; }
.page-title p { font-size: 16px; color: #5a7abf; margin-top: 8px; font-weight: 500; }
.page-title .pulse-dot { display: inline-block; width: 10px; height: 10px; background: #4a90d9; border-radius: 50%; margin-right: 8px; animation: pulse 2s infinite; }
@keyframes pulse { 0%, 100% { opacity: 1; transform: scale(1); } 50% { opacity: 0.5; transform: scale(0.8); } }

[data-testid="stChatInput"] {
    background: white !important; border-radius: 20px !important;
    border: 2px solid rgba(74,144,217,0.25) !important;
    box-shadow: 0 4px 24px rgba(13,43,110,0.10) !important; padding: 4px 8px !important; transition: all 0.25s ease !important;
}
[data-testid="stChatInput"]:focus-within { border-color: #4a90d9 !important; box-shadow: 0 4px 24px rgba(74,144,217,0.25) !important; }
[data-testid="stChatInput"] textarea { font-family: 'Nunito', sans-serif !important; font-size: 15px !important; color: #1a2b5e !important; }
[data-testid="stChatInput"] textarea::placeholder { color: #8aaee0 !important; }
[data-testid="stChatInput"] button { background: linear-gradient(135deg, #4a90d9, #2563c7) !important; border-radius: 12px !important; border: none !important; box-shadow: 0 2px 8px rgba(37,99,199,0.3) !important; }

[data-testid="stAlert"] { border-radius: 14px !important; border: none !important; font-family: 'Nunito', sans-serif !important; font-size: 14px !important; }
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(74,144,217,0.3); border-radius: 10px; }

.chip {
    background: rgba(255,255,255,0.10); border: 1px solid rgba(255,255,255,0.18);
    border-radius: 20px; padding: 5px 12px; font-size: 12px; color: #c8daff !important; line-height: 1.5;
}
.sidebar-section-label { font-size: 11px; font-weight: 700; letter-spacing: 1.5px; text-transform: uppercase; color: #6a9ad4 !important; margin: 0 0 8px; }

[data-testid="stSidebar"] [data-testid="stDownloadButton"] > button {
    width: 100% !important;
    background: rgba(255,255,255,0.08) !important;
    border: 1px solid rgba(255,255,255,0.2) !important;
    color: #c8daff !important;
    border-radius: 12px !important;
    font-size: 13px !important;
    font-weight: 600 !important;
    margin-top: 6px !important;
    padding: 10px 16px !important;
}
[data-testid="stSidebar"] [data-testid="stDownloadButton"] > button:hover {
    background: rgba(255,255,255,0.15) !important;
    border-color: rgba(100,160,255,0.5) !important;
}

/* Agent panel cards */
.agent-card {
    background: rgba(255,255,255,0.08);
    border: 1px solid rgba(255,255,255,0.15);
    border-radius: 14px;
    padding: 10px 14px;
    margin: 6px 0;
    cursor: pointer;
    transition: all 0.2s ease;
}
.agent-card:hover { background: rgba(255,255,255,0.14); border-color: rgba(100,160,255,0.5); }
.agent-card.active { background: rgba(74,144,217,0.25); border-color: rgba(100,160,255,0.7); }
.agent-card .agent-name { font-size: 14px; font-weight: 700; color: #ffffff !important; }
.agent-card .agent-desc { font-size: 11px; color: #8aaee0 !important; margin-top: 2px; line-height: 1.4; }

/* Agent result panel in main area */
.agent-result-panel {
    background: white;
    border-radius: 20px;
    padding: 28px 32px;
    margin: 16px 0;
    box-shadow: 0 4px 24px rgba(13,43,110,0.10);
    border: 1px solid rgba(74,144,217,0.15);
}
.agent-result-header {
    font-family: 'Playfair Display', serif;
    font-size: 22px;
    color: #0d2b6e;
    font-weight: 700;
    margin-bottom: 4px;
}
.agent-result-sub {
    font-size: 13px;
    color: #5a7abf;
    margin-bottom: 20px;
}
</style>
""", unsafe_allow_html=True)

# ── Session state defaults ────────────────────────────────────
defaults = {
    "agent_statuses": {},   # tracks Done/Running/Not run per agent
    "all_agents_results": {},  # stores results for all agents
    "llm": None,
    "retriever": None,
    "chat_history": [],
    "uploaded_names": [],
    "onboarding_done": False,
    "chroma_dir": None,
    "summaries": {},
    "processing": False,
    "session_id": str(uuid.uuid4()),
    "user_name": "",
    "user_whom": "",
    "user_age": 25,
    "user_conditions": [],
    "user_mood": "Neutral",
    # Agent state
    "active_agent": None,
    "agent_result": None,
    "agent_running": False,
    "smtp_config": None,
    "active_tab": "chat",   # "chat" | "agents"
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# Initialise orchestrator once
if "orchestrator" not in st.session_state:
    st.session_state.orchestrator = AgentOrchestrator()


# ── Export helpers ────────────────────────────────────────────
def build_txt_export(chat_history, user_name, user_mood, user_conditions):
    lines = []
    lines.append("=" * 60)
    lines.append("       MediChat — Conversation Export")
    lines.append("=" * 60)
    lines.append(f"Date     : {datetime.now().strftime('%d %B %Y, %I:%M %p')}")
    if user_name:
        lines.append(f"Patient  : {user_name}")
    if user_mood:
        lines.append(f"Mood     : {user_mood}")
    if user_conditions:
        lines.append(f"Conditions: {', '.join(user_conditions)}")
    lines.append("=" * 60)
    lines.append("")
    for msg in chat_history:
        role = "You" if msg["role"] == "user" else "MediChat"
        lines.append(f"[{role}]")
        lines.append(msg["content"])
        lines.append("")
    lines.append("-" * 60)
    lines.append("For informational purposes only. Always consult your doctor.")
    return "\n".join(lines)


def build_pdf_export(chat_history, user_name, user_mood, user_conditions):
    try:
        from fpdf import FPDF
    except ImportError:
        return None

    class PDF(FPDF):
        def header(self):
            self.set_fill_color(13, 43, 110)
            self.rect(0, 0, 210, 22, 'F')
            self.set_font("Helvetica", "B", 14)
            self.set_text_color(255, 255, 255)
            self.set_y(6)
            self.cell(0, 10, "MediChat  --  Medical Conversation Export", align="C")
            self.ln(18)

        def footer(self):
            self.set_y(-14)
            self.set_font("Helvetica", "I", 8)
            self.set_text_color(150, 150, 150)
            self.cell(0, 10, "For informational purposes only. Always consult your doctor.", align="C")

    try:
        pdf = PDF()
        pdf.set_auto_page_break(auto=True, margin=18)
        pdf.add_page()
        pdf.set_fill_color(232, 240, 255)
        pdf.set_draw_color(180, 200, 240)
        pdf.set_font("Helvetica", "", 9)
        pdf.set_text_color(30, 50, 110)
        meta_lines = [f"Date: {datetime.now().strftime('%d %B %Y, %I:%M %p')}"]
        if user_name:
            meta_lines.append(f"Patient: {user_name}")
        if user_mood:
            meta_lines.append(f"Mood: {user_mood}")
        if user_conditions:
            meta_lines.append(f"Conditions: {', '.join(user_conditions)}")
        meta_text = "   |   ".join(meta_lines)
        pdf.set_x(10)
        pdf.cell(190, 8, meta_text, border=1, fill=True, ln=True)
        pdf.ln(5)
        for msg in chat_history:
            is_user = msg["role"] == "user"
            content = msg["content"].encode("latin-1", errors="replace").decode("latin-1")
            pdf.set_font("Helvetica", "B", 8)
            pdf.set_text_color(100, 120, 180)
            label = f"  {user_name or 'You'}  " if is_user else "  MediChat  "
            if is_user:
                pdf.set_x(10 + 95)
                pdf.cell(95, 5, label, align="R", ln=True)
            else:
                pdf.set_x(10)
                pdf.cell(95, 5, label, align="L", ln=True)
            pdf.set_font("Helvetica", "", 10)
            if is_user:
                pdf.set_fill_color(220, 248, 198)
                pdf.set_text_color(20, 60, 20)
                pdf.set_x(10 + 40)
                pdf.multi_cell(150, 6, content, fill=True, border=0)
            else:
                pdf.set_fill_color(240, 245, 255)
                pdf.set_text_color(13, 43, 110)
                pdf.set_x(10)
                pdf.multi_cell(150, 6, content, fill=True, border=0)
            pdf.ln(3)
        buf = io.BytesIO()
        buf.write(pdf.output())
        buf.seek(0)
        return buf
    except Exception as e:
        print(f"[pdf_export] Error: {e}")
        return None


# ── Welcome message helper ────────────────────────────────────
def _set_welcome_message():
    st.session_state.pop("ob_step", None)
    name = st.session_state.user_name or "there"
    mood = st.session_state.user_mood
    mood_responses = {
        "Anxious":  f"Hey {name}, I can see this might feel a little nerve-wracking — that's completely okay. I'm here to help you understand your documents step by step, calmly and clearly. 💙",
        "Sad":      f"Hi {name}. I'm really glad you're here. Medical stuff can feel heavy sometimes — I'll do my best to explain everything gently and clearly. You're not alone in this. 🤍",
        "Irritable": f"Hey {name}. No fluff — just clear, direct answers. Upload your docs and let's get to it.",
        "Tired":    f"Hi {name}, I'll keep things short and simple. Upload your documents and ask me anything — I'll make it easy to follow. 🌙",
        "Happy":    f"Hey {name}! Great to have you here 😊 Upload your medical documents and I'll help you make sense of everything. Let's go!",
        "Relieved": f"Hi {name}, so glad you're feeling a bit better! Let's look through your documents together — one step at a time. 🌿",
        "Patient":  f"Hi {name}, I appreciate your patience. Upload your documents whenever you're ready and I'll walk through everything clearly with you. 🏥",
        "Strong":   f"Hi {name}! Ready to dive in? Upload your documents and I'll give you a thorough, detailed breakdown of everything in them.",
        "Unwell":   f"Hi {name}, hope you're hanging in there. I'll keep my answers gentle and clear — just upload your documents and ask anything. Take it easy. 💛",
        "Calm":     f"Hi {name}! Great to meet you. Upload your medical documents whenever you're ready and we'll walk through them together.",
        "Confused": f"Hey {name}! Don't worry — I'll explain everything in plain, simple language. No jargon, no confusion. Just upload your documents and ask away! 😊",
        "Neutral":  f"Hi {name}! Upload your medical documents and ask me anything — I'll explain everything clearly and simply.",
    }
    welcome = mood_responses.get(mood, mood_responses["Neutral"])
    st.session_state.chat_history = [{"role": "assistant", "content": welcome}]


# ── 4-Step Onboarding dialog ──────────────────────────────────
@st.dialog("Welcome to MediChat", width="large")
def show_onboarding():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700&family=DM+Sans:wght@300;400;500&display=swap');
    div[data-testid="stDialog"] > div {
        border-radius: 28px !important;
        background: rgba(255,255,255,0.97) !important;
        backdrop-filter: blur(16px) !important;
        max-width: 500px !important;
        box-shadow: 0 24px 60px rgba(0,0,0,0.18) !important;
    }
    div[data-testid="stDialog"] p,
    div[data-testid="stDialog"] label,
    div[data-testid="stDialog"] span:not(.ob-dot),
    div[data-testid="stDialog"] div:not(.ob-dots):not(.ob-dot) {
        color: #1a1830 !important;
        -webkit-text-fill-color: #1a1830 !important;
    }
    div[data-testid="stDialog"] input {
        background: #f5f5f5 !important;
        color: #1a1830 !important;
        -webkit-text-fill-color: #1a1830 !important;
    }
    div[data-testid="stDialog"] button {
        color: #1a1830 !important;
        -webkit-text-fill-color: #1a1830 !important;
    }
    div[data-testid="stDialog"] button[kind="primary"] {
        background: #2451b3 !important;
        color: #ffffff !important;
        -webkit-text-fill-color: #ffffff !important;
    }
    div[data-testid="stDialog"] button[kind="secondary"] {
        background: #f0f0f0 !important;
        color: #1a1830 !important;
    }
    .ob-label { font-size: 11px; letter-spacing: 1.8px; text-transform: uppercase; color: #8aaee0 !important; -webkit-text-fill-color: #8aaee0 !important; font-weight: 500; text-align: center; margin-bottom: 6px; font-family: 'DM Sans', sans-serif; }
    .ob-title { font-family: 'Playfair Display', serif; font-size: 24px; color: #1a1830 !important; -webkit-text-fill-color: #1a1830 !important; text-align: center; margin-bottom: 5px; line-height: 1.3; }
    .ob-sub   { font-size: 13px; color: #9090a0 !important; -webkit-text-fill-color: #9090a0 !important; text-align: center; margin-bottom: 22px; font-weight: 300; line-height: 1.5; font-family: 'DM Sans', sans-serif; }
    .ob-dots { display: flex; align-items: center; justify-content: center; gap: 6px; margin-bottom: 22px; }
    .ob-dot  { height: 6px; border-radius: 4px; display: inline-block; }
    .ob-dot.active { width: 22px; background: #2451b3; }
    .ob-dot.done   { width: 6px;  background: #a3b8e8; }
    .ob-dot.idle   { width: 6px;  background: #ddd; }
    .age-big   { font-family: 'Playfair Display', serif; font-size: 72px; font-weight: 700; color: #2451b3 !important; -webkit-text-fill-color: #2451b3 !important; text-align: center; line-height: 1; letter-spacing: -2px; }
    .age-unit  { text-align: center; font-size: 14px; color: #9090a0 !important; -webkit-text-fill-color: #9090a0 !important; font-weight: 300; font-family: 'DM Sans', sans-serif; }
    </style>
    """, unsafe_allow_html=True)

    if "ob_step" not in st.session_state:
        st.session_state.ob_step = 1
    step = st.session_state.ob_step

    dots_html = '<div class="ob-dots">'
    for i in range(1, 5):
        cls = "active" if i == step else ("done" if i < step else "idle")
        dots_html += f'<span class="ob-dot {cls}"></span>'
    dots_html += '</div>'
    st.markdown(dots_html, unsafe_allow_html=True)

    if step == 1:
        st.markdown(
            '<div class="ob-label">Step 1 of 4</div>'
            '<div class="ob-title">Let\'s get acquainted</div>'
            '<div class="ob-sub">Tell us a bit about yourself and who this report is for</div>',
            unsafe_allow_html=True
        )
        name = st.text_input("Your name (optional)", value=st.session_state.user_name, placeholder="e.g. Aryan")
        st.markdown('<p style="font-size:13px;font-weight:600;color:#6a7ab5;margin:16px 0 12px;">This report is for:</p>', unsafe_allow_html=True)
        whom_options = [("me","🧑","Myself"),("parent","👴","My Parent"),("child","👶","My Child"),("other","🤝","Friend")]
        cols = st.columns(4)
        for i, (key, icon, label) in enumerate(whom_options):
            with cols[i]:
                selected = st.session_state.user_whom == key
                btn_type = "primary" if selected else "secondary"
                if st.button(f"{icon}\n\n{label}", key=f"whom_{key}", use_container_width=True, type=btn_type):
                    st.session_state.user_name = name
                    st.session_state.user_whom = key
                    st.rerun()
        st.markdown("<br>", unsafe_allow_html=True)
        col_skip, col_next = st.columns([1, 1])
        with col_skip:
            if st.button("Skip for now", use_container_width=True, key="ob_skip1"):
                st.session_state.user_name = name
                st.session_state.onboarding_done = True
                _set_welcome_message()
                st.rerun()
        with col_next:
            if st.button("Continue →", use_container_width=True, type="primary", key="ob_next1"):
                st.session_state.user_name = name
                st.session_state.ob_step = 2
                st.rerun()

    elif step == 2:
        age_titles = {"me":"How old are you?","parent":"How old is your parent?","child":"How old is your child?","other":"How old are they?"}
        title = age_titles.get(st.session_state.user_whom, "How old are you?")
        st.markdown(f'<div class="ob-label">Step 2 of 4</div><div class="ob-title">{title}</div><div class="ob-sub">Drag to set the age</div>', unsafe_allow_html=True)
        selected_age = st.slider("Age", min_value=1, max_value=100, value=st.session_state.user_age, key="slider_age_temp", label_visibility="collapsed")
        st.markdown(f'<div class="age-big">{selected_age}</div><div class="age-unit">years old</div><br>', unsafe_allow_html=True)
        col_back, col_next = st.columns([1, 1])
        with col_back:
            if st.button("← Back", use_container_width=True, key="ob_back2"):
                st.session_state.user_age = selected_age
                st.session_state.ob_step = 1
                st.rerun()
        with col_next:
            if st.button("Continue →", use_container_width=True, type="primary", key="ob_next2"):
                st.session_state.user_age = selected_age
                st.session_state.ob_step = 3
                st.rerun()

    elif step == 3:
        st.markdown('<div class="ob-label">Step 3 of 4</div><div class="ob-title">Any existing conditions?</div><div class="ob-sub">Select all that apply — helps us explain your report more accurately</div>', unsafe_allow_html=True)
        cond_options = [("diabetes","Diabetes","🩸"),("hypertension","Hypertension","💓"),("heart","Heart Disease","❤️"),("thyroid","Thyroid","🦋"),("asthma","Asthma","🫁"),("neurological","Neurological","🧠"),("none","None","✅"),("other","Other","➕")]
        current = set(st.session_state.user_conditions)
        cols = st.columns(2)
        for i, (key, label, icon) in enumerate(cond_options):
            with cols[i % 2]:
                selected = key in current
                btn_type = "primary" if selected else "secondary"
                if st.button(f"{icon} {label}", key=f"cond_{key}", use_container_width=True, type=btn_type):
                    if key == "none":
                        st.session_state.user_conditions = ["none"]
                    else:
                        new = set(st.session_state.user_conditions) - {"none"}
                        if key in new: new.remove(key)
                        else: new.add(key)
                        st.session_state.user_conditions = list(new)
                    st.rerun()
        col_back, col_skip, col_next = st.columns([1, 1, 1])
        with col_back:
            if st.button("← Back", use_container_width=True, key="ob_back3"):
                st.session_state.ob_step = 2
                st.rerun()
        with col_skip:
            if st.button("Skip", use_container_width=True, key="ob_skip3"):
                st.session_state.user_conditions = []
                st.session_state.ob_step = 4
                st.rerun()
        with col_next:
            if st.button("Continue →", use_container_width=True, type="primary", key="ob_next3"):
                st.session_state.ob_step = 4
                st.rerun()

    elif step == 4:
        st.markdown('<div class="ob-label">Step 4 of 4</div><div class="ob-title">How are you feeling?</div><div class="ob-sub">We\'ll match our tone to how you feel right now</div>', unsafe_allow_html=True)
        all_moods = [("Happy","😊"),("Relieved","😌"),("Patient","🏥"),("Neutral","😐"),("Anxious","😟"),("Sad","😔"),("Tired","😴"),("Calm","🧘"),("Strong","💪"),("Unwell","🤒"),("Confused","😕"),("Irritable","😤")]
        current_mood = st.session_state.user_mood
        cols_per_row = 4
        rows = [all_moods[i:i+cols_per_row] for i in range(0, len(all_moods), cols_per_row)]
        for row in rows:
            cols = st.columns(len(row))
            for j, (mood_name, emoji) in enumerate(row):
                with cols[j]:
                    selected = mood_name == current_mood
                    btn_type = "primary" if selected else "secondary"
                    if st.button(f"{emoji}\n\n{mood_name}", key=f"mood_{mood_name}", use_container_width=True, type=btn_type):
                        st.session_state.user_mood = mood_name
                        st.rerun()
        col_back, col_done = st.columns([1, 1])
        with col_back:
            if st.button("← Back", use_container_width=True, key="ob_back4"):
                st.session_state.ob_step = 3
                st.rerun()
        with col_done:
            if st.button("Get started", use_container_width=True, type="primary", key="ob_done"):
                st.session_state.onboarding_done = True
                _set_welcome_message()
                st.rerun()


# ── Show onboarding if not done ───────────────────────────────
if not st.session_state.onboarding_done:
    if "page_loaded" not in st.session_state:
        st.session_state.page_loaded = True
        st.rerun()
    show_onboarding()


# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="sidebar-logo">
        <span class="logo-icon">🏥</span>
        <div class="logo-title">MediChat</div>
        <div class="logo-subtitle">AI Medical Assistant</div>
    </div>
    """, unsafe_allow_html=True)

    if st.session_state.onboarding_done and st.session_state.user_name:
        mood_emoji_map = {"Happy":"😊","Relieved":"😌","Patient":"🏥","Sad":"😔","Neutral":"😐","Anxious":"😟","Irritable":"😤","Tired":"😴","Strong":"💪","Unwell":"🤒","Calm":"🧘","Confused":"😕"}
        mood_emoji = mood_emoji_map.get(st.session_state.user_mood, "😐")
        whom_label = {"me":"Myself","parent":"Parent","child":"Child","other":"Other"}.get(st.session_state.user_whom, "")
        st.markdown(f"""
        <div style="background:rgba(255,255,255,0.08);border-radius:14px;padding:10px 14px;
            margin-bottom:16px;border:1px solid rgba(255,255,255,0.12);">
            <div style="font-size:12px;color:#8aaee0;margin-bottom:2px;">Logged in as</div>
            <div style="font-size:15px;color:#fff;font-weight:600;">
                {html.escape(st.session_state.user_name)} &nbsp;{mood_emoji} {html.escape(st.session_state.user_mood)}
            </div>
            {"<div style='font-size:11px;color:#6a9ad4;margin-top:2px;'>Report for: " + whom_label + " · Age: " + str(st.session_state.user_age) + "</div>" if whom_label else ""}
        </div>
        """, unsafe_allow_html=True)

    # ── Tab switcher: Chat vs Agents ──────────────────────────
    st.markdown('<p class="sidebar-section-label">Mode</p>', unsafe_allow_html=True)
    tab_col1, tab_col2 = st.columns(2)
    with tab_col1:
        chat_type = "primary" if st.session_state.active_tab == "chat" else "secondary"
        if st.button("💬 Chat", use_container_width=True, type=chat_type, key="tab_chat"):
            st.session_state.active_tab = "chat"
            st.session_state.active_agent = None
            st.session_state.agent_result = None
            st.rerun()
    with tab_col2:
        agent_type = "primary" if st.session_state.active_tab == "agents" else "secondary"
        if st.button("🤖 Agents", use_container_width=True, type=agent_type, key="tab_agents"):
            st.session_state.active_tab = "agents"
            st.rerun()

    st.markdown("---")

    # ── Document upload (always visible) ─────────────────────
    st.markdown('<p class="sidebar-section-label">Upload Documents</p>', unsafe_allow_html=True)
    uploaded_files = st.file_uploader(
        "Drop your files here",
        type=["pdf", "jpg", "jpeg", "png", "webp", "docx", "txt"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if uploaded_files:
        st.markdown(f"<p style='font-size:13px; color:#8aaee0; margin:8px 0 4px;'>📎 {len(uploaded_files)} file(s) ready to process</p>", unsafe_allow_html=True)
        for f in uploaded_files:
            st.markdown(f'<div class="doc-pill">📄 {html.escape(f.name)}</div>', unsafe_allow_html=True)

    if uploaded_files and st.button("✨  Process All Documents"):
        st.session_state.processing = True
        st.session_state.files_to_process = [{"name": f.name, "data": f.read()} for f in uploaded_files]
        st.rerun()

    if st.session_state.uploaded_names:
        st.markdown("---")
        st.markdown('<p class="sidebar-section-label">Loaded Documents</p>', unsafe_allow_html=True)
        for name in st.session_state.uploaded_names:
            st.markdown(f'<div class="doc-pill">✅ {html.escape(name)}</div>', unsafe_allow_html=True)

    # ── Agent panel (only when Agents tab active) ─────────────
    if st.session_state.active_tab == "agents":
        st.markdown("---")
        st.markdown('<p class="sidebar-section-label">Choose an Agent</p>', unsafe_allow_html=True)

        orchestrator: AgentOrchestrator = st.session_state.orchestrator
        descriptions = orchestrator.agent_descriptions

        for agent_name, agent_desc in descriptions.items():
            is_active = st.session_state.active_agent == agent_name
            card_class = "agent-card active" if is_active else "agent-card"
            st.markdown(f"""
            <div class="{card_class}">
                <div class="agent-name">{agent_name}</div>
                <div class="agent-desc">{agent_desc}</div>
            </div>
            """, unsafe_allow_html=True)
            btn_label = "▶ Running…" if (is_active and st.session_state.agent_running) else ("✓ Selected" if is_active else "Run Agent")
            if st.button(btn_label, key=f"run_{agent_name}", use_container_width=True):
                if st.session_state.retriever is None:
                    st.warning("⚠️ Please upload and process documents first.")
                else:
                    st.session_state.active_agent = agent_name
                    st.session_state.agent_result = None
                    st.session_state.agent_running = True
                    st.rerun()

        # ── Email SMTP config (only shown when email agent selected) ──
        if st.session_state.active_agent == "📧 Email Report":
            st.markdown("---")
            st.markdown('<p class="sidebar-section-label">Email Configuration</p>', unsafe_allow_html=True)

            with st.expander("⚙️ SMTP Settings", expanded=True):

                sender_email = st.text_input("Sender Gmail", value=os.getenv("SMTP_SENDER", ""), placeholder="yourname@gmail.com", key="smtp_sender")
                app_password = st.text_input("App Password", value=os.getenv("SMTP_PASSWORD", ""), type="password", placeholder="16-char app password", key="smtp_pass", help="Generate at myaccount.google.com → Security → App passwords")
                recipient_email = st.text_input("Recipient Email", placeholder="doctor@hospital.com", key="smtp_recipient")

                if sender_email and app_password:
                    st.session_state.smtp_config = {"sender_email": sender_email, "app_password": app_password}
                    st.session_state.recipient_email = recipient_email
                else:
                    st.session_state.smtp_config = None
                    st.session_state.recipient_email = recipient_email if "smtp_recipient" in st.session_state else ""

    # ── Export (chat tab only) ────────────────────────────────
    if st.session_state.active_tab == "chat" and st.session_state.chat_history and len(st.session_state.chat_history) > 1:
        st.markdown("---")
        st.markdown('<p class="sidebar-section-label">Export Chat</p>', unsafe_allow_html=True)
        fname_base = f"medichat_{datetime.now().strftime('%Y%m%d_%H%M')}"
        txt_content = build_txt_export(st.session_state.chat_history, st.session_state.user_name, st.session_state.user_mood, st.session_state.user_conditions)
        st.download_button(label="📄  Download as Text (.txt)", data=txt_content.encode("utf-8"), file_name=f"{fname_base}.txt", mime="text/plain", use_container_width=True)
        pdf_buf = build_pdf_export(st.session_state.chat_history, st.session_state.user_name, st.session_state.user_mood, st.session_state.user_conditions)
        if pdf_buf:
            st.download_button(label="📑  Download as PDF", data=pdf_buf, file_name=f"{fname_base}.pdf", mime="application/pdf", use_container_width=True)

    if st.session_state.active_tab == "chat":
        st.markdown("---")
        st.markdown('<p class="sidebar-section-label">Try asking</p>', unsafe_allow_html=True)
        for q in ["What does my creatinine level mean?","What medications were prescribed?","Is my blood sugar normal?","What was the diagnosis?","Which values are abnormal?","What follow-up is needed?"]:
            st.markdown(f'<div class="chip">💬 {q}</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<p style='font-size:11px; color:#4a6a9a; text-align:center; line-height:1.6;'>For informational purposes only.<br/>Always consult your doctor.</p>", unsafe_allow_html=True)


# ── Document processing (always runs, regardless of tab) ─────
if st.session_state.get("processing_error"):
    st.error(f"❌ {st.session_state.processing_error}")
    if st.button("Clear error"):
        st.session_state.processing_error = None
        st.rerun()

if st.session_state.get("processing"):
    st.markdown("""
    <style>
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to   { opacity: 1; transform: translateY(0); }
    }
    @keyframes bar {
        0%   { width: 0%; }
        100% { width: 100%; }
    }
    .processing-overlay {
        position: fixed; top: 0; left: 0;
        width: 100vw; height: 100vh;
        background: rgba(240, 244, 255, 0.97);
        backdrop-filter: blur(8px);
        z-index: 9999;
        display: flex; flex-direction: column;
        align-items: center; justify-content: center;
        animation: fadeIn 0.4s ease;
    }
    .processing-card {
        background: white;
        border-radius: 28px;
        padding: 48px 56px;
        text-align: center;
        box-shadow: 0 24px 64px rgba(13,43,110,0.15);
        border: 1px solid rgba(74,144,217,0.2);
        max-width: 460px;
        width: 90%;
    }
    .spinner-ring {
        width: 72px; height: 72px;
        border: 5px solid #e8eeff;
        border-top: 5px solid #2451b3;
        border-radius: 50%;
        animation: spin 1s linear infinite;
        margin: 0 auto 24px;
    }
    .processing-title {
        font-family: 'Playfair Display', serif;
        font-size: 26px; font-weight: 700;
        color: #0d2b6e; margin-bottom: 8px;
    }
    .processing-sub {
        font-size: 14px; color: #5a7abf;
        margin-bottom: 28px; line-height: 1.6;
    }
    .progress-bar-wrap {
        background: #e8eeff;
        border-radius: 20px;
        height: 6px;
        overflow: hidden;
        margin-bottom: 20px;
    }
    .progress-bar-fill {
        height: 6px;
        background: linear-gradient(90deg, #4a90d9, #2451b3);
        border-radius: 20px;
        animation: bar 3s ease-in-out infinite alternate;
    }
    .processing-steps {
        display: flex;
        flex-direction: column;
        gap: 10px;
        text-align: left;
    }
    .step-item {
        display: flex;
        align-items: center;
        gap: 10px;
        font-size: 13px;
        color: #5a7abf;
        font-family: 'Nunito', sans-serif;
    }
    .step-dot {
        width: 8px; height: 8px;
        background: #4a90d9;
        border-radius: 50%;
        flex-shrink: 0;
        animation: pulse 1.5s infinite;
    }
    </style>

    <div class="processing-overlay">
        <div class="processing-card">
            <div class="spinner-ring"></div>
            <div class="processing-title">Analysing your documents</div>
            <div class="processing-sub">
                Please wait while we read, extract,<br>and index all your medical content.
            </div>
            <div class="progress-bar-wrap">
                <div class="progress-bar-fill"></div>
            </div>
            <div class="processing-steps">
                <div class="step-item"><div class="step-dot"></div>Reading file contents</div>
                <div class="step-item"><div class="step-dot"></div>Extracting text and lab values</div>
                <div class="step-item"><div class="step-dot"></div>Building searchable index</div>
                <div class="step-item"><div class="step-dot"></div>Generating document summary</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    tmp_paths, original_names = [], []
    for file_info in st.session_state.get("files_to_process", []):
        suffix = Path(file_info["name"]).suffix
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(file_info["data"])
            tmp_paths.append(tmp.name)
            original_names.append(file_info["name"])
    try:
        if st.session_state.chroma_dir and os.path.exists(st.session_state.chroma_dir):
            shutil.rmtree(st.session_state.chroma_dir, ignore_errors=True)
        vectorstore, chroma_dir, summaries = ingest_multiple_files(
            tmp_paths, original_names, session_id=st.session_state.session_id
        )
        llm, retriever = build_qa_chain(vectorstore, mood=st.session_state.user_mood)
        st.session_state.llm = llm
        st.session_state.retriever = retriever
        st.session_state.chroma_dir = chroma_dir
        st.session_state.uploaded_names = original_names
        st.session_state.summaries = summaries
        st.session_state.chat_history = []
        st.success("✅ Documents processed successfully!")
    except Exception as e:
        st.session_state.processing_error = str(e)
        st.error(f"Failed to process documents: {e}")
    finally:
        for path in tmp_paths:
            if os.path.exists(path):
                os.unlink(path)
    st.session_state.processing = False
    st.session_state.files_to_process = []
    st.rerun()

# ── Main area ─────────────────────────────────────────────────
col1, col2, col3 = st.columns([1, 3, 1])
with col2:
    st.markdown("""
    <div class="page-title">
        <h1>🩺 Medical Document Assistant</h1>
        <p><span class="pulse-dot"></span>Your health, explained simply — ask me anything</p>
    </div>
    """, unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════
    #  AGENTS TAB
    # ══════════════════════════════════════════════════════════
    if st.session_state.active_tab == "agents":

        if st.session_state.retriever is None:
            st.markdown("""
            <div style="text-align:center; margin-top:32px; padding:32px;
                background:white; border-radius:20px;
                box-shadow:0 4px 20px rgba(13,43,110,0.08);
                border:1px dashed rgba(74,144,217,0.3);">
                <div style="font-size:48px; margin-bottom:12px;">🤖</div>
                <p style="color:#5a7abf; font-size:15px; margin:0; font-weight:600;">
                    Upload & process your documents first, then select an agent from the sidebar.
                </p>
            </div>
            """, unsafe_allow_html=True)

        elif st.session_state.active_agent is None:
            orchestrator: AgentOrchestrator = st.session_state.orchestrator

            agent_icons = {
                "📋 Summarizer":         ("📋", "#4a90d9"),
                "🔬 Lab Analyzer":       ("🔬", "#7c5cbf"),
                "💊 Medication Agent":   ("💊", "#2eaa5e"),
                "📅 Appointment Reminder": ("📅", "#d97b4a"),
                "📧 Email Report":       ("📧", "#c72563"),
            }

            status_style = {
                "done":    ("✅ Done",    "#2eaa5e", "#edfdf4"),
                "running": ("⏳ Running…","#d97b4a", "#fff8f0"),
                "":        ("— Not run", "#aab8d4", "#f5f7ff"),
            }

            # ── Run All Agents button ─────────────────────────
            st.markdown("<div style='text-align:center; margin-bottom:20px;'>", unsafe_allow_html=True)
            if st.button("⚡ Run All Agents", use_container_width=False, key="run_all"):
                if st.session_state.retriever is None:
                    st.warning("⚠️ Please upload and process documents first.")
                else:
                    for aname in orchestrator.agent_names:
                        st.session_state.agent_statuses[aname] = "running"
                    with st.spinner("Running all agents — this may take a minute…"):
                        for aname in orchestrator.agent_names:
                            try:
                                res = orchestrator.run(
                                    agent_name=aname,
                                    retriever=st.session_state.retriever,
                                    user_name=st.session_state.user_name,
                                    mood=st.session_state.user_mood,
                                    user_whom=st.session_state.user_whom,
                                    user_age=st.session_state.user_age,
                                    user_conditions=st.session_state.user_conditions,
                                    summaries=st.session_state.summaries,
                                    smtp_config=st.session_state.get("smtp_config"),
                                    recipient_email=st.session_state.get("recipient_email", ""),
                                )
                                st.session_state.all_agents_results[aname] = res
                                st.session_state.agent_statuses[aname] = "done"
                            except Exception as e:
                                st.session_state.all_agents_results[aname] = f"❌ Error: {e}"
                                st.session_state.agent_statuses[aname] = "done"
                    st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)

            # ── If Run All results exist, show in tabs ────────
            if st.session_state.all_agents_results:
                result_tabs = st.tabs(list(st.session_state.all_agents_results.keys()))
                for tab, (aname, ares) in zip(result_tabs, st.session_state.all_agents_results.items()):
                    with tab:
                        st.markdown(ares)
                st.markdown("---")

            # ── Clickable agent cards grid ────────────────────
            st.markdown("<p style='text-align:center; color:#5a7abf; font-size:14px; margin-bottom:16px;'>Or click a card to run a single agent</p>", unsafe_allow_html=True)

            agents_list = list(orchestrator.agent_descriptions.items())
            # Row 1: 3 cards
            row1 = st.columns(3)
            # Row 2: 2 cards centred
            row2_spacer1, row2_col1, row2_col2, row2_spacer2 = st.columns([0.5, 1, 1, 0.5])
            grid_cols = row1 + [row2_col1, row2_col2]

            for idx, (agent_name, agent_desc) in enumerate(agents_list):
                icon, color = agent_icons.get(agent_name, ("🤖", "#4a90d9"))
                status_key = st.session_state.agent_statuses.get(agent_name, "")
                status_label, status_color, status_bg = status_style.get(status_key, status_style[""])

                with grid_cols[idx]:
                    # Status badge + card UI
                    st.markdown(f"""
                    <div style="background:white; border-radius:18px; padding:22px 16px 14px;
                        text-align:center; box-shadow:0 4px 20px rgba(13,43,110,0.10);
                        border:2px solid {'rgba(74,144,217,0.35)' if status_key == 'done' else 'rgba(74,144,217,0.12)'};
                        margin-bottom:12px; min-height:160px; position:relative;
                        transition: all 0.2s ease;">
                        <div style="position:absolute; top:10px; right:12px;
                            background:{status_bg}; color:{status_color};
                            font-size:10px; font-weight:700; padding:3px 8px;
                            border-radius:20px; letter-spacing:0.5px;">
                            {status_label}
                        </div>
                        <div style="font-size:38px; margin-bottom:10px;">{icon}</div>
                        <div style="font-weight:700; color:#0d2b6e; font-size:14px; margin-bottom:6px;">{agent_name}</div>
                        <div style="font-size:12px; color:#5a7abf; line-height:1.5;">{agent_desc}</div>
                    </div>
                    """, unsafe_allow_html=True)

                    # Clickable button styled as "Run"
                    btn_label = "✅ View Result" if status_key == "done" else "▶ Run Agent"
                    if st.button(btn_label, key=f"card_btn_{agent_name}", use_container_width=True):
                        if st.session_state.retriever is None:
                            st.warning("⚠️ Upload and process documents first.")
                        else:
                            # If already has a result from Run All, show it directly
                            if agent_name in st.session_state.all_agents_results:
                                st.session_state.agent_result = st.session_state.all_agents_results[agent_name]
                                st.session_state.agent_running = False
                            else:
                                st.session_state.agent_result = None
                                st.session_state.agent_running = True
                            st.session_state.active_agent = agent_name
                            st.session_state.agent_statuses[agent_name] = "running"
                            st.rerun()

        else:
            # Run the agent if result not yet computed
            if st.session_state.agent_running and st.session_state.agent_result is None:
                with st.spinner(f"Running {st.session_state.active_agent}…"):
                    try:
                        orchestrator: AgentOrchestrator = st.session_state.orchestrator
                        result = orchestrator.run(
                            agent_name=st.session_state.active_agent,
                            retriever=st.session_state.retriever,
                            user_name=st.session_state.user_name,
                            mood=st.session_state.user_mood,
                            user_whom=st.session_state.user_whom,
                            user_age=st.session_state.user_age,
                            user_conditions=st.session_state.user_conditions,
                            summaries=st.session_state.summaries,
                            smtp_config=st.session_state.get("smtp_config"),
                            recipient_email=st.session_state.get("recipient_email", ""),
                        )
                        st.session_state.agent_result = result
                    except Exception as e:
                        st.session_state.agent_result = f"❌ Agent error: {e}"
                    finally:
                        st.session_state.agent_running = False
                st.rerun()

            # Display agent result
            if st.session_state.agent_result:
                agent_name = st.session_state.active_agent
                descriptions = st.session_state.orchestrator.agent_descriptions

                st.markdown(f"""
                <div class="agent-result-panel">
                    <div class="agent-result-header">{agent_name}</div>
                    <div class="agent-result-sub">{descriptions.get(agent_name, '')}</div>
                </div>
                """, unsafe_allow_html=True)

                # Render the result nicely
                st.markdown(st.session_state.agent_result)

                st.markdown("<br>", unsafe_allow_html=True)
                c1, c2, c3 = st.columns([1, 1, 1])
                with c1:
                    # Download agent result as txt
                    st.download_button(
                        label="📄 Download Result",
                        data=st.session_state.agent_result.encode("utf-8"),
                        file_name=f"medichat_{agent_name.replace(' ', '_').lower()}_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
                        mime="text/plain",
                        use_container_width=True,
                    )
                with c2:
                    if st.button("🔄 Run Again", use_container_width=True):
                        st.session_state.agent_result = None
                        st.session_state.agent_running = True
                        st.rerun()
                with c3:
                    if st.button("← Back to Agents", use_container_width=True):
                        st.session_state.agent_statuses[st.session_state.active_agent] = "done"
                        st.session_state.all_agents_results[st.session_state.active_agent] = st.session_state.agent_result
                        st.session_state.active_agent = None
                        st.session_state.agent_result = None
                        st.rerun()

    # ══════════════════════════════════════════════════════════
    #  CHAT TAB
    # ══════════════════════════════════════════════════════════
    else:
        for message in st.session_state.chat_history:
            if message["role"] == "user":
                st.markdown(
                    f'<div style="display:flex; justify-content:flex-end; margin:8px 0;">'
                    f'<div style="background:#dcf8c6; color:#111; border-radius:18px 18px 4px 18px;'
                    f'padding:10px 16px; max-width:70%; font-size:15px; line-height:1.6;'
                    f'box-shadow:0 1px 2px rgba(0,0,0,0.15); font-family:Nunito,sans-serif;">'
                    f'{html.escape(message["content"])}</div>'
                    f'<div style="width:36px; height:36px; background:linear-gradient(135deg,#1a3d8f,#4a90d9);'
                    f'border-radius:50%; display:flex; align-items:center; justify-content:center;'
                    f'font-size:18px; margin-left:8px; flex-shrink:0;">&#128100;</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
            else:
                safe_content = _render_message(message["content"])
                st.markdown(
                    f'<div style="display:flex; justify-content:flex-start; margin:8px 0;">'
                    f'<div style="width:36px; height:36px; background:linear-gradient(135deg,#ffffff,#e8f0ff);'
                    f'border-radius:50%; display:flex; align-items:center; justify-content:center;'
                    f'font-size:18px; margin-right:8px; flex-shrink:0;'
                    f'box-shadow:0 2px 8px rgba(13,43,110,0.15);">&#127973;</div>'
                    f'<div style="background:#ffffff; color:#111; border-radius:18px 18px 18px 4px;'
                    f'padding:12px 16px; max-width:70%; font-size:15px; line-height:1.7;'
                    f'box-shadow:0 1px 2px rgba(0,0,0,0.15); font-family:Nunito,sans-serif;">'
                    f'{safe_content}</div></div>',
                    unsafe_allow_html=True,
                )


        if st.session_state.get("summaries"):
            with st.expander("📋 Document Summaries", expanded=False):
                for name, summary in st.session_state.summaries.items():
                    st.markdown(f"**{name}**")
                    st.markdown(summary)
                    st.markdown("---")

        if st.session_state.llm is None:
            st.markdown("""
            <div style="text-align:center; margin-top:32px; padding:20px;
                background:white; border-radius:20px;
                box-shadow:0 4px 20px rgba(13,43,110,0.08);
                border:1px dashed rgba(74,144,217,0.3);">
                <p style="color:#5a7abf; font-size:15px; margin:0;">
                    📤 Upload your medical documents from the sidebar to start chatting!
                </p>
            </div>
            """, unsafe_allow_html=True)
        else:
            user_question = st.chat_input("Ask about your medical documents…")
            if user_question:
                history_before_question = list(st.session_state.chat_history)
                st.session_state.chat_history.append({"role": "user", "content": user_question})
                with st.spinner("Reading your documents…"):
                    try:
                        answer = get_answer(
                            llm=st.session_state.llm,
                            retriever=st.session_state.retriever,
                            question=user_question,
                            chat_history=history_before_question,
                            mood=st.session_state.user_mood,
                            user_name=st.session_state.user_name,
                            user_conditions=st.session_state.user_conditions,
                            user_whom=st.session_state.user_whom,
                            user_age=st.session_state.user_age,
                            summaries=st.session_state.get("summaries", {}),
                        )
                    except Exception as e:
                        answer = f"Something went wrong while reading your documents. Please try again. (Error: {e})"
                st.session_state.chat_history.append({"role": "assistant", "content": answer})
                st.rerun()
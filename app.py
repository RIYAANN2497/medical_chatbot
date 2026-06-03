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
import streamlit.components.v1 as components

load_dotenv()

import re as _re

# ── Markdown → safe HTML renderer ────────────────────────────
def _render_message(raw: str) -> str:
    import re as _re2
    parts = []
    in_list = False
    for line in raw.split("\n"):
        s = line.strip()
        if not s:
            if in_list:
                parts.append("</ul>")
                in_list = False
            continue
        
        is_bullet = s.startswith("- ") or s.startswith("* ") or bool(_re2.match(r"^\d+\.\s", s))
        
        if is_bullet:
            if not in_list:
                parts.append("<ul style='margin:6px 0;padding-left:20px;list-style:disc;'>")
                in_list = True
            text_raw = _re2.sub(r"^[-*]\s|^\d+\.\s", "", s)
            text = _re2.sub(r"\*\*(.*?)\*\*", r"<strong>\1</strong>", html.escape(text_raw))
            parts.append(f"<li style='margin-bottom:5px;'>{text}</li>")
        else:
            if in_list:
                parts.append("</ul>")
                in_list = False
            # Only style ## and ### headers — don't style **bold** lines as headers
            if s.startswith("## "):
                text = html.escape(s[3:])
                parts.append(f"<p style='font-size:15px;font-weight:800;color:#0d2b6e;margin:14px 0 4px;'>{text}</p>")
            elif s.startswith("### "):
                text = html.escape(s[4:])
                parts.append(f"<p style='font-size:14px;font-weight:700;color:#2451b3;margin:10px 0 4px;'>{text}</p>")
            else:
                # Render inline bold but keep it as a normal paragraph
                text = _re2.sub(r"\*\*(.*?)\*\*", r"<strong>\1</strong>", html.escape(s))
                parts.append(f"<p style='margin:3px 0;line-height:1.7;'>{text}</p>")
    
    if in_list:
        parts.append("</ul>")
    return "".join(parts)

# ── Appointment renderer ──────────────────────────────────────
def _render_appointment(raw: str):
    import re as re2
    sections = re2.split(r'\n(?=\*\*[📅🧪💊💡])', raw.strip())
    colors_map = {
        "📅": ("#eef3ff", "#2451b3", "#dce8ff"),
        "🧪": ("#f0fff4", "#1a8a4a", "#c6f0d8"),
        "💊": ("#fff8f0", "#c06000", "#ffe0b2"),
        "💡": ("#f5f0ff", "#6a3db8", "#e2d4f8"),
    }
    for section in sections:
        section = section.strip()
        if not section:
            continue
        emoji = next((c for c in section[:5] if c in colors_map), "📅")
        bg, accent, border = colors_map.get(emoji, ("#f8f9ff", "#2451b3", "#dce8ff"))
        lines = section.split("\n")
        title_raw = lines[0].strip().strip("*").strip()
        body_lines = lines[1:]

        items_html = ""
        current_appt = {}
        appt_blocks = []

        for line in body_lines:
            s = line.strip()
            if not s or s == "---":
                if current_appt:
                    appt_blocks.append(current_appt)
                    current_appt = {}
                continue
            bold_date = re2.match(r'^\*\*(.+)\*\*$', s)
            if bold_date:
                if current_appt:
                    appt_blocks.append(current_appt)
                current_appt = {"date": bold_date.group(1), "details": []}
            elif s.startswith("- ") and current_appt:
                current_appt["details"].append(s[2:])
            elif s.startswith("- "):
                items_html += f"<div style='display:flex;gap:8px;margin:6px 0;'><span style='color:{accent};font-weight:700;'>•</span><span style='color:#1a1a2e;font-size:14px;'>{html.escape(s[2:])}</span></div>"

        if current_appt:
            appt_blocks.append(current_appt)

        for appt in appt_blocks:
            details_html = ""
            for d in appt.get("details", []):
                key, _, val = d.partition(":")
                if val:
                    details_html += f"<div style='font-size:13px;color:#555;margin:3px 0;'><span style='font-weight:700;color:{accent};'>{html.escape(key.strip())}:</span> {html.escape(val.strip())}</div>"
                else:
                    details_html += f"<div style='font-size:13px;color:#555;margin:3px 0;'>{html.escape(d)}</div>"
            items_html += f"""
            <div style='background:white;border-radius:12px;padding:12px 16px;margin:8px 0;
                border-left:4px solid {accent};box-shadow:0 2px 8px rgba(0,0,0,0.06);'>
                <div style='font-size:14px;font-weight:800;color:{accent};margin-bottom:6px;'>📅 {html.escape(appt["date"])}</div>
                {details_html}
            </div>"""

        st.markdown(f"""
        <div style='background:{bg};border:1.5px solid {border};border-radius:20px;
            padding:20px 24px;margin:12px 0;'>
            <div style='font-size:16px;font-weight:800;color:{accent};margin-bottom:12px;
                padding-bottom:8px;border-bottom:1.5px solid {border};'>{html.escape(title_raw)}</div>
            {items_html}
        </div>
        """, unsafe_allow_html=True)
        
# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="MediChat — Your Medical Assistant",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ── Global CSS ────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Nunito:wght@400;500;600;700;800&family=Playfair+Display:wght@600;700&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] { font-family: 'Nunito', sans-serif; background-color: #f0f4ff; }
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
header { visibility: hidden; }
.block-container { padding: 0 !important; max-width: 100% !important; }
.stApp { background: #f0f4ff; min-height: 100vh; }

/* ── Sidebar ── */
[data-testid="collapsedControl"] { display: none !important; }
[data-testid="stSidebarCollapseButton"] { display: none !important; }

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #071a4a 0%, #0d2b6e 60%, #1a3d8f 100%) !important;
    border-right: none !important;
    box-shadow: 8px 0 40px rgba(7,26,74,0.35) !important;
    min-width: 300px !important;
    max-width: 300px !important;
}
[data-testid="stSidebar"] > div:first-child { padding: 0 !important; }
[data-testid="stSidebar"] * { color: #dce8ff !important; }
[data-testid="stFileUploader"] *:not(button) {
    color: #0d2b6e !important; opacity: 1 !important;
    -webkit-text-fill-color: #0d2b6e !important;
}
[data-testid="stFileUploader"] {
    background: rgba(255,255,255,0.08) !important;
    border: 2px dashed rgba(74,144,217,0.6) !important;
    border-radius: 16px !important;
    padding: 8px !important;
}
[data-testid="stFileUploader"] * {
    color: #c8daff !important;
    -webkit-text-fill-color: #c8daff !important;
}
[data-testid="stFileUploader"] button {
    background: rgba(74,144,217,0.25) !important;
    border: 1px solid rgba(74,144,217,0.5) !important;
    border-radius: 8px !important;
    color: #ffffff !important;
    -webkit-text-fill-color: #ffffff !important;
    font-weight: 700 !important;
    font-size: 12px !important;
    padding: 4px 12px !important;
    width: auto !important;
    height: auto !important;
    margin: 0 !important;
    border-radius: 8px !important;
}
[data-testid="stFileUploader"] small {
    display: none !important;
}
[data-testid="stSidebar"] .stMarkdown p,
[data-testid="stSidebar"] .stMarkdown li { color: #9ab4e0 !important; font-size: 13px; line-height: 1.7; }

/* Section labels */
.sidebar-section-label {
    font-size: 9px; font-weight: 800; letter-spacing: 2.5px;
    text-transform: uppercase; color: #4a6aaa !important;
    margin: 0 0 8px; display: block;
}

/* File uploader */
[data-testid="stFileUploader"] {
    background: rgba(5,15,50,0.6) !important;
    border: 2px dashed rgba(74,144,217,0.5) !important;
    border-radius: 16px !important;
    padding: 12px !important;
}
[data-testid="stFileUploader"]:hover {
    border-color: rgba(74,144,217,0.9) !important;
}
[data-testid="stFileUploader"] button {
    background: rgba(74,144,217,0.3) !important;
    border: 1px solid rgba(74,144,217,0.6) !important;
    color: #ffffff !important;
    -webkit-text-fill-color: #ffffff !important;
    border-radius: 10px !important;
    font-weight: 700 !important;
    font-size: 12px !important;
}
[data-testid="stFileUploader"] small,
[data-testid="stFileUploader"] span[class*="instructions"] {
    display: none !important;
    visibility: hidden !important;
    height: 0 !important;
    overflow: hidden !important;
}
[data-testid="stFileUploader"] * { color: ##0d2b6e !important; -webkit-text-fill-color: ##0d2b6e !important; }
[data-testid="stFileUploader"] button * { color: #ffffff !important; -webkit-text-fill-color: #ffffff !important; }

[data-testid="stFileUploader"] svg {
    display: none !important;
}
[data-testid="stFileUploader"] [class*="fileIcon"],
[data-testid="stFileUploader"] [class*="thumbnail"] {
    display: none !important;
}

/* Process button */
[data-testid="stSidebar"] .stButton > button {
    width: 100%;
    background: linear-gradient(135deg, #3a7bd5, #1a4fa8) !important;
    color: white !important; border: none !important;
    border-radius: 14px !important; padding: 15px 16px !important;
    font-weight: 800 !important; font-size: 14px !important;
    letter-spacing: 0.3px; margin-top: 12px;
    box-shadow: 0 6px 20px rgba(26,79,168,0.45) !important;
    transition: all 0.2s ease !important;
    white-space: nowrap !important;
    min-height: 56px !important;
    line-height: 1.2 !important;
}
[data-testid="stSidebar"] .stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 10px 28px rgba(26,79,168,0.55) !important;
}

/* Doc pills */
.doc-pill {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 12px; padding: 9px 14px;
    font-size: 12px; color: #9ab4e0 !important;
    margin: 5px 0; display: flex;
    align-items: center; gap: 8px;
    word-break: break-all; line-height: 1.4;
}

/* Chips */
.chip {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 10px; padding: 8px 12px;
    font-size: 12px; color: #9ab4e0 !important;
    line-height: 1.5; margin: 4px 0; display: block;
}

/* Download buttons */
[data-testid="stSidebar"] [data-testid="stDownloadButton"] > button {
    width: 100% !important;
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid rgba(255,255,255,0.10) !important;
    color: #9ab4e0 !important; border-radius: 12px !important;
    font-size: 12px !important; font-weight: 600 !important;
    margin-top: 6px !important; padding: 10px 14px !important;
}
[data-testid="stSidebar"] [data-testid="stDownloadButton"] > button:hover {
    background: rgba(255,255,255,0.10) !important;
}

/* Divider */
[data-testid="stSidebar"] hr { border-color: rgba(255,255,255,0.06) !important; margin: 18px 0 !important; }

[data-testid="stSidebar"] input {
    background: rgba(255,255,255,0.08) !important;
    border: 1px solid rgba(255,255,255,0.15) !important;
    border-radius: 10px !important;
    color: #ffffff !important;
    -webkit-text-fill-color: #ffffff !important;
}
[data-testid="stSidebar"] input::placeholder {
    color: rgba(255,255,255,0.35) !important;
    -webkit-text-fill-color: rgba(255,255,255,0.35) !important;
}
[data-testid="stSidebar"] .stExpander {
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 12px !important;
}

/* ── Top Navbar ── */
.top-navbar {
    position: sticky; top: 0; z-index: 999;
    background: rgba(255,255,255,0.95);
    backdrop-filter: blur(12px);
    border-bottom: 1px solid rgba(74,144,217,0.15);
    box-shadow: 0 2px 20px rgba(13,43,110,0.08);
    padding: 0 32px;
    display: flex; align-items: center; justify-content: space-between;
    height: 64px;
}
.navbar-brand {
    display: flex; align-items: center; gap: 10px;
}
.navbar-brand .brand-icon { font-size: 26px; }
.navbar-brand .brand-name {
    font-family: 'Playfair Display', serif; font-size: 20px;
    font-weight: 700; color: #0d2b6e;
}
.navbar-tabs {
    display: flex; align-items: center; gap: 4px;
}
.navbar-tab {
    padding: 8px 20px; border-radius: 12px; font-size: 14px;
    font-weight: 700; cursor: pointer; transition: all 0.2s ease;
    font-family: 'Nunito', sans-serif; border: none; background: transparent;
    color: #5a7abf; letter-spacing: 0.2px;
}
.navbar-tab.active {
    background: linear-gradient(135deg, #2451b3, #4a90d9);
    color: white !important; box-shadow: 0 4px 12px rgba(36,81,179,0.3);
}
.navbar-tab:hover:not(.active) { background: rgba(74,144,217,0.1); color: #2451b3; }

.navbar-user {
    display: flex; align-items: center; gap: 8px;
    background: rgba(74,144,217,0.08); border-radius: 20px;
    padding: 6px 14px; font-size: 13px; color: #2451b3; font-weight: 600;
}

/* ── Page wrapper ── */
.page-content {
    padding: 24px 32px 40px;
    max-width: 1100px;
    margin: 0 auto;
}

/* ── Agent landing hero ── */
.agents-hero {
    text-align: center; padding: 36px 0 28px;
}
.agents-hero h1 {
    font-family: 'Playfair Display', serif; font-size: 34px;
    font-weight: 700; color: #0d2b6e; margin: 0 0 8px;
}
.agents-hero p { font-size: 15px; color: #5a7abf; margin: 0; font-weight: 500; }

/* ── Agent grid cards ── */
.agent-grid-card {
    background: white; border-radius: 20px;
    padding: 28px 20px 20px; text-align: center;
    box-shadow: 0 4px 24px rgba(13,43,110,0.08);
    border: 2px solid rgba(74,144,217,0.10);
    transition: all 0.25s ease; position: relative;
    min-height: 190px; cursor: pointer;
}
.agent-grid-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 12px 32px rgba(13,43,110,0.14);
    border-color: rgba(74,144,217,0.35);
}
.agent-grid-card.done { border-color: rgba(46,170,94,0.4); }
.agent-card-status {
    position: absolute; top: 12px; right: 14px;
    font-size: 10px; font-weight: 700; padding: 3px 8px;
    border-radius: 20px; letter-spacing: 0.5px;
}
.agent-card-icon { font-size: 40px; margin-bottom: 12px; display: block; }
.agent-card-name {
    font-weight: 800; color: #0d2b6e; font-size: 14px;
    margin-bottom: 6px; font-family: 'Nunito', sans-serif;
}
.agent-card-desc { font-size: 12px; color: #5a7abf; line-height: 1.5; }

/* ── Agent result panel ── */
.agent-result-panel {
    background: white; border-radius: 20px; padding: 28px 32px;
    margin: 8px 0 20px; box-shadow: 0 4px 24px rgba(13,43,110,0.10);
    border: 1px solid rgba(74,144,217,0.15);
}
.agent-result-header {
    font-family: 'Playfair Display', serif; font-size: 24px;
    color: #0d2b6e; font-weight: 700; margin-bottom: 4px;
}
.agent-result-sub { font-size: 13px; color: #5a7abf; margin-bottom: 20px; }

/* ── Chat page ── */
.chat-page-wrapper {
    display: flex; flex-direction: column; height: calc(100vh - 64px);
}
.chat-messages-area {
    flex: 1; overflow-y: auto; padding: 24px 0 16px;
}
.chat-input-area {
    padding: 12px 0 24px; border-top: 1px solid rgba(74,144,217,0.1);
    background: #f0f4ff;
}

/* Chat input override */
[data-testid="stChatInput"] {
    background: white !important; border-radius: 20px !important;
    border: 2px solid rgba(74,144,217,0.25) !important;
    box-shadow: 0 4px 24px rgba(13,43,110,0.10) !important;
    padding: 4px 8px !important; transition: all 0.25s ease !important;
}
[data-testid="stChatInput"]:focus-within {
    border-color: #4a90d9 !important;
    box-shadow: 0 4px 24px rgba(74,144,217,0.25) !important;
}
[data-testid="stChatInput"] textarea {
    font-family: 'Nunito', sans-serif !important;
    font-size: 15px !important; color: #1a2b5e !important;
}
[data-testid="stChatInput"] textarea::placeholder { color: #8aaee0 !important; }
[data-testid="stChatInput"] button {
    background: linear-gradient(135deg, #4a90d9, #2563c7) !important;
    border-radius: 12px !important; border: none !important;
    box-shadow: 0 2px 8px rgba(37,99,199,0.3) !important;
}

/* ── Documents page ── */
.docs-section-title {
    font-family: 'Playfair Display', serif; font-size: 26px;
    color: #0d2b6e; font-weight: 700; margin-bottom: 6px;
}
.docs-section-sub { font-size: 14px; color: #5a7abf; margin-bottom: 24px; }
.doc-card {
    background: white; border-radius: 16px; padding: 16px 20px;
    box-shadow: 0 2px 12px rgba(13,43,110,0.07);
    border: 1px solid rgba(74,144,217,0.12);
    display: flex; align-items: center; gap: 14px; margin-bottom: 10px;
}
.doc-card-icon { font-size: 28px; flex-shrink: 0; }
.doc-card-name { font-weight: 700; color: #0d2b6e; font-size: 14px; }
.doc-card-status { font-size: 12px; color: #2eaa5e; margin-top: 2px; }

/* ── Processing overlay ── */
@keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
@keyframes fadeIn { from { opacity: 0; transform: translateY(20px); } to { opacity: 1; transform: translateY(0); } }
@keyframes bar { 0% { width: 0%; } 100% { width: 100%; } }
@keyframes pulse { 0%, 100% { opacity: 1; transform: scale(1); } 50% { opacity: 0.5; transform: scale(0.8); } }

.processing-overlay {
    position: fixed; top: 0; left: 0; width: 100vw; height: 100vh;
    background: rgba(240, 244, 255, 0.97); backdrop-filter: blur(8px);
    z-index: 9999; display: flex; flex-direction: column;
    align-items: center; justify-content: center;
    animation: fadeIn 0.4s ease;
}
.processing-card {
    background: white; border-radius: 28px; padding: 48px 56px;
    text-align: center; box-shadow: 0 24px 64px rgba(13,43,110,0.15);
    border: 1px solid rgba(74,144,217,0.2); max-width: 460px; width: 90%;
}
.spinner-ring {
    width: 72px; height: 72px; border: 5px solid #e8eeff;
    border-top: 5px solid #2451b3; border-radius: 50%;
    animation: spin 1s linear infinite; margin: 0 auto 24px;
}
.processing-title {
    font-family: 'Playfair Display', serif; font-size: 26px;
    font-weight: 700; color: #0d2b6e; margin-bottom: 8px;
}
.processing-sub { font-size: 14px; color: #5a7abf; margin-bottom: 28px; line-height: 1.6; }
.progress-bar-wrap {
    background: #e8eeff; border-radius: 20px; height: 6px;
    overflow: hidden; margin-bottom: 20px;
}
.progress-bar-fill {
    height: 6px; background: linear-gradient(90deg, #4a90d9, #2451b3);
    border-radius: 20px; animation: bar 3s ease-in-out infinite alternate;
}
.processing-steps { display: flex; flex-direction: column; gap: 10px; text-align: left; }
.step-item {
    display: flex; align-items: center; gap: 10px;
    font-size: 13px; color: #5a7abf; font-family: 'Nunito', sans-serif;
}
.step-dot {
    width: 8px; height: 8px; background: #4a90d9; border-radius: 50%;
    flex-shrink: 0; animation: pulse 1.5s infinite;
}

/* ── Alerts ── */
[data-testid="stAlert"] {
    border-radius: 14px !important; border: none !important;
    font-family: 'Nunito', sans-serif !important; font-size: 14px !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(74,144,217,0.3); border-radius: 10px; }

/* Main area top padding reset */
.main > div:first-child { padding-top: 0 !important; }

/* Run all agents button in main area */
.run-all-btn-wrap { display: flex; justify-content: center; margin-bottom: 24px; }
</style>
""", unsafe_allow_html=True)


# ── Session state defaults ────────────────────────────────────
defaults = {
    "run_all_triggered": False,
    "agent_statuses": {},
    "all_agents_results": {},
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
    "active_agent": None,
    "agent_result": None,
    "agent_running": False,
    "active_tab": "agents",
    "recipient_email": "",
    "docs_just_processed": False,
    "user_language": "English",  # ← new
    "voice_input_field": "",

}

for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

if "orchestrator" not in st.session_state:
    st.session_state.orchestrator = AgentOrchestrator()


# ── Export helpers ────────────────────────────────────────────
def build_txt_export(chat_history, user_name, user_mood, user_conditions):
    lines = ["=" * 60, "       MediChat — Conversation Export", "=" * 60]
    lines.append(f"Date     : {datetime.now().strftime('%d %B %Y, %I:%M %p')}")
    if user_name: lines.append(f"Patient  : {user_name}")
    if user_mood: lines.append(f"Mood     : {user_mood}")
    if user_conditions: lines.append(f"Conditions: {', '.join(user_conditions)}")
    lines += ["=" * 60, ""]
    for msg in chat_history:
        role = "You" if msg["role"] == "user" else "MediChat"
        lines += [f"[{role}]", msg["content"], ""]
    lines += ["-" * 60, "For informational purposes only. Always consult your doctor."]
    return "\n".join(lines)


def build_pdf_export(chat_history, user_name, user_mood, user_conditions):
    try:
        from fpdf import FPDF
    except ImportError:
        return None
    class PDF(FPDF):
        def header(self):
            self.set_fill_color(13, 43, 110); self.rect(0, 0, 210, 22, 'F')
            self.set_font("Helvetica", "B", 14); self.set_text_color(255, 255, 255)
            self.set_y(6); self.cell(0, 10, "MediChat  --  Medical Conversation Export", align="C"); self.ln(18)
        def footer(self):
            self.set_y(-14); self.set_font("Helvetica", "I", 8); self.set_text_color(150, 150, 150)
            self.cell(0, 10, "For informational purposes only. Always consult your doctor.", align="C")
    try:
        pdf = PDF(); pdf.set_auto_page_break(auto=True, margin=18); pdf.add_page()
        pdf.set_fill_color(232, 240, 255); pdf.set_draw_color(180, 200, 240)
        pdf.set_font("Helvetica", "", 9); pdf.set_text_color(30, 50, 110)
        meta_lines = [f"Date: {datetime.now().strftime('%d %B %Y, %I:%M %p')}"]
        if user_name: meta_lines.append(f"Patient: {user_name}")
        if user_mood: meta_lines.append(f"Mood: {user_mood}")
        if user_conditions: meta_lines.append(f"Conditions: {', '.join(user_conditions)}")
        pdf.set_x(10); pdf.cell(190, 8, "   |   ".join(meta_lines), border=1, fill=True, ln=True); pdf.ln(5)
        for msg in chat_history:
            is_user = msg["role"] == "user"
            content = msg["content"].encode("latin-1", errors="replace").decode("latin-1")
            pdf.set_font("Helvetica", "B", 8); pdf.set_text_color(100, 120, 180)
            label = f"  {user_name or 'You'}  " if is_user else "  MediChat  "
            if is_user:
                pdf.set_x(10 + 95); pdf.cell(95, 5, label, align="R", ln=True)
                pdf.set_font("Helvetica", "", 10); pdf.set_fill_color(220, 248, 198); pdf.set_text_color(20, 60, 20)
                pdf.set_x(10 + 40); pdf.multi_cell(150, 6, content, fill=True, border=0)
            else:
                pdf.set_x(10); pdf.cell(95, 5, label, align="L", ln=True)
                pdf.set_font("Helvetica", "", 10); pdf.set_fill_color(240, 245, 255); pdf.set_text_color(13, 43, 110)
                pdf.set_x(10); pdf.multi_cell(150, 6, content, fill=True, border=0)
            pdf.ln(3)
        buf = io.BytesIO(); buf.write(pdf.output()); buf.seek(0); return buf
    except Exception as e:
        print(f"[pdf_export] Error: {e}"); return None


def _set_welcome_message():
    st.session_state.pop("ob_step", None)
    name = st.session_state.user_name or "there"
    mood = st.session_state.user_mood
    mood_responses = {
        "Anxious":   f"Hey {name} 💙 I know medical stuff can feel stressful — totally get it. I'm here to make it way less scary, promise. Go ahead and drop your reports in the sidebar whenever you're ready, and I'll walk you through everything calmly. What's on your mind?",
        "Sad":       f"Hey {name} 🤍 I'm really glad you came here. Medical reports can feel overwhelming, but you don't have to figure it all out alone. Upload your docs when you're ready and we'll go through them together, nice and easy. How are you holding up?",
        "Irritable": f"Hey {name}. I'll keep it short and useful — no fluff. Drop your docs in the sidebar and ask me whatever you need. What do you want to know?",
        "Tired":     f"Hey {name} 🌙 I'll keep things super simple for you. Just upload your reports on the left and I'll break everything down — no long explanations, just the stuff that matters. What would you like to start with?",
        "Happy":     f"Hey {name}! 😊 Love the energy — let's get into it! Drop your medical docs in the sidebar and I'll help you make sense of everything. What are you curious about?",
        "Relieved":  f"Hey {name} 🌿 Glad you're feeling a bit lighter! Upload your reports whenever you're ready and we'll go through them together. Anything specific you want to check on?",
        "Patient":   f"Hey {name} 🏥 Thanks for being here. Whenever you're ready, upload your documents on the left and I'll walk you through everything step by step. Where would you like to start?",
        "Strong":    f"Hey {name}! 💪 Alright, let's do this. Drop your reports in the sidebar and I'll give you a solid breakdown. What do you want to dive into first?",
        "Unwell":    f"Hey {name} 💛 Hope you're hanging in there. Take your time — upload your docs when you're ready and I'll explain everything gently. No rush at all. How are you feeling right now?",
        "Calm":      f"Hey {name} 😊 Good to meet you! Upload your medical documents on the left whenever you're ready and we'll go through them together. What would you like to know?",
        "Confused":  f"Hey {name}! 😊 Don't worry at all — that's literally why I'm here. I'll explain everything in plain, simple words. Just upload your reports on the left and ask me anything. What's confusing you the most?",
        "Neutral":   f"Hey {name}! 👋 I'm here to help you make sense of your medical reports. Just upload your docs in the sidebar and ask me anything — I'll keep it simple and clear. What can I help you with?",
    }
    st.session_state.chat_history = [{"role": "assistant", "content": mood_responses.get(mood, mood_responses["Neutral"])}]
    lang = st.session_state.get("user_language", "English")
    if lang != "English":
        from langchain_groq import ChatGroq
        from langchain_core.output_parsers import StrOutputParser
        _llm = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0.1)
        msg = st.session_state.chat_history[0]["content"]
        translated = (_llm | StrOutputParser()).invoke(
            f"Translate this exactly to {lang}, keep the same warm friendly tone, keep emojis: {msg}"
        )
        st.session_state.chat_history[0]["content"] = translated


def _set_docs_ready_message():
    """Called after documents are processed — the bot leads by offering to walk through them."""
    name = st.session_state.user_name or "there"
    mood = st.session_state.user_mood
    mood_responses = {
        "Anxious":   f"Alright {name}, I've gone through your reports 💙 Everything's loaded up and ready. Want me to walk you through what I found, or is there something specific you want to ask about first?",
        "Sad":       f"Okay {name}, I've read through everything 🤍 Your reports are all loaded up. I can give you a gentle walkthrough, or if you'd rather just ask about something specific — totally up to you. What feels right?",
        "Irritable": f"Done — your docs are processed. I can give you a quick summary or you can jump straight to questions. What do you need?",
        "Tired":     f"All set, {name} 🌙 Your reports are loaded. Want me to give you a quick rundown of the main things, or do you have a specific question? I'll keep it short either way.",
        "Happy":     f"Awesome, {name}! 🎉 I've gone through all your documents. Want me to walk you through what I found, or do you wanna jump straight to questions?",
        "Relieved":  f"All done, {name} 🌿 Your reports are loaded and ready. Want me to give you an overview, or is there something specific on your mind?",
        "Patient":   f"Alright {name}, everything's processed and ready 🏥 I can walk you through the full picture step by step, or you can ask about specific things. What works for you?",
        "Strong":    f"All loaded up, {name} 💪 Your reports are ready to go. Want the full breakdown or do you want to dive into specifics?",
        "Unwell":    f"Okay {name}, your docs are all set 💛 No rush — whenever you're ready, I can walk you through what's in there, or you can ask me anything. What would help most right now?",
        "Calm":      f"All set, {name} 😊 I've gone through your reports. I can give you an overview or we can go question by question — what do you prefer?",
        "Confused":  f"Okay {name}, good news — I've read through everything! 😊 Want me to explain what's in your reports in simple words? Or if you have a specific question, just ask — no question is too basic, I promise.",
        "Neutral":   f"All set, {name}! 👋 I've gone through your reports and everything's loaded up. Want me to give you an overview, or do you have something specific you want to ask about?",
    }
    st.session_state.chat_history = [{"role": "assistant", "content": mood_responses.get(mood, mood_responses["Neutral"])}]
    lang = st.session_state.get("user_language", "English")
    if lang != "English":
        from langchain_groq import ChatGroq
        from langchain_core.output_parsers import StrOutputParser
        _llm = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0.1)
        msg = st.session_state.chat_history[0]["content"]
        translated = (_llm | StrOutputParser()).invoke(
            f"Translate this exactly to {lang}, keep the same warm friendly tone, keep emojis: {msg}"
        )
        st.session_state.chat_history[0]["content"] = translated


# ── Onboarding dialog ─────────────────────────────────────────
@st.dialog("Welcome to MediChat", width="large")
def show_onboarding():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700&family=DM+Sans:wght@300;400;500&display=swap');
    div[data-testid="stDialog"] > div {
        border-radius: 28px !important; background: rgba(255,255,255,0.97) !important;
        backdrop-filter: blur(16px) !important; max-width: 500px !important;
        box-shadow: 0 24px 60px rgba(0,0,0,0.18) !important;
    }
    div[data-testid="stDialog"] p, div[data-testid="stDialog"] label,
    div[data-testid="stDialog"] span:not(.ob-dot),
    div[data-testid="stDialog"] div:not(.ob-dots):not(.ob-dot) {
        color: #1a1830 !important; -webkit-text-fill-color: #1a1830 !important;
    }
    div[data-testid="stDialog"] input { background: #f5f5f5 !important; color: #1a1830 !important; -webkit-text-fill-color: #1a1830 !important; }
    div[data-testid="stDialog"] button { color: #1a1830 !important; -webkit-text-fill-color: #1a1830 !important; }
    div[data-testid="stDialog"] button[kind="primary"] { background: #2451b3 !important; color: #ffffff !important; -webkit-text-fill-color: #ffffff !important; }
    div[data-testid="stDialog"] button[kind="secondary"] { background: #f0f0f0 !important; color: #1a1830 !important; }
    .ob-label { font-size: 11px; letter-spacing: 1.8px; text-transform: uppercase; color: #8aaee0 !important; -webkit-text-fill-color: #8aaee0 !important; font-weight: 500; text-align: center; margin-bottom: 6px; }
    .ob-title { font-family: 'Playfair Display', serif; font-size: 24px; color: #1a1830 !important; -webkit-text-fill-color: #1a1830 !important; text-align: center; margin-bottom: 5px; line-height: 1.3; }
    .ob-sub   { font-size: 13px; color: #9090a0 !important; -webkit-text-fill-color: #9090a0 !important; text-align: center; margin-bottom: 22px; font-weight: 300; line-height: 1.5; }
    .ob-dots  { display: flex; align-items: center; justify-content: center; gap: 6px; margin-bottom: 22px; }
    .ob-dot   { height: 6px; border-radius: 4px; display: inline-block; }
    .ob-dot.active { width: 22px; background: #2451b3; }
    .ob-dot.done   { width: 6px;  background: #a3b8e8; }
    .ob-dot.idle   { width: 6px;  background: #ddd; }
    .age-big  { font-family: 'Playfair Display', serif; font-size: 72px; font-weight: 700; color: #2451b3 !important; -webkit-text-fill-color: #2451b3 !important; text-align: center; line-height: 1; letter-spacing: -2px; }
    .age-unit { text-align: center; font-size: 14px; color: #9090a0 !important; -webkit-text-fill-color: #9090a0 !important; font-weight: 300; }
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
        st.markdown('<div class="ob-label">Step 1 of 4</div><div class="ob-title">Let\'s get acquainted</div><div class="ob-sub">Tell us a bit about yourself and who this report is for</div>', unsafe_allow_html=True)
        name = st.text_input("Your name (optional)", value=st.session_state.user_name, placeholder="e.g. Aryan")
        st.markdown('<p style="font-size:13px;font-weight:600;color:#6a7ab5;margin:16px 0 12px;">This report is for:</p>', unsafe_allow_html=True)
        whom_options = [("me","🧑","Myself"),("parent","👴","My Parent"),("child","👶","My Child"),("other","🤝","Friend")]
        cols = st.columns(4)
        for i, (key, icon, label) in enumerate(whom_options):
            with cols[i]:
                selected = st.session_state.user_whom == key
                if st.button(f"{icon}\n\n{label}", key=f"whom_{key}", use_container_width=True, type="primary" if selected else "secondary"):
                    st.session_state.user_name = name; st.session_state.user_whom = key; st.rerun()
        st.markdown("<br>", unsafe_allow_html=True)
        col_skip, col_next = st.columns([1, 1])
        with col_skip:
            if st.button("Skip for now", use_container_width=True, key="ob_skip1"):
                st.session_state.user_name = name; st.session_state.onboarding_done = True; _set_welcome_message(); st.rerun()
        with col_next:
            if st.button("Continue →", use_container_width=True, type="primary", key="ob_next1"):
                st.session_state.user_name = name; st.session_state.ob_step = 2; st.rerun()

    elif step == 2:
        age_titles = {"me":"How old are you?","parent":"How old is your parent?","child":"How old is your child?","other":"How old are they?"}
        title = age_titles.get(st.session_state.user_whom, "How old are you?")
        st.markdown(f'<div class="ob-label">Step 2 of 4</div><div class="ob-title">{title}</div><div class="ob-sub">Drag to set the age</div>', unsafe_allow_html=True)
        selected_age = st.slider("Age", min_value=1, max_value=100, value=st.session_state.user_age, key="slider_age_temp", label_visibility="collapsed")
        st.markdown(f'<div class="age-big">{selected_age}</div><div class="age-unit">years old</div><br>', unsafe_allow_html=True)
        col_back, col_next = st.columns([1, 1])
        with col_back:
            if st.button("← Back", use_container_width=True, key="ob_back2"):
                st.session_state.user_age = selected_age; st.session_state.ob_step = 1; st.rerun()
        with col_next:
            if st.button("Continue →", use_container_width=True, type="primary", key="ob_next2"):
                st.session_state.user_age = selected_age; st.session_state.ob_step = 3; st.rerun()

    elif step == 3:
        st.markdown('<div class="ob-label">Step 3 of 4</div><div class="ob-title">Any existing conditions?</div><div class="ob-sub">Select all that apply</div>', unsafe_allow_html=True)
        cond_options = [("diabetes","Diabetes","🩸"),("hypertension","Hypertension","💓"),("heart","Heart Disease","❤️"),("thyroid","Thyroid","🦋"),("asthma","Asthma","🫁"),("neurological","Neurological","🧠"),("none","None","✅"),("other","Other","➕")]
        current = set(st.session_state.user_conditions)
        cols = st.columns(2)
        for i, (key, label, icon) in enumerate(cond_options):
            with cols[i % 2]:
                selected = key in current
                if st.button(f"{icon} {label}", key=f"cond_{key}", use_container_width=True, type="primary" if selected else "secondary"):
                    if key == "none": st.session_state.user_conditions = ["none"]
                    else:
                        new = set(st.session_state.user_conditions) - {"none"}
                        if key in new: new.remove(key)
                        else: new.add(key)
                        st.session_state.user_conditions = list(new)
                    st.rerun()
        col_back, col_skip, col_next = st.columns([1, 1, 1])
        with col_back:
            if st.button("← Back", use_container_width=True, key="ob_back3"): st.session_state.ob_step = 2; st.rerun()
        with col_skip:
            if st.button("Skip", use_container_width=True, key="ob_skip3"): st.session_state.user_conditions = []; st.session_state.ob_step = 4; st.rerun()
        with col_next:
            if st.button("Continue →", use_container_width=True, type="primary", key="ob_next3"): st.session_state.ob_step = 4; st.rerun()

    elif step == 4:
        st.markdown('<div class="ob-label">Step 4 of 4</div><div class="ob-title">How are you feeling?</div><div class="ob-sub">We\'ll match our tone to how you feel right now</div>', unsafe_allow_html=True)

        # Language picker
        st.markdown('<p style="font-size:13px;font-weight:600;color:#6a7ab5;margin:16px 0 8px;">🌐 Preferred Language</p>', unsafe_allow_html=True)
        languages = ["English", "Hindi", "Malayalam"]
        lang_cols1 = st.columns(3)
        for i, lang in enumerate(languages):
            with lang_cols1[i]:
                selected = st.session_state.user_language == lang
                if st.button(lang, key=f"lang_{lang}", use_container_width=True,
                             type="primary" if selected else "secondary"):
                    st.session_state.user_language = lang
                    st.rerun()
        st.markdown("<br>", unsafe_allow_html=True)
        all_moods = [("Happy","😊"),("Relieved","😌"),("Patient","🏥"),("Neutral","😐"),("Anxious","😟"),("Sad","😔"),("Tired","😴"),("Calm","🧘"),("Strong","💪"),("Unwell","🤒"),("Confused","😕"),("Irritable","😤")]
        current_mood = st.session_state.user_mood
        cols_per_row = 4
        rows = [all_moods[i:i+cols_per_row] for i in range(0, len(all_moods), cols_per_row)]
        for row in rows:
            cols = st.columns(len(row))
            for j, (mood_name, emoji) in enumerate(row):
                with cols[j]:
                    if st.button(f"{emoji}\n\n{mood_name}", key=f"mood_{mood_name}", use_container_width=True, type="primary" if mood_name == current_mood else "secondary"):
                        st.session_state.user_mood = mood_name; st.rerun()
        col_back, col_done = st.columns([1, 1])
        with col_back:
            if st.button("← Back", use_container_width=True, key="ob_back4"): st.session_state.ob_step = 3; st.rerun()
        with col_done:
            if st.button("Get started", use_container_width=True, type="primary", key="ob_done"):
                st.session_state.onboarding_done = True; _set_welcome_message(); st.rerun()


# ── ONBOARDING GATE (flicker-free) ───────────────────────────
if not st.session_state.get("onboarding_done", False):
    show_onboarding()
    st.stop()


# ── SIDEBAR ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding:36px 20px 28px; margin:-16px -16px 24px;
        background:linear-gradient(180deg,rgba(0,0,0,0.3),rgba(0,0,0,0.1));
        border-bottom:1px solid rgba(255,255,255,0.06);">
        <div style="width:64px;height:64px;background:linear-gradient(135deg,#1a4fa8,#3a7bd5);
            border-radius:20px;display:flex;align-items:center;justify-content:center;
            font-size:32px;margin:0 auto 14px;box-shadow:0 8px 24px rgba(26,79,168,0.5);">🏥</div>
        <div style="font-family:'Playfair Display',serif;font-size:22px;
            font-weight:700;color:#ffffff;letter-spacing:0.5px;margin-bottom:4px;">MediChat</div>
        <div style="font-size:9px;color:#4a6aaa;letter-spacing:3px;text-transform:uppercase;">
            AI Medical Assistant</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <script>
    function cleanUploader() {
        const uploader = window.parent.document.querySelector('[data-testid="stFileUploader"]');
        if (uploader) {
            const small = uploader.querySelector('small');
            if (small) small.style.display = 'none';
            const svgs = uploader.querySelectorAll('svg');
            svgs.forEach(s => s.style.display = 'none');
            const imgs = uploader.querySelectorAll('img');
            imgs.forEach(i => i.style.display = 'none');
        }
    }
    setInterval(cleanUploader, 300);
    </script>
    """, unsafe_allow_html=True)

    # User profile chip
    if st.session_state.user_name:
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

    # ── Always-visible document upload ───────────────────────
    st.markdown('<p style="font-size:9px;font-weight:800;letter-spacing:2.5px;text-transform:uppercase;color:#4a6aaa;margin:0 0 8px;display:block;">📂 Upload Documents</p>', unsafe_allow_html=True)
    uploaded_files = st.file_uploader(
        "Drop your files here",
        type=["pdf", "jpg", "jpeg", "png", "webp", "docx", "txt"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if uploaded_files:
        for f in uploaded_files:
            already_done = f.name in st.session_state.uploaded_names
            status_icon = "✅" if already_done else "📄"
            status_text = "Processed & ready" if already_done else "Ready to process"
            status_color = "#1a8a4a" if already_done else "#2451b3"
            st.markdown(f'''
            <div style="background:#ffffff;border:1px solid {'#2eaa5e' if already_done else '#2451b3'};
                border-radius:12px;padding:10px 14px;margin:6px 0;
                display:flex;align-items:center;gap:10px;">
                <span style="font-size:18px;">{status_icon}</span>
                <div>
                    <div style="font-size:13px;font-weight:700;color:#0d2b6e;-webkit-text-fill-color:#0d2b6e;">{html.escape(f.name)}</div>
                    <div style="font-size:11px;font-weight:600;color:{status_color};-webkit-text-fill-color:{status_color};">{status_text}</div>
                </div>
            </div>''', unsafe_allow_html=True)

    if uploaded_files:
        col_l, col_btn, col_r = st.columns([0.5, 4, 0.5])
        with col_btn:
            if st.button("✨ Process All Documents", use_container_width=True):
                st.session_state.processing = True
                st.session_state.files_to_process = [{"name": f.name, "data": f.read()} for f in uploaded_files]
                st.rerun()



    # ── Chat-specific: export & suggestions ──────────────────
    if st.session_state.active_tab == "chat":
        if st.session_state.chat_history and len(st.session_state.chat_history) > 1:
            st.markdown("---")
            st.markdown('<p class="sidebar-section-label">Export Chat</p>', unsafe_allow_html=True)
            fname_base = f"medichat_{datetime.now().strftime('%Y%m%d_%H%M')}"
            txt_content = build_txt_export(st.session_state.chat_history, st.session_state.user_name, st.session_state.user_mood, st.session_state.user_conditions)
            st.download_button(label="📄  Download as Text", data=txt_content.encode("utf-8"), file_name=f"{fname_base}.txt", mime="text/plain", use_container_width=True)
            pdf_buf = build_pdf_export(st.session_state.chat_history, st.session_state.user_name, st.session_state.user_mood, st.session_state.user_conditions)
            if pdf_buf:
                st.download_button(label="📑  Download as PDF", data=pdf_buf, file_name=f"{fname_base}.pdf", mime="application/pdf", use_container_width=True)

        st.markdown("---")
        st.markdown('<p class="sidebar-section-label">Try asking</p>', unsafe_allow_html=True)
        for q in ["What does my creatinine level mean?","What medications were prescribed?","Is my blood sugar normal?","What was the diagnosis?","Which values are abnormal?","What follow-up is needed?"]:
            if st.button(f"💬 {q}", key=f"chip_{q}", use_container_width=True):
                st.session_state.active_tab = "chat"
                history_before = list(st.session_state.chat_history)
                st.session_state.chat_history.append({"role": "user", "content": q})
                from qa_chain import get_answer
                if st.session_state.llm:
                    answer = get_answer(
                        llm=st.session_state.llm,
                        retriever=st.session_state.retriever,
                        question=q,
                        chat_history=history_before,
                        mood=st.session_state.user_mood,
                        user_name=st.session_state.user_name,
                        user_conditions=st.session_state.user_conditions,
                        user_whom=st.session_state.user_whom,
                        user_age=st.session_state.user_age,
                        summaries=st.session_state.get("summaries", {}),
                        user_language=st.session_state.get("user_language", "English"),
                    )
                    st.session_state.chat_history.append({"role": "assistant", "content": answer})
                st.rerun()

    st.markdown("---")
    st.markdown('<p class="sidebar-section-label">🌐 Language</p>', unsafe_allow_html=True)
    st.markdown("""
    <style>
    [data-testid="stSidebar"] [data-testid="stSelectbox"] > div > div {
        background: rgba(255,255,255,0.12) !important;
        border: 1px solid rgba(255,255,255,0.2) !important;
        border-radius: 10px !important;
        color: #ffffff !important;
    }
    [data-testid="stSidebar"] [data-testid="stSelectbox"] > div > div > div {
        color: #ffffff !important;
        -webkit-text-fill-color: #ffffff !important;
    }
    [data-testid="stSidebar"] [data-testid="stSelectbox"] svg {
        display: block !important;
        fill: #ffffff !important;
    }
    </style>
    """, unsafe_allow_html=True)

    languages = ["English", "Hindi", "Malayalam"]
    current_lang = st.session_state.get("user_language", "English")
    current_index = languages.index(current_lang) if current_lang in languages else 0

    selected_lang = st.selectbox(
        "Language",
        options=languages,
        index=current_index,
        label_visibility="collapsed",
        key="sidebar_lang",
    )
    if selected_lang != st.session_state.get("user_language", "English"):
        st.session_state.user_language = selected_lang
        from langchain_groq import ChatGroq
        from langchain_core.output_parsers import StrOutputParser
        _llm = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0.1)
        ack = (_llm | StrOutputParser()).invoke(
            f"In {selected_lang} only, write one short warm sentence (max 12 words) saying you'll now respond in {selected_lang}. Keep emojis."
        )
        st.session_state.chat_history.append({"role": "assistant", "content": ack})
        st.rerun()
    else:
        st.session_state.user_language = selected_lang

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<p style='font-size:11px; color:#4a6a9a; text-align:center; line-height:1.6;'>For informational purposes only.<br/>Always consult your doctor.</p>", unsafe_allow_html=True)

# ── DOCUMENT PROCESSING (always runs before rendering) ───────
if st.session_state.get("processing_error"):
    st.error(f"❌ {st.session_state.processing_error}")
    if st.button("Clear error"):
        st.session_state.processing_error = None
        st.rerun()

if st.session_state.get("processing"):
    st.markdown("""
    <div class="processing-overlay">
        <div class="processing-card">
            <div class="spinner-ring"></div>
            <div class="processing-title">Analysing your documents</div>
            <div class="processing-sub">Please wait while we read, extract,<br>and index all your medical content.</div>
            <div class="progress-bar-wrap"><div class="progress-bar-fill"></div></div>
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
            tmp.write(file_info["data"]); tmp_paths.append(tmp.name); original_names.append(file_info["name"])
    try:
        if st.session_state.chroma_dir and os.path.exists(st.session_state.chroma_dir):
            shutil.rmtree(st.session_state.chroma_dir, ignore_errors=True)
        st.session_state.chroma_dir = None
        vectorstore, chroma_dir, summaries = ingest_multiple_files(tmp_paths, original_names, session_id=st.session_state.session_id)
        llm, retriever = build_qa_chain(vectorstore, mood=st.session_state.user_mood)
        st.session_state.llm = llm
        st.session_state.retriever = retriever
        st.session_state.chroma_dir = chroma_dir
        st.session_state.uploaded_names = original_names
        st.session_state.summaries = summaries
        _set_docs_ready_message()
    except Exception as e:
        st.session_state.processing_error = str(e)
    finally:
        for path in tmp_paths:
            if os.path.exists(path): os.unlink(path)
    st.session_state.processing = False
    st.session_state.files_to_process = []
    st.rerun()


# ── TOP NAVBAR ────────────────────────────────────────────────
mood_emoji_map = {"Happy":"😊","Relieved":"😌","Patient":"🏥","Sad":"😔","Neutral":"😐","Anxious":"😟","Irritable":"😤","Tired":"😴","Strong":"💪","Unwell":"🤒","Calm":"🧘","Confused":"😕"}
user_display = f"{mood_emoji_map.get(st.session_state.user_mood,'😐')} {st.session_state.user_name}" if st.session_state.user_name else "👤 Guest"

active_tab = st.session_state.active_tab

# Render navbar HTML (display only)
agents_active   = "active" if active_tab == "agents"    else ""
chat_active     = "active" if active_tab == "chat"       else ""
docs_active     = "active" if active_tab == "documents"  else ""

# ── TOP NAVBAR (pure Streamlit columns, no CSS hacks) ─────────
st.markdown(f"""
<div style="background:rgba(255,255,255,0.95);backdrop-filter:blur(12px);
    border-bottom:1px solid rgba(74,144,217,0.15);
    box-shadow:0 2px 20px rgba(13,43,110,0.08);
    padding:0 32px; height:64px;
    display:flex;align-items:center;justify-content:space-between;
    margin-bottom:8px;">
    <div style="display:flex;align-items:center;gap:10px;">
        <span style="font-size:26px;">🏥</span>
        <span style="font-family:'Playfair Display',serif;font-size:20px;font-weight:700;color:#0d2b6e;">MediChat</span>
    </div>
    <div style="font-size:13px;color:#2451b3;font-weight:600;
        background:rgba(74,144,217,0.08);border-radius:20px;padding:6px 14px;">
        {user_display}
    </div>
</div>
""", unsafe_allow_html=True)

nav_c1, nav_c2, nav_c3, nav_c4, nav_c5 = st.columns([2, 1, 1, 1, 2])
with nav_c2:
    if st.button("🤖 Agents", key="nav_agents", use_container_width=True,
                 type="primary" if active_tab == "agents" else "secondary"):
        st.session_state.active_tab = "agents"
        st.session_state.active_agent = None
        st.session_state.agent_result = None
        st.rerun()
with nav_c3:
    if st.button("💬 Chat", key="nav_chat", use_container_width=True,
                 type="primary" if active_tab == "chat" else "secondary"):
        st.session_state.active_tab = "chat"
        st.rerun()
with nav_c4:
    if st.button("📁 Docs", key="nav_docs", use_container_width=True,
                 type="primary" if active_tab == "documents" else "secondary"):
        st.session_state.active_tab = "documents"
        st.rerun()




# ── Main content ──────────────────────────────────────────────
_, main_col, _ = st.columns([1, 8, 1])

with main_col:

    # ══════════════════════════════════════════════════════════
    #  AGENTS TAB
    # ══════════════════════════════════════════════════════════
    if active_tab == "agents":

        # Hero header — always visible
        st.markdown("""
        <div class="agents-hero">
            <h1>🤖 AI Medical Agents</h1>
            <p>Powerful tools that analyse your documents and extract insights automatically</p>
        </div>
        """, unsafe_allow_html=True)

        orchestrator: AgentOrchestrator = st.session_state.orchestrator

        agent_icons = {
            "📋 Summarizer":           ("📋", "#4a90d9"),
            "🔬 Lab Analyzer":         ("🔬", "#7c5cbf"),
            "💊 Medication Agent":     ("💊", "#2eaa5e"),
            "📅 Appointment Reminder": ("📅", "#d97b4a"),
            "📧 Email Report":         ("📧", "#c72563"),
            "🩻 Image Explainer":      ("🩻", "#0a9396"),  # ← add this
        }
        status_style = {
            "done":    ("✅ Done",      "#2eaa5e", "#edfdf4"),
            "running": ("⏳ Running…",  "#d97b4a", "#fff8f0"),
            "pending": ("🕐 Pending…",  "#4a90d9", "#f0f4ff"),
            "waiting": ("⏸ Waiting…",  "#c72563", "#fff0f5"),
            "":        ("— Not run",   "#aab8d4", "#f5f7ff"),
        }

        # ── No documents yet ─────────────────────────────────
        if st.session_state.retriever is None:
            st.markdown("""
            <div style="text-align:center; margin:16px auto 32px; padding:40px 32px;
                background:white; border-radius:24px; max-width:520px;
                box-shadow:0 4px 24px rgba(13,43,110,0.08);
                border:2px dashed rgba(74,144,217,0.25);">
                <div style="font-size:52px; margin-bottom:16px;">📂</div>
                <p style="color:#0d2b6e; font-size:17px; font-weight:700; margin:0 0 8px;">
                    No documents loaded yet
                </p>
                <p style="color:#5a7abf; font-size:14px; margin:0; line-height:1.6;">
                    Upload your medical documents from the sidebar,<br>
                    then come back here to run the agents.
                </p>
            </div>
            """, unsafe_allow_html=True)

        # ── Viewing a single agent result ────────────────────
        elif st.session_state.active_agent is not None:

            # Run agent if needed
            if st.session_state.agent_running and st.session_state.agent_result is None:
                with st.spinner(f"Running {st.session_state.active_agent}…"):
                    try:
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
                        st.session_state.agent_statuses[st.session_state.active_agent] = "done"
                        st.session_state.all_agents_results[st.session_state.active_agent] = result
                    except Exception as e:
                        st.session_state.agent_result = f"❌ Agent error: {e}"
                    finally:
                        st.session_state.agent_running = False
                st.rerun()

            if st.session_state.agent_result:
                agent_name = st.session_state.active_agent
                descriptions = orchestrator.agent_descriptions

                # ── Email agent gets Gmail-style UI ──────────
                if agent_name == "📧 Email Report":
                    st.markdown(f"""
                    <div class="agent-result-panel">
                        <div class="agent-result-header">{agent_name}</div>
                        <div class="agent-result-sub">{descriptions.get(agent_name, '')}</div>
                    </div>
                    """, unsafe_allow_html=True)

                    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

                    # Gmail-style compose window
                    st.markdown("""
                    <div style="background:white;border-radius:16px;overflow:hidden;
                        box-shadow:0 4px 24px rgba(13,43,110,0.12);
                        border:0.5px solid rgba(74,144,217,0.2);">
                        <div style="background:#1a3d8f;padding:10px 18px;
                            display:flex;align-items:center;justify-content:space-between;">
                            <span style="font-size:13px;font-weight:600;color:#fff;">New message</span>
                            <div style="display:flex;gap:14px;">
                                <span style="color:rgba(255,255,255,0.7);font-size:16px;cursor:pointer;">—</span>
                                <span style="color:rgba(255,255,255,0.7);font-size:16px;cursor:pointer;">⛶</span>
                                <span style="color:rgba(255,255,255,0.7);font-size:16px;cursor:pointer;">✕</span>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    # To field
                    st.markdown("""
                    <div style="background:white;border-left:0.5px solid rgba(74,144,217,0.2);
                        border-right:0.5px solid rgba(74,144,217,0.2);
                        border-bottom:0.5px solid rgba(74,144,217,0.1);
                        padding:2px 18px 0;">
                        <p style="font-size:11px;color:#8aaee0;margin:6px 0 0;font-weight:600;
                            letter-spacing:0.5px;text-transform:uppercase;">To</p>
                    </div>
                    """, unsafe_allow_html=True)
                    recipient_email = st.text_input(
                        "to_field",
                        value=st.session_state.get("recipient_email", ""),
                        placeholder="doctor@hospital.com",
                        label_visibility="collapsed",
                        key="email_recipient_main",
                    )
                    st.session_state.recipient_email = recipient_email

                    # Subject field
                    st.markdown("""
                    <div style="background:white;border-left:0.5px solid rgba(74,144,217,0.2);
                        border-right:0.5px solid rgba(74,144,217,0.2);
                        border-bottom:0.5px solid rgba(74,144,217,0.1);
                        padding:2px 18px 0;">
                        <p style="font-size:11px;color:#8aaee0;margin:6px 0 0;font-weight:600;
                            letter-spacing:0.5px;text-transform:uppercase;">Subject</p>
                    </div>
                    """, unsafe_allow_html=True)
                    email_subject = st.text_input(
                        "subject_field",
                        value="Medical Report Summary — MediChat",
                        label_visibility="collapsed",
                        key="email_subject_main",
                    )

                    # Body — parse draft from agent result
                    draft_body = st.session_state.agent_result
                    lines = draft_body.split("\n")
                    body_lines = []
                    skip_prefixes = (
                        "to:", "subject:", "📧 draft", "---", "_to send",
                        "draft email ready", "no recipient", "family member",
                    )
                    found_body = False
                    for line in lines:
                        clean = line.replace("**", "").replace("*", "").strip()
                        if not clean:
                            continue
                        if clean.lower().startswith(skip_prefixes):
                            continue
                        # Start body only when we hit the salutation
                        if clean.lower().startswith("dear"):
                            found_body = True
                        if found_body:
                            body_lines.append(line)
                    draft_body = "\n".join(body_lines).strip()

                    st.markdown("""
                    <div style="background:white;border-left:0.5px solid rgba(74,144,217,0.2);
                        border-right:0.5px solid rgba(74,144,217,0.2);
                        padding:2px 18px 0;">
                        <p style="font-size:11px;color:#8aaee0;margin:6px 0 0;font-weight:600;
                            letter-spacing:0.5px;text-transform:uppercase;">Message</p>
                    </div>
                    """, unsafe_allow_html=True)
                    email_body = st.text_area(
                        "body_field",
                        value=draft_body,
                        height=240,
                        label_visibility="collapsed",
                        key="email_body_main",
                    )

                    # Bottom bar with Send button
                    st.markdown("""
                    <div style="background:white;border:0.5px solid rgba(74,144,217,0.2);
                        border-top:none;border-radius:0 0 16px 16px;
                        padding:10px 18px;display:flex;align-items:center;gap:10px;">
                    </div>
                    """, unsafe_allow_html=True)

                    send_col, dl_col, back_col = st.columns([2, 1, 1])
                    with send_col:
                        if st.button("📤  Send Email", use_container_width=True, type="primary", key="send_email_btn"):
                            if not recipient_email:
                                st.error("Please enter a recipient email address.")
                            else:
                                with st.spinner("Sending…"):
                                    try:
                                        import resend
                                        resend.api_key = os.getenv("RESEND_API_KEY")
                                        footer = "\n\n---\nSent via MediChat. For informational purposes only. Always consult your doctor."
                                        resend.Emails.send({
                                            "from": "MediChat <onboarding@resend.dev>",
                                            "to": recipient_email,
                                            "subject": email_subject,
                                            "text": email_body + footer,
                                        })
                                        st.success(f"✅ Email sent to {recipient_email}!")
                                    except Exception as e:
                                        st.error(f"❌ Failed to send: {e}")
                    with dl_col:
                        st.download_button(
                            label="📄 Download",
                            data=email_body.encode("utf-8"),
                            file_name=f"medichat_email_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
                            mime="text/plain",
                            use_container_width=True,
                        )
                    with back_col:
                        if st.button("← Back", use_container_width=True, key="email_back_btn"):
                            st.session_state.active_agent = None
                            st.session_state.agent_result = None
                            st.rerun()

                # ── All other agents ─────────────────────────
                else:
                    st.markdown(f"""
                    <div class="agent-result-panel">
                        <div class="agent-result-header">{agent_name}</div>
                        <div class="agent-result-sub">{descriptions.get(agent_name, '')}</div>
                    </div>
                    """, unsafe_allow_html=True)

                    if agent_name == "📅 Appointment Reminder":
                        _render_appointment(st.session_state.agent_result)
                    else:
                        st.markdown(st.session_state.agent_result)
                    st.markdown("<br>", unsafe_allow_html=True)

                    c1, c2, c3 = st.columns([1, 1, 1])
                    with c1:
                        st.download_button(
                            label="📄 Download Result",
                            data=st.session_state.agent_result.encode("utf-8"),
                            file_name=f"medichat_{agent_name.replace(' ','_').lower()}_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
                            mime="text/plain", use_container_width=True,
                        )
                    with c2:
                        if st.button("🔄 Run Again", use_container_width=True):
                            st.session_state.agent_result = None
                            st.session_state.agent_running = True
                            st.rerun()
                    with c3:
                        if st.button("← Back to Agents", use_container_width=True):
                            st.session_state.active_agent = None
                            st.session_state.agent_result = None
                            st.rerun()

        # ── Agent grid — landing (documents loaded) ──────────
        else:
            # ── Run All Agents ────────────────────────────────
            col_l, col_btn, col_r = st.columns([2, 2, 2])
            with col_btn:
                if st.button("⚡ Run All Agents", use_container_width=True, key="run_all"):
                    st.session_state.run_all_triggered = True
                    st.session_state.all_agents_results = {}
                    for aname in orchestrator.agent_names:
                        st.session_state.agent_statuses[aname] = "pending"
                    st.rerun()

            if st.session_state.get("run_all_triggered"):
                any_pending = any(
                    st.session_state.agent_statuses.get(a) == "pending"
                    for a in orchestrator.agent_names
                )
                if any_pending:
                    with st.spinner("⏳ Running all agents…"):
                        for aname in orchestrator.agent_names:
                            if st.session_state.agent_statuses.get(aname) == "pending":
                                st.session_state.agent_statuses[aname] = "running"
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
                    st.session_state.run_all_triggered = False
                    st.rerun()


                # All done
                all_done = all(
                    st.session_state.agent_statuses.get(a) in ("done", "waiting")
                    for a in orchestrator.agent_names
                )
                if all_done:
                    st.session_state.run_all_triggered = False

            # ── If Run All results exist, show in tabs ────────
            if st.session_state.all_agents_results:
                st.markdown("<br>", unsafe_allow_html=True)
                result_tabs = st.tabs(list(st.session_state.all_agents_results.keys()))
                for tab, (aname, ares) in zip(result_tabs, st.session_state.all_agents_results.items()):
                    with tab:
                        st.markdown(ares)
                st.markdown("---")

            # ── Agent cards grid ──────────────────────────────
            st.markdown("<p style='text-align:center; color:#5a7abf; font-size:14px; margin:8px 0 20px;'>Click a card to run a single agent</p>", unsafe_allow_html=True)

            agents_list = list(orchestrator.agent_descriptions.items())
            row1 = st.columns(3)
            row2 = st.columns(3)
            grid_cols = list(row1) + list(row2)

            for idx, (agent_name, agent_desc) in enumerate(agents_list):
                icon, color = agent_icons.get(agent_name, ("🤖", "#4a90d9"))
                status_key = st.session_state.agent_statuses.get(agent_name, "")
                status_label, status_color, status_bg = status_style.get(status_key, status_style[""])
                if agent_name == "📧 Email Report" and status_key == "done":
                    status_label = "📝 Draft Ready"
                done_border = "rgba(46,170,94,0.4)" if status_key == "done" else "rgba(74,144,217,0.12)"

                with grid_cols[idx]:
                    st.markdown(f"""
                    <div class="agent-grid-card {'done' if status_key=='done' else ''}">
                        <div class="agent-card-status" style="background:{status_bg}; color:{status_color};">{status_label}</div>
                        <span class="agent-card-icon">{icon}</span>
                        <div class="agent-card-name">{agent_name}</div>
                        <div class="agent-card-desc">{agent_desc}</div>
                    </div>
                    """, unsafe_allow_html=True)

                    btn_label = "✅ View Result" if status_key == "done" else "▶ Run Agent"
                    if st.button(btn_label, key=f"card_btn_{agent_name}", use_container_width=True):
                        st.session_state.active_tab = "agents"
                        if agent_name in st.session_state.all_agents_results:
                            st.session_state.agent_result = st.session_state.all_agents_results[agent_name]
                            st.session_state.agent_running = False
                        else:
                            st.session_state.agent_result = None
                            st.session_state.agent_running = True
                        st.session_state.active_agent = agent_name
                        st.session_state.agent_statuses[agent_name] = "running"
                        st.rerun()

    # ══════════════════════════════════════════════════════════
    #  CHAT TAB
    # ══════════════════════════════════════════════════════════
    elif active_tab == "chat":

        # Render all messages
        for message in st.session_state.chat_history:
            if message["role"] == "user":
                st.markdown(
                    f'<div style="display:flex;justify-content:flex-end;margin:8px 0;">'
                    f'<div style="background:#dcf8c6;color:#111;border-radius:18px 18px 4px 18px;'
                    f'padding:10px 16px;max-width:72%;font-size:15px;line-height:1.6;'
                    f'box-shadow:0 1px 2px rgba(0,0,0,0.12);font-family:Nunito,sans-serif;">'
                    f'{html.escape(message["content"])}</div>'
                    f'<div style="width:36px;height:36px;background:linear-gradient(135deg,#1a3d8f,#4a90d9);'
                    f'border-radius:50%;display:flex;align-items:center;justify-content:center;'
                    f'font-size:18px;margin-left:8px;flex-shrink:0;">&#128100;</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
            else:
                safe_content = _render_message(message["content"])
                st.markdown(
                    f'<div style="display:flex;justify-content:flex-start;margin:8px 0;">'
                    f'<div style="width:36px;height:36px;background:linear-gradient(135deg,#ffffff,#e8f0ff);'
                    f'border-radius:50%;display:flex;align-items:center;justify-content:center;'
                    f'font-size:18px;margin-right:8px;flex-shrink:0;'
                    f'box-shadow:0 2px 8px rgba(13,43,110,0.15);">&#127973;</div>'
                    f'<div style="background:#ffffff;color:#111;border-radius:18px 18px 18px 4px;'
                    f'padding:12px 16px;max-width:72%;font-size:15px;line-height:1.7;'
                    f'box-shadow:0 1px 2px rgba(0,0,0,0.12);font-family:Nunito,sans-serif;">'
                    f'{safe_content}</div></div>',
                    unsafe_allow_html=True,
                )

        if st.session_state.llm is None:
            st.markdown("""
            <div style="text-align:center;margin:32px auto;padding:28px 32px;
                background:white;border-radius:20px;max-width:480px;
                box-shadow:0 4px 20px rgba(13,43,110,0.08);
                border:1.5px dashed rgba(74,144,217,0.3);">
                <div style="font-size:44px;margin-bottom:12px;">📤</div>
                <p style="color:#0d2b6e;font-size:16px;font-weight:700;margin:0 0 6px;">
                    No documents loaded
                </p>
                <p style="color:#5a7abf;font-size:14px;margin:0;line-height:1.6;">
                    Upload your medical documents from the sidebar to start chatting.
                </p>
            </div>
            """, unsafe_allow_html=True)
        else:
            voice_transcript = st.text_input(
                "voice_hidden",
                value="",
                key="voice_input_field",
                label_visibility="collapsed",
            )

            # Hide the text input visually
            st.markdown("""
            <style>
            div[data-testid="stTextInput"]:has(input[aria-label="voice_hidden"]) {
                position: absolute !important;
                left: -9999px !important;
                opacity: 0 !important;
                pointer-events: none !important;
                height: 0 !important;
                overflow: hidden !important;
            }
            </style>
            """, unsafe_allow_html=True)


            current_ui_lang = st.session_state.get("user_language", "English")
            lang_code_map = {"English": "en-IN", "Hindi": "hi-IN", "Malayalam": "ml-IN"}
            recognition_lang = lang_code_map.get(current_ui_lang, "en-IN")

            # Mic button + JS — injected via components.html (display only, no return needed)
            components.html(
                f"""
                <div style="display:flex;align-items:center;justify-content:flex-end;
                    padding:4px 4px 2px;gap:10px;">

                    <span id="voice-status" style="
                        font-size:12px;color:#5a7abf;
                        font-family:'Nunito',sans-serif;
                        opacity:0;transition:opacity 0.3s;
                        font-style:italic;">
                        Listening…
                    </span>

                    <button id="mic-btn" onclick="toggleVoice()" title="Voice input"
                        style="width:44px;height:44px;border-radius:50%;
                        border:2px solid #4a90d9;background:white;cursor:pointer;
                        display:flex;align-items:center;justify-content:center;
                        transition:all 0.25s;flex-shrink:0;padding:0;
                        box-shadow:0 2px 8px rgba(74,144,217,0.2);">
                        <svg id="mic-icon" width="20" height="20" viewBox="0 0 24 24"
                            fill="none" stroke="#4a90d9" stroke-width="2"
                            stroke-linecap="round" stroke-linejoin="round">
                            <rect x="9" y="2" width="6" height="11" rx="3"/>
                            <path d="M5 10a7 7 0 0 0 14 0"/>
                            <line x1="12" y1="17" x2="12" y2="22"/>
                            <line x1="8" y1="22" x2="16" y2="22"/>
                        </svg>
                    </button>
                </div>

                <script>
                const RECOGNITION_LANG = "{recognition_lang}";
                let recognition = null;
                let isListening  = false;

                const LANG_MAP = {{
                    'hi': 'Hindi', 'hi-in': 'Hindi',
                    'ml': 'Malayalam', 'ml-in': 'Malayalam',
                    'en': 'English', 'en-us': 'English',
                    'en-in': 'English', 'en-gb': 'English',
                }};

                function detectLang(code) {{
                    if (!code) return 'English';
                    const lower = code.toLowerCase();
                    return LANG_MAP[lower] || LANG_MAP[lower.split('-')[0]] || '{current_ui_lang}';
                }}

                // Find Streamlit's hidden text input in the parent frame and set its value,
                // then fire React's synthetic onChange so Streamlit picks it up.
                function setStreamlitInput(value) {{
                    const inputs = window.parent.document.querySelectorAll('input[type="text"]');
                    for (const inp of inputs) {{
                        // Our hidden input has aria-label matching its label
                        if (inp.getAttribute('aria-label') === 'voice_hidden') {
                            const nativeInputValueSetter = Object.getOwnPropertyDescriptor(
                                window.HTMLInputElement.prototype, 'value'
                            ).set;
                            nativeInputValueSetter.call(inp, value);
                                inp.dispatchEvent(new Event('input', {{ bubbles: true }}));
                            break;
                        }
                    }}
                }}

                function toggleVoice() {{
                    const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
                    if (!SR) {{
                        alert('Voice input needs Chrome or Edge. Please switch browsers.');
                        return;
                    }}

                    if (isListening) {{
                        recognition.stop();
                        return;
                    }}

                    recognition = new SR();
                    recognition.continuous     = false;
                    recognition.interimResults = false;
                    recognition.lang           = RECOGNITION_LANG;

                   recognition.onstart = () => {{
                        isListening = true;
                        const btn  = document.getElementById('mic-btn');
                        const icon = document.getElementById('mic-icon');
                        btn.style.background   = '#fef2f2';
                        btn.style.borderColor  = '#ef4444';
                        btn.style.boxShadow    = '0 2px 12px rgba(239,68,68,0.35)';
                        icon.setAttribute('stroke', '#ef4444');
                        document.getElementById('voice-status').style.opacity = '1';
                    }};

                    recognition.onresult = (e) => {{
                        const transcript = e.results[0][0].transcript.trim();

                        // Try to get detected language from the result
                        // (Chrome exposes this on the SpeechRecognitionResult)
                        let rawLang = '';
                        try { rawLang = e.results[0][0].lang || ''; } catch(_) {}

                        const detectedLang = detectLang(rawLang);

                        // Bundle transcript + lang into JSON and write to the hidden input
                        const payload = JSON.stringify({{ text: transcript, lang: detectedLang }});
                        setStreamlitInput(payload);
                    }};

                    recognition.onerror = (e) => {{
                        console.warn('Speech recognition error:', e.error);
                        stopListening();
                    }};

                    recognition.onend = stopListening;

                    recognition.start();
                }}

                function stopListening() {{
                    isListening = false;
                    const btn  = document.getElementById('mic-btn');
                    const icon = document.getElementById('mic-icon');
                    btn.style.background  = 'white';
                    btn.style.borderColor = '#4a90d9';
                    btn.style.boxShadow   = '0 2px 8px rgba(74,144,217,0.2)';
                    icon.setAttribute('stroke', '#4a90d9');
                    document.getElementById('voice-status').style.opacity = '0';
                }}
                </script>
                """,
                height=58,
            )

            # ── Process voice transcript if received ──────────────
            if voice_transcript and voice_transcript.strip():
                # Clear it immediately so it doesn't re-fire on next rerun
                st.session_state.voice_input_field = ""
                try:
                    import json as _json
                    payload    = _json.loads(voice_transcript)
                    voice_text = payload.get("text", "").strip()
                    voice_lang = payload.get("lang", "English")
                except Exception:
                    # Fallback: treat raw string as plain transcript
                    voice_text = voice_transcript.strip()
                    voice_lang = st.session_state.get("user_language", "English")

                if voice_text:
                    # Auto-switch language if different from current
                    current_lang = st.session_state.get("user_language", "English")
                    if voice_lang != current_lang and voice_lang in ["English", "Hindi", "Malayalam"]:
                        st.session_state.user_language = voice_lang
                        from langchain_groq import ChatGroq as _ChatGroq
                        from langchain_core.output_parsers import StrOutputParser as _SOP
                        _lang_llm = _ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0.1)
                        ack = (_lang_llm | _SOP()).invoke(
                            f"In {voice_lang} only, write one warm sentence (max 10 words) "
                            f"saying you detected they're speaking {voice_lang} and will "
                            f"reply in it. Keep emojis."
                        )
                        st.session_state.chat_history.append({"role": "assistant", "content": ack})

                    # Send through same pipeline as typed questions
                    history_before = list(st.session_state.chat_history)
                    st.session_state.chat_history.append({"role": "user", "content": voice_text})
                    with st.spinner("Reading your documents…"):
                        try:
                            answer = get_answer(
                                llm=st.session_state.llm,
                                retriever=st.session_state.retriever,
                                question=voice_text,
                                chat_history=history_before,
                                mood=st.session_state.user_mood,
                                user_name=st.session_state.user_name,
                                user_conditions=st.session_state.user_conditions,
                                user_whom=st.session_state.user_whom,
                                user_age=st.session_state.user_age,
                                summaries=st.session_state.get("summaries", {}),
                                user_language=st.session_state.get("user_language", "English"),
                            )
                        except Exception as e:
                            answer = f"Something went wrong processing your voice input. (Error: {e})"
                    st.session_state.chat_history.append({"role": "assistant", "content": answer})
                    st.rerun()

            # ── Text chat input ───────────────────────────────────
            user_question = st.chat_input("Ask about your medical documents…")
            if user_question:
                history_before = list(st.session_state.chat_history)
                st.session_state.chat_history.append({"role": "user", "content": user_question})
                with st.spinner("Reading your documents…"):
                    try:
                        answer = get_answer(
                            llm=st.session_state.llm,
                            retriever=st.session_state.retriever,
                            question=user_question,
                            chat_history=history_before,
                            mood=st.session_state.user_mood,
                            user_name=st.session_state.user_name,
                            user_conditions=st.session_state.user_conditions,
                            user_whom=st.session_state.user_whom,
                            user_age=st.session_state.user_age,
                            summaries=st.session_state.get("summaries", {}),
                            user_language=st.session_state.get("user_language", "English"),
                        )
                    except Exception as e:
                        answer = f"Something went wrong while reading your documents. (Error: {e})"
                st.session_state.chat_history.append({"role": "assistant", "content": answer})
                st.rerun()


    # ══════════════════════════════════════════════════════════
    #  DOCUMENTS TAB
    # ══════════════════════════════════════════════════════════
    elif active_tab == "documents":

        st.markdown("""
        <div style="padding: 32px 0 8px;">
            <div class="docs-section-title">📁 Your Documents</div>
            <div class="docs-section-sub">All uploaded and processed medical files</div>
        </div>
        """, unsafe_allow_html=True)

        if not st.session_state.uploaded_names:
            st.markdown("""
            <div style="text-align:center;margin:24px auto;padding:40px 32px;
                background:white;border-radius:24px;max-width:480px;
                box-shadow:0 4px 24px rgba(13,43,110,0.08);
                border:2px dashed rgba(74,144,217,0.25);">
                <div style="font-size:52px;margin-bottom:16px;">📂</div>
                <p style="color:#0d2b6e;font-size:17px;font-weight:700;margin:0 0 8px;">
                    No documents uploaded yet
                </p>
                <p style="color:#5a7abf;font-size:14px;margin:0;line-height:1.6;">
                    Use the sidebar to upload PDF, DOCX, image, or text files.<br>
                    Once processed, they'll appear here.
                </p>
            </div>
            """, unsafe_allow_html=True)
        else:
            for doc_name in st.session_state.uploaded_names:
                ext = Path(doc_name).suffix.lower()
                ext_icon = {"pdf":"📄","docx":"📝","jpg":"🖼️","jpeg":"🖼️","png":"🖼️","webp":"🖼️","txt":"📃"}.get(ext.lstrip("."), "📄")
                st.markdown(f"""
                <div class="doc-card">
                    <div class="doc-card-icon">{ext_icon}</div>
                    <div>
                        <div class="doc-card-name">{html.escape(doc_name)}</div>
                        <div class="doc-card-status">✅ Processed &amp; indexed</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            if st.session_state.get("summaries"):
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown('<div class="docs-section-title" style="font-size:20px;">📋 Document Summaries</div>', unsafe_allow_html=True)
                for doc_name, summary in st.session_state.summaries.items():
                    with st.expander(f"📄 {doc_name}", expanded=False):
                        st.markdown(summary)
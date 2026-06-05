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
            if s.startswith("## "):
                text = html.escape(s[3:])
                parts.append(f"<p style='font-size:15px;font-weight:800;color:#0d2b6e;margin:14px 0 4px;'>{text}</p>")
            elif s.startswith("### "):
                text = html.escape(s[4:])
                parts.append(f"<p style='font-size:14px;font-weight:700;color:#2451b3;margin:10px 0 4px;'>{text}</p>")
            else:
                text = _re2.sub(r"\*\*(.*?)\*\*", r"<strong>\1</strong>", html.escape(s))
                parts.append(f"<p style='margin:3px 0;line-height:1.7;'>{text}</p>")

    if in_list:
        parts.append("</ul>")
    return "".join(parts)


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
                items_html += (
                    f"<div style='display:flex;gap:8px;margin:6px 0;'>"
                    f"<span style='color:{accent};font-weight:700;'>•</span>"
                    f"<span style='color:#1a1a2e;font-size:14px;'>{html.escape(s[2:])}</span></div>"
                )

        if current_appt:
            appt_blocks.append(current_appt)

        for appt in appt_blocks:
            details_html = ""
            for d in appt.get("details", []):
                key, _, val = d.partition(":")
                if val:
                    details_html += (
                        f"<div style='font-size:13px;color:#555;margin:3px 0;'>"
                        f"<span style='font-weight:700;color:{accent};'>{html.escape(key.strip())}:</span>"
                        f" {html.escape(val.strip())}</div>"
                    )
                else:
                    details_html += f"<div style='font-size:13px;color:#555;margin:3px 0;'>{html.escape(d)}</div>"
            items_html += (
                f"<div style='background:white;border-radius:12px;padding:12px 16px;margin:8px 0;"
                f"border-left:4px solid {accent};box-shadow:0 2px 8px rgba(0,0,0,0.06);'>"
                f"<div style='font-size:14px;font-weight:800;color:{accent};margin-bottom:6px;'>"
                f"📅 {html.escape(appt['date'])}</div>{details_html}</div>"
            )

        st.markdown(
            f"<div style='background:{bg};border:1.5px solid {border};border-radius:20px;"
            f"padding:20px 24px;margin:12px 0;'>"
            f"<div style='font-size:16px;font-weight:800;color:{accent};margin-bottom:12px;"
            f"padding-bottom:8px;border-bottom:1.5px solid {border};'>{html.escape(title_raw)}</div>"
            f"{items_html}</div>",
            unsafe_allow_html=True,
        )

def _render_scan_result(raw: str):
    import re as re3
    import html as _html

    # ── parse sections ────────────────────────────────────────
    parts = re3.split(r'\n##\s+', raw.strip())
    sections = {}
    for part in parts:
        lines = part.strip().split("\n")
        if not lines:
            continue
        header_raw = lines[0]
        body = "\n".join(lines[1:]).strip()
        header_clean = re3.sub(r'[^\w\s]', '', header_raw).strip().lower()
        sections[header_clean] = body

    def get(keys):
        for k in keys:
            for sk, sv in sections.items():
                if k in sk:
                    return sv
        return ""

    scan_type    = get(["what kind", "kind of scan"])
    in_picture   = get(["picture", "whats in"])
    looks_good   = get(["looks good", "what looks"])
    worth_noting = get(["worth noting", "noting"])
    ask_doctor   = get(["ask your doctor", "ask"])
    clinical     = get(["clinical summary", "clinical"])

    # ── detect body region for diagram ───────────────────────
    scan_lower = (scan_type + " " + in_picture + " " + worth_noting).lower()
    if any(w in scan_lower for w in ["chest", "lung", "heart", "thorax", "rib", "clavicle", "pulmon"]):
        body_region = "chest"
    elif any(w in scan_lower for w in ["brain", "head", "skull", "cranial", "cerebr", "neuro"]):
        body_region = "head"
    elif any(w in scan_lower for w in ["knee", "femur", "tibia", "patella", "meniscus"]):
        body_region = "knee"
    elif any(w in scan_lower for w in ["spine", "vertebra", "lumbar", "cervical", "thoracic", "disc", "sacr"]):
        body_region = "spine"
    elif any(w in scan_lower for w in ["abdomen", "liver", "kidney", "gallbladder", "pancreas", "bowel", "colon"]):
        body_region = "abdomen"
    elif any(w in scan_lower for w in ["shoulder", "humerus", "rotator", "scapula"]):
        body_region = "shoulder"
    else:
        body_region = "chest"

    # ── detect problem area from worth_noting ─────────────────
    worth_lower = worth_noting.lower()
    has_issue = not any(p in worth_lower for p in ["nothing alarming", "no alarming", "looks normal", "no issues", "appears normal"])

    # ── build problem spots per region ───────────────────────
    problem_spots = []
    if body_region == "chest":
        if any(w in worth_lower for w in ["left lung", "left", "opaque", "opacity", "smaller"]):
            problem_spots.append({"x": 38, "y": 38, "label": "Left lung"})
        if any(w in worth_lower for w in ["right lung", "consolidat", "infiltrat", "haziness", "hazy"]):
            problem_spots.append({"x": 62, "y": 38, "label": "Right lung"})
        if any(w in worth_lower for w in ["heart", "cardiac", "cardiomegaly"]):
            problem_spots.append({"x": 50, "y": 44, "label": "Heart"})
    elif body_region == "head":
        if any(w in worth_lower for w in ["left", "temporal", "frontal"]):
            problem_spots.append({"x": 38, "y": 42, "label": "Left side"})
        if any(w in worth_lower for w in ["right"]):
            problem_spots.append({"x": 62, "y": 42, "label": "Right side"})
        if any(w in worth_lower for w in ["mass", "lesion", "tumor", "bleed", "hemorrhage"]):
            problem_spots.append({"x": 50, "y": 38, "label": "Area of concern"})
    elif body_region == "spine":
        if any(w in worth_lower for w in ["lumbar", "lower"]):
            problem_spots.append({"x": 50, "y": 72, "label": "Lower spine"})
        if any(w in worth_lower for w in ["cervical", "neck", "upper"]):
            problem_spots.append({"x": 50, "y": 22, "label": "Upper spine"})
        if any(w in worth_lower for w in ["disc", "bulge", "herniat"]):
            problem_spots.append({"x": 50, "y": 58, "label": "Disc area"})

    if has_issue and not problem_spots:
        # generic fallback spot
        region_defaults = {
            "chest": {"x": 50, "y": 40, "label": "Area of concern"},
            "head": {"x": 50, "y": 40, "label": "Area of concern"},
            "knee": {"x": 50, "y": 50, "label": "Joint area"},
            "spine": {"x": 50, "y": 55, "label": "Spinal area"},
            "abdomen": {"x": 50, "y": 55, "label": "Abdominal area"},
            "shoulder": {"x": 55, "y": 35, "label": "Shoulder joint"},
        }
        problem_spots.append(region_defaults.get(body_region, {"x": 50, "y": 45, "label": "Area of concern"}))

    # ── SVG body diagrams ─────────────────────────────────────
    def make_svg(region, spots):
        # Build animated pulse circles for each problem spot
        pulses = ""
        for s in spots:
            px, py = s["x"], s["y"]
            lbl = _html.escape(s["label"])
            pulses += f"""
  <circle cx="{px}" cy="{py}" r="4" fill="#ef4444" opacity="0.9">
    <animate attributeName="r" values="4;9;4" dur="1.8s" repeatCount="indefinite"/>
    <animate attributeName="opacity" values="0.9;0.1;0.9" dur="1.8s" repeatCount="indefinite"/>
  </circle>
  <circle cx="{px}" cy="{py}" r="3.5" fill="#ef4444"/>
  <text x="{px}" y="{py - 12}" text-anchor="middle" font-size="5.5" fill="#ef4444" font-weight="bold" font-family="Nunito,sans-serif">{lbl}</text>"""

        if region == "chest":
            return f"""<svg viewBox="0 0 100 110" xmlns="http://www.w3.org/2000/svg" style="width:100%;height:100%;">
  <!-- body outline -->
  <rect x="28" y="18" width="44" height="62" rx="8" fill="#dce8ff" stroke="#4a90d9" stroke-width="1.2"/>
  <!-- neck -->
  <rect x="42" y="10" width="16" height="12" rx="4" fill="#dce8ff" stroke="#4a90d9" stroke-width="1.2"/>
  <!-- head -->
  <ellipse cx="50" cy="7" rx="10" ry="8" fill="#dce8ff" stroke="#4a90d9" stroke-width="1.2"/>
  <!-- ribs left -->
  <path d="M44 30 Q36 33 35 38" stroke="#7aaee0" stroke-width="0.8" fill="none"/>
  <path d="M44 36 Q35 39 34 44" stroke="#7aaee0" stroke-width="0.8" fill="none"/>
  <path d="M44 42 Q35 45 35 50" stroke="#7aaee0" stroke-width="0.8" fill="none"/>
  <!-- ribs right -->
  <path d="M56 30 Q64 33 65 38" stroke="#7aaee0" stroke-width="0.8" fill="none"/>
  <path d="M56 36 Q65 39 66 44" stroke="#7aaee0" stroke-width="0.8" fill="none"/>
  <path d="M56 42 Q65 45 65 50" stroke="#7aaee0" stroke-width="0.8" fill="none"/>
  <!-- lungs -->
  <ellipse cx="38" cy="38" rx="9" ry="13" fill="#bfdbfe" opacity="0.7" stroke="#60a5fa" stroke-width="0.8"/>
  <ellipse cx="62" cy="38" rx="9" ry="13" fill="#bfdbfe" opacity="0.7" stroke="#60a5fa" stroke-width="0.8"/>
  <!-- heart -->
  <ellipse cx="50" cy="44" rx="5" ry="6" fill="#fca5a5" opacity="0.85" stroke="#f87171" stroke-width="0.8"/>
  <!-- spine -->
  <line x1="50" y1="22" x2="50" y2="78" stroke="#94a3b8" stroke-width="1.5" stroke-dasharray="2,2"/>
  <!-- labels -->
  <text x="38" y="55" text-anchor="middle" font-size="4.5" fill="#1e40af" font-family="Nunito,sans-serif">L Lung</text>
  <text x="62" y="55" text-anchor="middle" font-size="4.5" fill="#1e40af" font-family="Nunito,sans-serif">R Lung</text>
  <text x="50" y="53" text-anchor="middle" font-size="3.8" fill="#991b1b" font-family="Nunito,sans-serif">Heart</text>
  {pulses}
</svg>"""

        elif region == "head":
            return f"""<svg viewBox="0 0 100 110" xmlns="http://www.w3.org/2000/svg" style="width:100%;height:100%;">
  <!-- skull -->
  <ellipse cx="50" cy="42" rx="28" ry="32" fill="#dce8ff" stroke="#4a90d9" stroke-width="1.2"/>
  <!-- jaw -->
  <path d="M30 58 Q50 78 70 58" fill="#dce8ff" stroke="#4a90d9" stroke-width="1.2"/>
  <!-- brain outline -->
  <ellipse cx="50" cy="36" rx="20" ry="22" fill="#bfdbfe" opacity="0.5" stroke="#60a5fa" stroke-width="0.8"/>
  <!-- left hemisphere -->
  <path d="M50 18 Q32 22 30 40 Q31 55 50 58" stroke="#7aaee0" stroke-width="0.8" fill="none"/>
  <!-- right hemisphere -->
  <path d="M50 18 Q68 22 70 40 Q69 55 50 58" stroke="#7aaee0" stroke-width="0.8" fill="none"/>
  <!-- midline -->
  <line x1="50" y1="16" x2="50" y2="58" stroke="#94a3b8" stroke-width="0.8" stroke-dasharray="2,2"/>
  <!-- eyes -->
  <ellipse cx="40" cy="66" rx="5" ry="3" fill="white" stroke="#4a90d9" stroke-width="0.8"/>
  <ellipse cx="60" cy="66" rx="5" ry="3" fill="white" stroke="#4a90d9" stroke-width="0.8"/>
  <!-- nose -->
  <path d="M47 72 Q50 76 53 72" stroke="#4a90d9" stroke-width="0.8" fill="none"/>
  <!-- labels -->
  <text x="38" y="34" text-anchor="middle" font-size="4.5" fill="#1e40af" font-family="Nunito,sans-serif">L</text>
  <text x="62" y="34" text-anchor="middle" font-size="4.5" fill="#1e40af" font-family="Nunito,sans-serif">R</text>
  {pulses}
</svg>"""

        elif region == "spine":
            return f"""<svg viewBox="0 0 100 110" xmlns="http://www.w3.org/2000/svg" style="width:100%;height:100%;">
  <!-- vertebrae -->
  {"".join(f'<rect x="42" y="{10 + i*9}" width="16" height="7" rx="2" fill="#dce8ff" stroke="#4a90d9" stroke-width="1"/>' for i in range(9))}
  <!-- disc pads -->
  {"".join(f'<rect x="43" y="{17 + i*9}" width="14" height="2" rx="1" fill="#bfdbfe" opacity="0.8"/>' for i in range(8))}
  <!-- spinal canal -->
  <line x1="50" y1="10" x2="50" y2="88" stroke="#94a3b8" stroke-width="1" stroke-dasharray="1.5,1.5"/>
  <!-- labels -->
  <text x="68" y="18" font-size="4.5" fill="#1e40af" font-family="Nunito,sans-serif">Cervical</text>
  <text x="68" y="40" font-size="4.5" fill="#1e40af" font-family="Nunito,sans-serif">Thoracic</text>
  <text x="68" y="65" font-size="4.5" fill="#1e40af" font-family="Nunito,sans-serif">Lumbar</text>
  <text x="68" y="82" font-size="4.5" fill="#1e40af" font-family="Nunito,sans-serif">Sacrum</text>
  {pulses}
</svg>"""

        else:
            # generic torso fallback
            return f"""<svg viewBox="0 0 100 110" xmlns="http://www.w3.org/2000/svg" style="width:100%;height:100%;">
  <rect x="28" y="18" width="44" height="72" rx="8" fill="#dce8ff" stroke="#4a90d9" stroke-width="1.2"/>
  <rect x="42" y="10" width="16" height="12" rx="4" fill="#dce8ff" stroke="#4a90d9" stroke-width="1.2"/>
  <ellipse cx="50" cy="7" rx="10" ry="8" fill="#dce8ff" stroke="#4a90d9" stroke-width="1.2"/>
  <ellipse cx="38" cy="38" rx="9" ry="13" fill="#bfdbfe" opacity="0.6" stroke="#60a5fa" stroke-width="0.8"/>
  <ellipse cx="62" cy="38" rx="9" ry="13" fill="#bfdbfe" opacity="0.6" stroke="#60a5fa" stroke-width="0.8"/>
  <ellipse cx="50" cy="60" rx="12" ry="15" fill="#c7d2fe" opacity="0.5" stroke="#818cf8" stroke-width="0.8"/>
  {pulses}
</svg>"""

    svg_diagram = make_svg(body_region, problem_spots)
    has_diagram = bool(problem_spots)

    # ── build bullet html (strips ** from labels) ─────────────
    def bullets_html(text):
        items = [l.lstrip("-* ").strip() for l in text.split("\n") if l.strip().startswith(("-", "*", "•"))]
        if not items:
            items = [s.strip() for s in text.split("\n") if s.strip()]
        out = ""
        for item in items:
            # strip stray ** that appear before the em-dash
            item_clean = re3.sub(r'\*\*', '', item)
            bold_part = re3.sub(r'__(.+?)__', r'<strong>\1</strong>', _html.escape(item_clean))
            out += (
                "<div style='display:flex;gap:12px;align-items:flex-start;padding:10px 0;"
                "border-bottom:1px solid #eef2ff;'>"
                "<span style='color:#4a90d9;font-size:16px;flex-shrink:0;margin-top:2px;'>▸</span>"
                f"<span style='font-size:14px;color:#1a1a2e;line-height:1.6;'>{bold_part}</span>"
                "</div>"
            )
        return out

    # ── question cards ────────────────────────────────────────
    q_items = [l.lstrip("-* 0123456789.").strip() for l in ask_doctor.split("\n") if l.strip() and len(l.strip()) > 8]
    q_cards = ""
    for q in q_items[:4]:
        safe_q = _html.escape(re3.sub(r'\*\*', '', q))
        q_cards += (
            f"<div style='background:#f0f4ff;border:1.5px solid #c5d4f5;border-radius:14px;"
            f"padding:14px 16px;margin-bottom:10px;font-size:14px;color:#1a2b5e;line-height:1.5;'>"
            f"<span style='margin-right:8px;'>📋</span>{safe_q}"
            f"</div>"
        )

    # ── clinical lines ────────────────────────────────────────
    clinical_lines = [re3.sub(r'\*\*', '', l.strip().lstrip("- ")) for l in clinical.split("\n") if l.strip()]
    clinical_inner = "".join(
        f"<div style='font-size:13px;color:#3a3a5c;padding:5px 0;border-bottom:1px solid #e8eeff;'>"
        f"<span style='color:#6a7abf;margin-right:6px;'>›</span>{_html.escape(l)}</div>"
        for l in clinical_lines
    )

    worth_safe = _html.escape(re3.sub(r'\*\*', '', worth_noting))
    good_safe  = _html.escape(re3.sub(r'\*\*', '', looks_good))
    scan_safe  = _html.escape(re3.sub(r'\*\*', '', scan_type)) if scan_type else "Medical scan analysis ready."

    # ── body diagram panel ────────────────────────────────────
    diagram_label = {"chest": "Chest", "head": "Head / Brain", "knee": "Knee", "spine": "Spine", "abdomen": "Abdomen", "shoulder": "Shoulder"}.get(body_region, "Body")
    issue_count = len(problem_spots)
    issue_badge = (
        f"<span style='background:#fef2f2;color:#dc2626;font-size:11px;font-weight:700;"
        f"padding:3px 10px;border-radius:20px;margin-left:8px;'>{issue_count} area{'s' if issue_count > 1 else ''} flagged</span>"
        if has_diagram and has_issue else
        "<span style='background:#f0fdf4;color:#16a34a;font-size:11px;font-weight:700;padding:3px 10px;border-radius:20px;margin-left:8px;'>No concerns detected</span>"
    )

    diagram_panel = (
        "<div style='animation:scanFadeUp 0.5s 0.08s ease both;background:white;"
        "border:1.5px solid #e0e8f8;border-radius:20px;padding:22px 24px;margin-bottom:14px;'>"
        "<div style='display:flex;align-items:center;gap:8px;margin-bottom:16px;'>"
        "<div style='width:36px;height:36px;background:#eef3ff;border-radius:10px;"
        "display:flex;align-items:center;justify-content:center;font-size:18px;'>🫁</div>"
        f"<div style='font-size:16px;font-weight:800;color:#0d2b6e;'>{diagram_label} — Where to look</div>"
        f"{issue_badge}"
        "</div>"
        "<div style='display:grid;grid-template-columns:160px 1fr;gap:20px;align-items:start;'>"
        f"<div style='height:180px;'>{svg_diagram}</div>"
        "<div>"
        + (
            "".join(
                f"<div style='display:flex;align-items:center;gap:10px;padding:10px 14px;"
                f"background:#fef2f2;border:1.5px solid #fecaca;border-radius:12px;margin-bottom:8px;'>"
                f"<span style='font-size:18px;animation:scanPulse 1.8s infinite;'>🔴</span>"
                f"<div><div style='font-size:13px;font-weight:700;color:#991b1b;'>{_html.escape(s['label'])}</div>"
                f"<div style='font-size:12px;color:#7f1d1d;'>Marked on diagram</div></div></div>"
                for s in problem_spots
            )
            if problem_spots and has_issue else
            "<div style='font-size:13px;color:#16a34a;padding:10px 0;'>Everything appears within normal range on this scan.</div>"
        )
        + "</div></div></div>"
    )

    # ── final html assembly ───────────────────────────────────
    html_out = (
        "<style>"
        "@keyframes scanFadeUp{from{opacity:0;transform:translateY(18px)}to{opacity:1;transform:translateY(0)}}"
        "@keyframes scanPulse{0%,100%{transform:scale(1);opacity:1}50%{transform:scale(1.15);opacity:0.6}}"
        "</style>"
        "<div style='font-family:Nunito,sans-serif;max-width:720px;'>"

        # dark header (no duplicate title, just scan type sentence)
        "<div style='animation:scanFadeUp 0.5s ease both;background:linear-gradient(135deg,#071a4a,#1a3d8f);"
        "border-radius:24px;padding:24px 28px;margin-bottom:16px;color:white;'>"
        "<div style='display:flex;align-items:center;gap:12px;margin-bottom:10px;'>"
        "<span style='font-size:36px;animation:scanPulse 2s infinite;'>🩻</span>"
        "<div style='font-size:11px;letter-spacing:2px;text-transform:uppercase;color:#7a9cd8;'>Scan Analysis</div>"
        "</div>"
        "<div style='background:rgba(255,255,255,0.1);border-radius:12px;padding:10px 16px;"
        f"font-size:15px;line-height:1.6;color:#dce8ff;'>{scan_safe}</div>"
        "</div>"

        # animated body diagram
        + diagram_panel +

        # what's in the picture
        "<div style='animation:scanFadeUp 0.5s 0.15s ease both;background:white;border:1.5px solid #e0e8f8;"
        "border-radius:20px;padding:22px 24px;margin-bottom:14px;'>"
        "<div style='display:flex;align-items:center;gap:10px;margin-bottom:14px;'>"
        "<div style='width:36px;height:36px;background:#eef3ff;border-radius:10px;"
        "display:flex;align-items:center;justify-content:center;font-size:18px;'>🔬</div>"
        "<div style='font-size:16px;font-weight:800;color:#0d2b6e;'>What's in the picture</div>"
        "</div>"
        + bullets_html(in_picture) +
        "</div>"

        # good + noting
        "<div style='display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-bottom:14px;'>"
        "<div style='animation:scanFadeUp 0.5s 0.2s ease both;background:#f0fdf5;border:1.5px solid #bbf0d4;"
        "border-radius:20px;padding:20px 22px;'>"
        "<div style='display:flex;align-items:center;gap:8px;margin-bottom:10px;'>"
        "<span style='font-size:20px;'>✅</span>"
        "<div style='font-size:14px;font-weight:800;color:#14532d;'>What looks good</div>"
        "</div>"
        f"<div style='font-size:13px;color:#166534;line-height:1.65;'>{good_safe}</div>"
        "</div>"
        "<div style='animation:scanFadeUp 0.5s 0.25s ease both;background:#fffbeb;border:1.5px solid #fde68a;"
        "border-radius:20px;padding:20px 22px;'>"
        "<div style='display:flex;align-items:center;gap:8px;margin-bottom:10px;'>"
        "<span style='font-size:20px;'>🔶</span>"
        "<div style='font-size:14px;font-weight:800;color:#78350f;'>Worth noting</div>"
        "</div>"
        f"<div style='font-size:13px;color:#92400e;line-height:1.65;'>{worth_safe}</div>"
        "</div>"
        "</div>"

        # doctor questions
        "<div style='animation:scanFadeUp 0.5s 0.3s ease both;background:white;border:1.5px solid #e0e8f8;"
        "border-radius:20px;padding:22px 24px;margin-bottom:14px;'>"
        "<div style='display:flex;align-items:center;gap:10px;margin-bottom:16px;'>"
        "<div style='width:36px;height:36px;background:#fff0f6;border-radius:10px;"
        "display:flex;align-items:center;justify-content:center;font-size:18px;'>💬</div>"
        "<div><div style='font-size:16px;font-weight:800;color:#0d2b6e;'>Questions to ask your doctor</div>"
        "<div style='font-size:12px;color:#8aaee0;margin-top:1px;'>Bring these to your next appointment</div></div>"
        "</div>"
        + (q_cards if q_cards else "<p style='color:#8aaee0;font-size:13px;'>No questions extracted.</p>") +
        "</div>"

        # clinical summary — uses a unique id so toggle JS is bulletproof
        "<div style='animation:scanFadeUp 0.5s 0.35s ease both;background:#f8f9ff;border:1.5px solid #dce8ff;"
        "border-radius:20px;padding:20px 24px;margin-bottom:16px;'>"
        "<div id='clin-toggle' style='display:flex;align-items:center;justify-content:space-between;cursor:pointer;'"
        " onclick=\"var b=document.getElementById('clin-body');var c=document.getElementById('clin-chev');"
        "if(b.style.display==='none'){b.style.display='block';c.textContent='▲';}else{b.style.display='none';c.textContent='▼';}\""
        ">"
        "<div style='display:flex;align-items:center;gap:10px;'>"
        "<span style='font-size:18px;'>👨‍⚕️</span>"
        "<div style='font-size:14px;font-weight:800;color:#2451b3;'>Clinical summary</div>"
        "<span style='font-size:11px;background:#e8eeff;color:#2451b3;padding:2px 8px;border-radius:20px;'>For your doctor</span>"
        "</div>"
        "<span id='clin-chev' style='color:#8aaee0;font-size:13px;'>▼</span>"
        "</div>"
        "<div id='clin-body' style='display:none;margin-top:14px;border-top:1px solid #dce8ff;padding-top:14px;'>"
        + (clinical_inner if clinical_inner else "<p style='color:#8aaee0;font-size:13px;'>No clinical data available.</p>") +
        "</div>"
        "</div>"

        # disclaimer
        "<div style='animation:scanFadeUp 0.5s 0.4s ease both;text-align:center;"
        "font-size:12px;color:#8aaee0;padding:4px 0 8px;'>"
        "⚠️ AI-assisted interpretation only — always consult a qualified radiologist or physician"
        "</div>"
        "</div>"
    )

    st.markdown(html_out, unsafe_allow_html=True)

st.set_page_config(
    page_title="MediChat — Your Medical Assistant",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Nunito:wght@400;500;600;700;800&family=Playfair+Display:wght@600;700&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] { font-family: 'Nunito', sans-serif; background-color: #f0f4ff; }
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
header { visibility: hidden; }
.block-container { padding: 0 !important; max-width: 100% !important; }
.stApp { background: #f0f4ff; min-height: 100vh; }

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
[data-testid="stSidebar"] [data-testid="stFileUploader"] * { color: #0d2b6e !important; -webkit-text-fill-color: #0d2b6e !important; }
[data-testid="stFileUploader"] {
    background: rgba(255,255,255,0.08) !important;
    border: 2px dashed rgba(74,144,217,0.6) !important;
    border-radius: 16px !important;
    padding: 8px !important;
}


[data-testid="stFileUploader"] p {
    color: #ffffff !important;
    -webkit-text-fill-color: #ffffff !important;
    font-size: 13px !important;
    font-weight: 700 !important;
}
[data-testid="stUploadedFile"] {
    background: white !important;
    border-radius: 8px !important;
    padding: 4px 8px !important;
}
[data-testid="stUploadedFileName"] {
    color: #0d2b6e !important;
    -webkit-text-fill-color: #0d2b6e !important;
    font-weight: 700 !important;
    font-size: 13px !important;
}
[data-testid="stUploadedFile"] span,
[data-testid="stUploadedFile"] div,
[data-testid="stUploadedFile"] p,
[data-testid="stUploadedFile"] * {
    color: #0d2b6e !important;
    -webkit-text-fill-color: #0d2b6e !important;
}
[data-testid="stSidebar"] [data-testid="stUploadedFile"] * {
    color: #0d2b6e !important;
    -webkit-text-fill-color: #0d2b6e !important;
}

[data-testid="stFileUploader"] button {
    background: rgba(74,144,217,0.25) !important;
    border: 1px solid rgba(74,144,217,0.5) !important;
    border-radius: 8px !important;
    color: #0d2b6e !important;
    -webkit-text-fill-color: #0d2b6e !important;
    font-weight: 700 !important;
    font-size: 12px !important;
    padding: 4px 12px !important;
    width: auto !important;
    height: auto !important;
    margin: 0 !important;
}
[data-testid="stSidebar"] .stMarkdown p,
[data-testid="stSidebar"] .stMarkdown li { color: #9ab4e0 !important; font-size: 13px; line-height: 1.7; }

.sidebar-section-label {
    font-size: 9px; font-weight: 800; letter-spacing: 2.5px;
    text-transform: uppercase; color: #4a6aaa !important;
    margin: 0 0 8px; display: block;
}

[data-testid="stFileUploader"]:hover { border-color: rgba(74,144,217,0.9) !important; }
[data-testid="stFileUploader"] [data-testid="stFileUploaderDropzone"] { display: flex !important; align-items: center !important; justify-content: center !important; min-height: 90px !important; border: none !important; background: transparent !important; flex-direction: column !important; gap: 0 !important; }
[data-testid="stFileUploader"] [data-testid="stFileUploaderDropzone"] > *:not(button) { display: none !important; }
[data-testid="stFileUploader"] [data-testid="stFileUploaderDropzone"] button {
    width: 48px !important;
    height: 48px !important;
    border-radius: 50% !important;
    background: rgba(74,144,217,0.15) !important;
    border: 1.5px solid rgba(74,144,217,0.4) !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    padding: 0 !important;
    font-size: 0 !important;
    color: transparent !important;
    -webkit-text-fill-color: transparent !important;
    transition: all 0.2s ease !important;
}



/* Make the arrow icon white */
[data-testid="stFileUploader"] [data-testid="stFileUploaderDropzone"] button svg {
    display: block !important;
    width: 20px !important;
    height: 20px !important;
    fill: none !important;
    stroke: #ffffff !important;
    stroke-width: 2 !important;
}
[data-testid="stFileUploader"] [data-testid="stFileUploaderDropzone"] button:hover {
    background: rgba(74,144,217,0.28) !important;
    border-color: rgba(74,144,217,0.7) !important;
    transform: scale(1.06) !important;
}
[data-testid="stFileUploaderDropzoneInstructions"] {
    display: none !important;
    visibility: hidden !important;
    height: 0 !important;
    max-height: 0 !important;
    overflow: hidden !important;
    pointer-events: none !important;
}

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

.doc-pill {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 12px; padding: 9px 14px;
    font-size: 12px; color: #9ab4e0 !important;
    margin: 5px 0; display: flex;
    align-items: center; gap: 8px;
    word-break: break-all; line-height: 1.4;
}

.chip {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 10px; padding: 8px 12px;
    font-size: 12px; color: #9ab4e0 !important;
    line-height: 1.5; margin: 4px 0; display: block;
}

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
.agent-card-name { font-weight: 800; color: #0d2b6e; font-size: 14px; margin-bottom: 6px; }
.agent-card-desc { font-size: 12px; color: #5a7abf; line-height: 1.5; }

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
.progress-bar-wrap { background: #e8eeff; border-radius: 20px; height: 6px; overflow: hidden; margin-bottom: 20px; }
.progress-bar-fill {
    height: 6px; background: linear-gradient(90deg, #4a90d9, #2451b3);
    border-radius: 20px; animation: bar 3s ease-in-out infinite alternate;
}
.processing-steps { display: flex; flex-direction: column; gap: 10px; text-align: left; }
.step-item { display: flex; align-items: center; gap: 10px; font-size: 13px; color: #5a7abf; }
.step-dot { width: 8px; height: 8px; background: #4a90d9; border-radius: 50%; flex-shrink: 0; animation: pulse 1.5s infinite; }

[data-testid="stAlert"] { border-radius: 14px !important; border: none !important; font-family: 'Nunito', sans-serif !important; font-size: 14px !important; }

[data-testid="stAudioInput"] span,
[data-testid="stAudioInput"] p,
[data-testid="stAudioInput"] time,
[data-testid="stAudioInput"] > div > div:not(:first-child) {
    display: none !important;
}
[data-testid="stAudioInput"] button {
    width: 44px !important;
    height: 44px !important;
    border-radius: 50% !important;
    background: rgba(74,144,217,0.12) !important;
    border: 1.5px solid rgba(74,144,217,0.35) !important;
    box-shadow: none !important;
    transition: all 0.2s ease !important;
}
[data-testid="stAudioInput"] button:hover {
    background: rgba(74,144,217,0.22) !important;
    border-color: rgba(74,144,217,0.6) !important;
    transform: scale(1.05) !important;
}
[data-testid="stAudioInput"] button svg {
    width: 18px !important;
    height: 18px !important;
}


::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(74,144,217,0.3); border-radius: 10px; }

.main > div:first-child { padding-top: 0 !important; }
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
    "image_texts": {},          # ← stores raw clinical vision descriptions
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
    "user_language": "English",
    "prefill_input": "",
    "last_audio_id": None,
    "voice_processed": False,
    "audio_key": 0,
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
    if user_name:
        lines.append(f"Patient  : {user_name}")
    if user_mood:
        lines.append(f"Mood     : {user_mood}")
    if user_conditions:
        lines.append(f"Conditions: {', '.join(user_conditions)}")
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
            self.set_fill_color(13, 43, 110)
            self.rect(0, 0, 210, 22, "F")
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
        pdf.set_x(10)
        pdf.cell(190, 8, "   |   ".join(meta_lines), border=1, fill=True, ln=True)
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
                pdf.set_font("Helvetica", "", 10)
                pdf.set_fill_color(220, 248, 198)
                pdf.set_text_color(20, 60, 20)
                pdf.set_x(10 + 40)
                pdf.multi_cell(150, 6, content, fill=True, border=0)
            else:
                pdf.set_x(10)
                pdf.cell(95, 5, label, align="L", ln=True)
                pdf.set_font("Helvetica", "", 10)
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
    dots_html += "</div>"
    st.markdown(dots_html, unsafe_allow_html=True)

    if step == 1:
        st.markdown(
            '<div class="ob-label">Step 1 of 4</div>'
            '<div class="ob-title">Let\'s get acquainted</div>'
            '<div class="ob-sub">Tell us a bit about yourself and who this report is for</div>',
            unsafe_allow_html=True,
        )
        name = st.text_input("Your name (optional)", value=st.session_state.user_name, placeholder="e.g. Aryan")
        st.markdown('<p style="font-size:13px;font-weight:600;color:#6a7ab5;margin:16px 0 12px;">This report is for:</p>', unsafe_allow_html=True)
        whom_options = [("me", "🧑", "Myself"), ("parent", "👴", "My Parent"), ("child", "👶", "My Child"), ("other", "🤝", "Friend")]
        cols = st.columns(4)
        for i, (key, icon, label) in enumerate(whom_options):
            with cols[i]:
                selected = st.session_state.user_whom == key
                if st.button(f"{icon}\n\n{label}", key=f"whom_{key}", use_container_width=True, type="primary" if selected else "secondary"):
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
        age_titles = {
            "me": "How old are you?",
            "parent": "How old is your parent?",
            "child": "How old is your child?",
            "other": "How old are they?",
        }
        title = age_titles.get(st.session_state.user_whom, "How old are you?")
        st.markdown(
            f'<div class="ob-label">Step 2 of 4</div>'
            f'<div class="ob-title">{title}</div>'
            f'<div class="ob-sub">Drag to set the age</div>',
            unsafe_allow_html=True,
        )
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
        st.markdown(
            '<div class="ob-label">Step 3 of 4</div>'
            '<div class="ob-title">Any existing conditions?</div>'
            '<div class="ob-sub">Select all that apply</div>',
            unsafe_allow_html=True,
        )
        cond_options = [
            ("diabetes", "Diabetes", "🩸"), ("hypertension", "Hypertension", "💓"),
            ("heart", "Heart Disease", "❤️"), ("thyroid", "Thyroid", "🦋"),
            ("asthma", "Asthma", "🫁"), ("neurological", "Neurological", "🧠"),
            ("none", "None", "✅"), ("other", "Other", "➕"),
        ]
        current = set(st.session_state.user_conditions)
        cols = st.columns(2)
        for i, (key, label, icon) in enumerate(cond_options):
            with cols[i % 2]:
                selected = key in current
                if st.button(f"{icon} {label}", key=f"cond_{key}", use_container_width=True, type="primary" if selected else "secondary"):
                    if key == "none":
                        st.session_state.user_conditions = ["none"]
                    else:
                        new = set(st.session_state.user_conditions) - {"none"}
                        if key in new:
                            new.remove(key)
                        else:
                            new.add(key)
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
        st.markdown(
            '<div class="ob-label">Step 4 of 4</div>'
            '<div class="ob-title">How are you feeling?</div>'
            '<div class="ob-sub">We\'ll match our tone to how you feel right now</div>',
            unsafe_allow_html=True,
        )
        st.markdown('<p style="font-size:13px;font-weight:600;color:#6a7ab5;margin:16px 0 8px;">🌐 Preferred Language</p>', unsafe_allow_html=True)
        languages = ["English", "Hindi", "Malayalam"]
        lang_cols1 = st.columns(3)
        for i, lang in enumerate(languages):
            with lang_cols1[i]:
                selected = st.session_state.user_language == lang
                if st.button(lang, key=f"lang_{lang}", use_container_width=True, type="primary" if selected else "secondary"):
                    st.session_state.user_language = lang
                    st.rerun()
        st.markdown("<br>", unsafe_allow_html=True)
        all_moods = [
            ("Happy", "😊"), ("Relieved", "😌"), ("Patient", "🏥"), ("Neutral", "😐"),
            ("Anxious", "😟"), ("Sad", "😔"), ("Tired", "😴"), ("Calm", "🧘"),
            ("Strong", "💪"), ("Unwell", "🤒"), ("Confused", "😕"), ("Irritable", "😤"),
        ]
        current_mood = st.session_state.user_mood
        cols_per_row = 4
        rows = [all_moods[i:i + cols_per_row] for i in range(0, len(all_moods), cols_per_row)]
        for row in rows:
            cols = st.columns(len(row))
            for j, (mood_name, emoji) in enumerate(row):
                with cols[j]:
                    if st.button(f"{emoji}\n\n{mood_name}", key=f"mood_{mood_name}", use_container_width=True, type="primary" if mood_name == current_mood else "secondary"):
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

    if st.session_state.user_name:
        mood_emoji_map = {
            "Happy": "😊", "Relieved": "😌", "Patient": "🏥", "Sad": "😔",
            "Neutral": "😐", "Anxious": "😟", "Irritable": "😤", "Tired": "😴",
            "Strong": "💪", "Unwell": "🤒", "Calm": "🧘", "Confused": "😕",
        }
        mood_emoji = mood_emoji_map.get(st.session_state.user_mood, "😐")
        whom_label = {"me": "Myself", "parent": "Parent", "child": "Child", "other": "Other"}.get(st.session_state.user_whom, "")
        whom_line = (
            f"<div style='font-size:11px;color:#6a9ad4;margin-top:2px;'>"
            f"Report for: {whom_label} · Age: {st.session_state.user_age}</div>"
            if whom_label else ""
        )
        st.markdown(
            f"<div style='background:rgba(255,255,255,0.08);border-radius:14px;padding:10px 14px;"
            f"margin-bottom:16px;border:1px solid rgba(255,255,255,0.12);'>"
            f"<div style='font-size:12px;color:#8aaee0;margin-bottom:2px;'>Logged in as</div>"
            f"<div style='font-size:15px;color:#fff;font-weight:600;'>"
            f"{html.escape(st.session_state.user_name)} &nbsp;{mood_emoji} {html.escape(st.session_state.user_mood)}"
            f"</div>{whom_line}</div>",
            unsafe_allow_html=True,
        )

    st.markdown(
        '<p style="font-size:9px;font-weight:800;letter-spacing:2.5px;text-transform:uppercase;'
        'color:#4a6aaa;margin:0 0 8px;display:block;">📂 Upload Documents</p>',
        unsafe_allow_html=True,
    )
    st.markdown("""
    <style>
    [data-testid="stFileUploader"] {
        position: relative !important;
    }
    [data-testid="stFileUploaderDropzone"] {
        opacity: 0 !important;
        position: relative !important;
        width: 100% !important;
        height: 110px !important;
        cursor: pointer !important;
        z-index: 10 !important;
        margin-top: -118px !important;
    }
    </style>

    <div style="
        width: 100%;
        min-height: 110px;
        border: 2px dashed rgba(74,144,217,0.6);
        border-radius: 16px;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        gap: 10px;
        background: rgba(255,255,255,0.04);
        cursor: pointer;
        margin-bottom: 8px;
    ">
        <div style="
            width: 52px;
            height: 52px;
            border-radius: 50%;
            background: rgba(74,144,217,0.20);
            border: 2px solid rgba(74,144,217,0.6);
            display: flex;
            align-items: center;
            justify-content: center;
        ">
            <svg xmlns='http://www.w3.org/2000/svg' width='22' height='22' viewBox='0 0 24 24' 
                 fill='none' stroke='white' stroke-width='2.5' 
                 stroke-linecap='round' stroke-linejoin='round'>
                <path d='M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4'/>
                <polyline points='17 8 12 3 7 8'/>
                <line x1='12' y1='3' x2='12' y2='15'/>
            </svg>
        </div>
        <div style="font-size:12px; color:rgba(255,255,255,0.5);">
            Click or drag files here
        </div>
    </div>
    """, unsafe_allow_html=True)

    uploaded_files = st.file_uploader(
        "Drop your files here",
        type=["pdf", "jpg", "jpeg", "png", "webp", "docx", "txt"],
        accept_multiple_files=True,
        label_visibility="collapsed",
        key="file_uploader",
    )

    if uploaded_files:
        for f in uploaded_files:
            already_done = f.name in st.session_state.uploaded_names
            status_icon = "✅" if already_done else "📄"
            status_text = "Processed & ready" if already_done else "Ready to process"
            status_color = "#1a8a4a" if already_done else "#2451b3"
            border_color = "#2eaa5e" if already_done else "#2451b3"
            st.markdown(
                f"<div style='background:#ffffff;border:1px solid {border_color};"
                f"border-radius:12px;padding:10px 14px;margin:6px 0;"
                f"display:flex;align-items:center;gap:10px;'>"
                f"<span style='font-size:18px;'>{status_icon}</span>"
                f"<div>"
                f"<div style='font-size:13px;font-weight:700;color:#0d2b6e;-webkit-text-fill-color:#0d2b6e;'>{html.escape(f.name)}</div>"
                f"<div style='font-size:11px;font-weight:600;color:{status_color};-webkit-text-fill-color:{status_color};'>{status_text}</div>"
                f"</div></div>",
                unsafe_allow_html=True,
            )

    if uploaded_files:
        col_l, col_btn, col_r = st.columns([0.5, 4, 0.5])
        with col_btn:
            if st.button("✨ Process All Documents", use_container_width=True):
                st.session_state.processing = True
                st.session_state.files_to_process = [{"name": f.name, "data": f.read()} for f in uploaded_files]
                st.rerun()

    if st.session_state.active_tab == "chat":
        if st.session_state.chat_history and len(st.session_state.chat_history) > 1:
            st.markdown("---")
            st.markdown('<p class="sidebar-section-label">Export Chat</p>', unsafe_allow_html=True)
            fname_base = f"medichat_{datetime.now().strftime('%Y%m%d_%H%M')}"
            txt_content = build_txt_export(
                st.session_state.chat_history,
                st.session_state.user_name,
                st.session_state.user_mood,
                st.session_state.user_conditions,
            )
            st.download_button(
                label="📄  Download as Text",
                data=txt_content.encode("utf-8"),
                file_name=f"{fname_base}.txt",
                mime="text/plain",
                use_container_width=True,
            )
            pdf_buf = build_pdf_export(
                st.session_state.chat_history,
                st.session_state.user_name,
                st.session_state.user_mood,
                st.session_state.user_conditions,
            )
            if pdf_buf:
                st.download_button(
                    label="📑  Download as PDF",
                    data=pdf_buf,
                    file_name=f"{fname_base}.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                )

        st.markdown("---")
        st.markdown('<p class="sidebar-section-label">Try asking</p>', unsafe_allow_html=True)
        for q in [
            "What does my creatinine level mean?",
            "What medications were prescribed?",
            "Is my blood sugar normal?",
            "What was the diagnosis?",
            "Which values are abnormal?",
            "What follow-up is needed?",
        ]:
            if st.button(f"💬 {q}", key=f"chip_{q}", use_container_width=True):
                st.session_state.active_tab = "chat"
                history_before = list(st.session_state.chat_history)
                st.session_state.chat_history.append({"role": "user", "content": q})
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
    st.markdown(
        "<p style='font-size:11px; color:#4a6a9a; text-align:center; line-height:1.6;'>"
        "For informational purposes only.<br/>Always consult your doctor.</p>",
        unsafe_allow_html=True,
    )


# ── DOCUMENT PROCESSING ───────────────────────────────────────
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
            tmp.write(file_info["data"])
            tmp_paths.append(tmp.name)
            original_names.append(file_info["name"])
    try:
        if st.session_state.chroma_dir and os.path.exists(st.session_state.chroma_dir):
            shutil.rmtree(st.session_state.chroma_dir, ignore_errors=True)
        st.session_state.chroma_dir = None
        # ── CHANGED: unpack 4 values now ─────────────────────────────────────
        vectorstore, chroma_dir, summaries, image_texts = ingest_multiple_files(
            tmp_paths, original_names, session_id=st.session_state.session_id
        )
        llm, retriever = build_qa_chain(vectorstore, mood=st.session_state.user_mood)
        st.session_state.llm = llm
        st.session_state.retriever = retriever
        st.session_state.chroma_dir = chroma_dir
        st.session_state.uploaded_names = original_names
        st.session_state.summaries = summaries
        st.session_state.image_texts = image_texts   # ← save image descriptions
        _set_docs_ready_message()
    except Exception as e:
        st.session_state.processing_error = str(e)
    finally:
        for path in tmp_paths:
            if os.path.exists(path):
                os.unlink(path)
    st.session_state.processing = False
    st.session_state.files_to_process = []
    st.rerun()


# ── TOP NAVBAR ────────────────────────────────────────────────
mood_emoji_map = {
    "Happy": "😊", "Relieved": "😌", "Patient": "🏥", "Sad": "😔",
    "Neutral": "😐", "Anxious": "😟", "Irritable": "😤", "Tired": "😴",
    "Strong": "💪", "Unwell": "🤒", "Calm": "🧘", "Confused": "😕",
}
user_display = (
    f"{mood_emoji_map.get(st.session_state.user_mood, '😐')} {st.session_state.user_name}"
    if st.session_state.user_name else "👤 Guest"
)

active_tab = st.session_state.active_tab

st.markdown(
    f"<div style='background:rgba(255,255,255,0.95);backdrop-filter:blur(12px);"
    f"border-bottom:1px solid rgba(74,144,217,0.15);"
    f"box-shadow:0 2px 20px rgba(13,43,110,0.08);"
    f"padding:0 32px; height:64px;"
    f"display:flex;align-items:center;justify-content:space-between;"
    f"margin-bottom:8px;'>"
    f"<div style='display:flex;align-items:center;gap:10px;'>"
    f"<span style='font-size:26px;'>🏥</span>"
    f"<span style='font-family:Playfair Display,serif;font-size:20px;font-weight:700;color:#0d2b6e;'>MediChat</span>"
    f"</div>"
    f"<div style='font-size:13px;color:#2451b3;font-weight:600;"
    f"background:rgba(74,144,217,0.08);border-radius:20px;padding:6px 14px;'>"
    f"{user_display}</div></div>",
    unsafe_allow_html=True,
)

nav_c1, nav_c2, nav_c3, nav_c4, nav_c5 = st.columns([2, 1, 1, 1, 2])
with nav_c2:
    if st.button(
        "🤖 Agents", key="nav_agents", use_container_width=True,
        type="primary" if active_tab == "agents" else "secondary",
    ):
        st.session_state.active_tab = "agents"
        st.session_state.active_agent = None
        st.session_state.agent_result = None
        st.rerun()
with nav_c3:
    if st.button(
        "💬 Chat", key="nav_chat", use_container_width=True,
        type="primary" if active_tab == "chat" else "secondary",
    ):
        st.session_state.active_tab = "chat"
        st.rerun()
with nav_c4:
    if st.button(
        "📁 Docs", key="nav_docs", use_container_width=True,
        type="primary" if active_tab == "documents" else "secondary",
    ):
        st.session_state.active_tab = "documents"
        st.rerun()


# ── Main content ──────────────────────────────────────────────
_, main_col, _ = st.columns([1, 8, 1])

with main_col:

    # ══════════════════════════════════════════════════════════
    #  AGENTS TAB
    # ══════════════════════════════════════════════════════════
    if active_tab == "agents":

        st.markdown("""
        <div class="agents-hero" style="text-align:center;padding:36px 0 28px;">
            <h1 style="font-family:'Playfair Display',serif;font-size:34px;font-weight:700;color:#0d2b6e;margin:0 0 8px;">🤖 AI Medical Agents</h1>
            <p style="font-size:15px;color:#5a7abf;margin:0;font-weight:500;">Powerful tools that analyse your documents and extract insights automatically</p>
        </div>
        """, unsafe_allow_html=True)

        orchestrator: AgentOrchestrator = st.session_state.orchestrator

        agent_icons = {
            "📋 Summarizer":           ("📋", "#4a90d9"),
            "🔬 Lab Analyzer":         ("🔬", "#7c5cbf"),
            "💊 Medication Agent":     ("💊", "#2eaa5e"),
            "📅 Appointment Reminder": ("📅", "#d97b4a"),
            "📧 Email Report":         ("📧", "#c72563"),
            "🩻 Scan Interpreter":     ("🩻", "#0a9396"),
        }
        status_style = {
            "done":    ("✅ Done",     "#2eaa5e", "#edfdf4"),
            "running": ("⏳ Running…", "#d97b4a", "#fff8f0"),
            "pending": ("🕐 Pending…", "#4a90d9", "#f0f4ff"),
            "waiting": ("⏸ Waiting…", "#c72563", "#fff0f5"),
            "":        ("— Not run",  "#aab8d4", "#f5f7ff"),
        }

        if st.session_state.retriever is None:
            st.markdown("""
            <div style="text-align:center;margin:16px auto 32px;padding:40px 32px;
                background:white;border-radius:24px;max-width:520px;
                box-shadow:0 4px 24px rgba(13,43,110,0.08);
                border:2px dashed rgba(74,144,217,0.25);">
                <div style="font-size:52px;margin-bottom:16px;">📂</div>
                <p style="color:#0d2b6e;font-size:17px;font-weight:700;margin:0 0 8px;">No documents loaded yet</p>
                <p style="color:#5a7abf;font-size:14px;margin:0;line-height:1.6;">
                    Upload your medical documents from the sidebar,<br>then come back here to run the agents.
                </p>
            </div>
            """, unsafe_allow_html=True)

        elif st.session_state.active_agent is not None:

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
                            image_texts=st.session_state.get("image_texts", {}),  # ← ADDED
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

                if agent_name == "📧 Email Report":
                    st.markdown(
                        f"<div class='agent-result-panel'>"
                        f"<div class='agent-result-header'>{agent_name}</div>"
                        f"<div class='agent-result-sub'>{descriptions.get(agent_name, '')}</div>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )

                    st.markdown("""
                    <div style="background:white;border-radius:16px;overflow:hidden;
                        box-shadow:0 4px 24px rgba(13,43,110,0.12);
                        border:0.5px solid rgba(74,144,217,0.2);">
                        <div style="background:#1a3d8f;padding:10px 18px;
                            display:flex;align-items:center;justify-content:space-between;">
                            <span style="font-size:13px;font-weight:600;color:#fff;">New message</span>
                            <div style="display:flex;gap:14px;">
                                <span style="color:rgba(255,255,255,0.7);font-size:16px;">—</span>
                                <span style="color:rgba(255,255,255,0.7);font-size:16px;">⛶</span>
                                <span style="color:rgba(255,255,255,0.7);font-size:16px;">✕</span>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

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

                else:
                    if agent_name not in ("📅 Appointment Reminder", "🩻 Scan Interpreter"):
                        st.markdown(
                            f"<div class='agent-result-panel'>"
                            f"<div class='agent-result-header'>{agent_name}</div>"
                            f"<div class='agent-result-sub'>{descriptions.get(agent_name, '')}</div>"
                            f"</div>",
                            unsafe_allow_html=True,
                        )

                    if agent_name == "📅 Appointment Reminder":
                        _render_appointment(st.session_state.agent_result)
                    elif agent_name == "🩻 Scan Interpreter":
                        _render_scan_result(st.session_state.agent_result)
                    else:
                        st.markdown(st.session_state.agent_result)

                    st.markdown("<br>", unsafe_allow_html=True)
                    c1, c2, c3 = st.columns([1, 1, 1])
                    with c1:
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
                            st.session_state.active_agent = None
                            st.session_state.agent_result = None
                            st.rerun()

        else:
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
                                        image_texts=st.session_state.get("image_texts", {}),  # ← ADDED
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

                all_done = all(
                    st.session_state.agent_statuses.get(a) in ("done", "waiting")
                    for a in orchestrator.agent_names
                )
                if all_done:
                    st.session_state.run_all_triggered = False

            if st.session_state.all_agents_results:
                st.markdown("<br>", unsafe_allow_html=True)
                result_tabs = st.tabs(list(st.session_state.all_agents_results.keys()))
                for tab, (aname, ares) in zip(result_tabs, st.session_state.all_agents_results.items()):
                    with tab:
                        st.markdown(ares)
                st.markdown("---")

            st.markdown(
                "<p style='text-align:center;color:#5a7abf;font-size:14px;margin:8px 0 20px;'>"
                "Click a card to run a single agent</p>",
                unsafe_allow_html=True,
            )

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

                with grid_cols[idx]:
                    st.markdown(
                        f"<div class='agent-grid-card {'done' if status_key == 'done' else ''}'>"
                        f"<div class='agent-card-status' style='background:{status_bg};color:{status_color};'>{status_label}</div>"
                        f"<span class='agent-card-icon'>{icon}</span>"
                        f"<div class='agent-card-name'>{agent_name}</div>"
                        f"<div class='agent-card-desc'>{agent_desc}</div>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )
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
                <p style="color:#0d2b6e;font-size:16px;font-weight:700;margin:0 0 6px;">No documents loaded</p>
                <p style="color:#5a7abf;font-size:14px;margin:0;line-height:1.6;">
                    Upload your medical documents from the sidebar to start chatting.
                </p>
            </div>
            """, unsafe_allow_html=True)

        else:
            col_input, col_mic, col_lang = st.columns([11, 1, 1])

            with col_mic:
                audio = st.audio_input("🎤", label_visibility="collapsed",
                                    key=f"audio_input_{st.session_state.get('audio_key', 0)}")

            with col_lang:
                lang_options = ["English", "Hindi", "Malayalam"]
                current_lang = st.session_state.get("user_language", "English")
                lang_icons = {"English": "🌐", "Hindi": "🇮🇳", "Malayalam": "🌴"}
                selected_lang = st.selectbox(
                    "Language",
                    options=lang_options,
                    index=lang_options.index(current_lang),
                    format_func=lambda x: lang_icons[x],
                    label_visibility="collapsed",
                    key="inline_lang_selector",
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

            with col_input:
                user_input = st.chat_input("Ask about your medical documents…")

            if audio:
                audio_bytes = audio.read()
                audio_id = hash(audio_bytes)
                if audio_id != st.session_state.get("last_audio_id"):
                    st.session_state["last_audio_id"] = audio_id
                    with st.spinner("Transcribing…"):
                        try:
                            from groq import Groq
                            groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
                            transcription = groq_client.audio.transcriptions.create(
                                file=("audio.wav", audio_bytes, "audio/wav"),
                                model="whisper-large-v3",
                                language="en",
                            )
                            st.session_state["prefill_input"] = transcription.text
                            st.session_state["voice_processed"] = False
                            st.session_state["audio_key"] = st.session_state.get("audio_key", 0) + 1
                        except Exception as e:
                            st.error(f"Transcription failed: {e}")

            if not user_input:
                prefill = st.session_state.get("prefill_input", "")
                if prefill and not st.session_state.get("voice_processed", False):
                    user_input = prefill
                    st.session_state["voice_processed"] = True
                    st.session_state.pop("prefill_input", None)

            if user_input:
                history_before = list(st.session_state.chat_history)
                st.session_state.chat_history.append({"role": "user", "content": user_input})
                with st.spinner("Reading your documents…"):
                    try:
                        answer = get_answer(
                            llm=st.session_state.llm,
                            retriever=st.session_state.retriever,
                            question=user_input,
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
        <div style="padding:32px 0 8px;">
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
                <p style="color:#0d2b6e;font-size:17px;font-weight:700;margin:0 0 8px;">No documents uploaded yet</p>
                <p style="color:#5a7abf;font-size:14px;margin:0;line-height:1.6;">
                    Use the sidebar to upload PDF, DOCX, image, or text files.<br>
                    Once processed, they'll appear here.
                </p>
            </div>
            """, unsafe_allow_html=True)
        else:
            for doc_name in st.session_state.uploaded_names:
                ext = Path(doc_name).suffix.lower()
                ext_icon = {
                    "pdf": "📄", "docx": "📝", "jpg": "🖼️", "jpeg": "🖼️",
                    "png": "🖼️", "webp": "🖼️", "txt": "📃",
                }.get(ext.lstrip("."), "📄")
                st.markdown(
                    f"<div class='doc-card'>"
                    f"<div class='doc-card-icon'>{ext_icon}</div>"
                    f"<div>"
                    f"<div class='doc-card-name'>{html.escape(doc_name)}</div>"
                    f"<div class='doc-card-status'>✅ Processed &amp; indexed</div>"
                    f"</div></div>",
                    unsafe_allow_html=True,
                )

            if st.session_state.get("summaries"):
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown(
                    '<div class="docs-section-title" style="font-size:20px;">📋 Document Summaries</div>',
                    unsafe_allow_html=True,
                )
                for doc_name, summary in st.session_state.summaries.items():
                    with st.expander(f"📄 {doc_name}", expanded=False):
                        st.markdown(summary)
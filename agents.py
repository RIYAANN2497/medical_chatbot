"""
agents.py — MediChat Multi-Agent System
========================================
Five specialised agents, each with a run() method.
An Orchestrator class routes to the correct agent based on selection.

Agents:
  1. SummarizerAgent      — structured full-document summary
  2. LabAnalyzerAgent     — flags & explains abnormal lab values
  3. MedicationAgent      — extracts medications, dosages, instructions
  4. AppointmentAgent     — extracts dates, follow-ups, reminders
  5. EmailAgent           — composes & sends report via SMTP
"""

from __future__ import annotations

import os
from typing import Any
import resend
from pathlib import Path

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI


# ── Shared LLM factory ────────────────────────────────────────────────────────

def _llm(max_tokens: int = 800) -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        temperature=0.1,
        max_output_tokens=max_tokens,
    )


def _invoke(prompt_template: str, variables: dict, max_tokens: int = 800) -> str:
    """Helper: build a simple chain and invoke it."""
    pt = PromptTemplate.from_template(prompt_template)
    chain = pt | _llm(max_tokens) | StrOutputParser()
    return chain.invoke(variables)

def _get_context(retriever, query: str, max_chars: int = 10_000) -> str:
    """Retrieve relevant chunks and join them, capped at max_chars."""
    docs = retriever.invoke(query)
    parts: list[str] = []
    total = 0
    for doc in docs:
        chunk = doc.page_content
        if total + len(chunk) > max_chars:
            break
        parts.append(chunk)
        total += len(chunk)
    return "\n\n".join(parts)
    
def _require_context(retriever, query: str, agent_name: str) -> str | None:
    """Returns context string, or None if nothing was retrieved."""
    context = _get_context(retriever, query)
    if not context.strip():
        return None
    return context





# ── 1. Summarizer Agent ───────────────────────────────────────────────────────

class SummarizerAgent:
    """Produces a structured, easy-to-read summary of all uploaded documents."""

    NAME = "📋 Summarizer"
    DESCRIPTION = "Get a full structured summary of all your uploaded documents."

    PROMPT = """
You are MediChat, a warm medical assistant. Create a clear, structured summary of the medical document(s) below.

Format your response EXACTLY like this:

**👤 Patient Details**
Name, age, gender, referring doctor if mentioned.

**🔍 Key Findings**
- Bullet each important finding in one plain-language sentence

**🔴 Abnormal Values**
Present as a markdown table:
| Test | Value | Normal Range | What it means |
|------|-------|--------------|---------------|
Only include abnormal values. Explain each in simple words.

**🩺 Diagnosis / Impression**
What the doctor concluded, in plain language.

**📋 Recommendations**
- What the doctor recommends, one bullet per point

**💊 Medications Prescribed**
List each medication with dosage and frequency in one line each. If none, write "No medications in this document."

**📄 Documents Covered**
- List document names

Keep it warm, clear, and jargon-free. Define any medical term in brackets right after using it.

Document content:
{context}

Summary:
"""

    def run(self, retriever, summaries: dict | None = None, **_) -> str:
        context = _require_context(retriever, "patient diagnosis findings recommendations medications", self.NAME)
        if context is None:
            return "⚠️ No document content found. Please upload your documents first."
        result = _invoke(self.PROMPT, {"context": context}, max_tokens=1000)
        return result


# ── 2. Lab Analyzer Agent ─────────────────────────────────────────────────────

class LabAnalyzerAgent:
    """Extracts and interprets all lab/test values, flagging abnormal results."""

    NAME = "🔬 Lab Analyzer"
    DESCRIPTION = "Analyze all lab results — flags abnormal values and explains what they mean."

    PROMPT = """
You are MediChat's Lab Analyzer Agent. Extract every test/lab result from the document content below.

For each result, provide a table row in this format:

| Test | Value | Unit | Normal Range | Status | Plain-language meaning |
|------|-------|------|--------------|--------|------------------------|

Rules:
- Status must be: ✅ Normal, ⚠️ Borderline, or 🔴 Abnormal
- For ABNORMAL values, add a brief explanation of what it means clinically in the last column
- For NORMAL values, just write "Within normal range"
- If normal range is not in the document, use standard reference ranges
- Include ALL tests — blood, urine, imaging findings, vitals, anything measurable
- After the table, add a short section: **Summary of Concerns** listing only the abnormal/borderline values and what action may be needed

Document content:
{context}

Analysis:
"""

    def run(self, retriever, **_) -> str:
        context = _require_context(
            retriever,
            "lab results blood test urine creatinine glucose hemoglobin CBC LFT KFT vitals",
            self.NAME,
        )
        if context is None:
            return "⚠️ No lab results found in the uploaded documents."
        return _invoke(self.PROMPT, {"context": context}, max_tokens=1200)


# ── 3. Medication Agent ───────────────────────────────────────────────────────

class MedicationAgent:
    """Extracts all medications with dosages, frequency, and instructions."""

    NAME = "💊 Medication Agent"
    DESCRIPTION = "Extract all prescribed medications, dosages, and instructions."

    PROMPT = """
You are MediChat's Medication Agent. Extract every medication mentioned in the document below.

Format as a clear list:

For each medication:
**[Medication Name]** — [Dosage] — [Frequency]
- Purpose: What it is typically used for (in plain language)
- Instructions: How/when to take it (from the document)
- Important notes: Any warnings, interactions, or special instructions mentioned

After listing all medications, add:

**General Reminders**
- Never stop a medication without consulting your doctor
- Always take medications at the same time each day unless told otherwise
- Keep a list of all your medications for every doctor visit

If no medications are found, say so clearly.

Document content:
{context}

Medications:
"""

    def run(self, retriever, **_) -> str:
        context = _require_context(
            retriever,
            "medication prescription tablet capsule dose mg ml twice daily morning tablet drug",
            self.NAME,
        )
        if context is None:
            return "⚠️ No document content found. Please upload your documents first."
        return _invoke(self.PROMPT, {"context": context}, max_tokens=1000)


# ── 4. Appointment / Reminder Agent ──────────────────────────────────────────

class AppointmentAgent:
    """Extracts follow-up dates, appointments, and generates reminders."""

    NAME = "📅 Appointment Reminder"
    DESCRIPTION = "Find all follow-up dates and appointments in your documents."

    PROMPT = """
You are MediChat's Appointment Agent. Extract every date, follow-up instruction, and appointment
mentioned in the document below.

Format your output EXACTLY like this — clean, spaced out, easy to read:

**📅 Upcoming Appointments**

For each appointment, format it like this:
**[Date or timeframe]**
- Type: [what kind of appointment]
- Why: [reason in one plain sentence]
- Bring: [what to bring]

---

**🧪 Tests Ordered**
- [Test name] — [when it should be done]

---

**💊 Medication Refills**
- [Medication] — [refill date or duration if mentioned]

---

**💡 Reminders**
- Save all dates in your phone calendar
- Bring previous reports to every visit
- Fast before blood tests unless told otherwise

If no dates are found, list any general follow-up recommendations instead.

Document content:
{context}

Appointments and Reminders:
"""

    def run(self, retriever, **_) -> str:
        context = _require_context(
            retriever,
            "appointment follow-up date review next visit schedule test repeat",
            self.NAME,
        )
        if context is None:
            return "⚠️ No document content found. Please upload your documents first."
        return _invoke(self.PROMPT, {"context": context}, max_tokens=800)


# ── 5. Email Agent ────────────────────────────────────────────────────────────

class EmailAgent:
    """Composes a professional medical summary email and sends it via SMTP."""

    NAME = "📧 Email Report"
    DESCRIPTION = "Send a medical summary report to a doctor or hospital via email."

    COMPOSE_PROMPT = """
You are MediChat's Email Agent. Write a professional, concise medical summary email
that someone would send to their doctor or hospital on behalf of themselves or a family member.

Sender name: {user_name}
Whom the report is for: {user_whom}

SALUTATION RULE — always address the recipient as:
"Dear Doctor," or "Dear Dr. [Name],"
NEVER use "Dear Friend", "Dear Sir/Madam", or any other greeting.

SUBJECT LINE RULE:
- If report is for Myself: "Medical Report Summary — [Sender Name]"
- If report is for My parent: "Medical Report Summary — [Sender Name]'s Parent"
- If report is for My child: "Medical Report Summary — [Sender Name]'s Child"
- If report is for Friend/Other: "Medical Report Summary — [Sender Name]'s Family Member"

WORDING RULE — adjust possessive language based on who the report is for:
- Myself       → "my results", "my report", "I have been experiencing"
- My parent    → "my parent's results", "their report", "my parent has been experiencing"
- My child     → "my child's results", "their report", "my child has been experiencing"
- Friend/Other → "my family member's results", "their report", "they have been experiencing"
Never say "my results" or "I" when the report is for someone else.

Write a professional email with:
- Subject line (first line, prefixed with "Subject: ")
- Greeting: "Dear Doctor,"
- Brief introduction (1-2 sentences — who is writing and whose report this is)
- Key findings summary (5-8 bullet points from the document)
- Any abnormal values worth flagging
- A polite closing requesting review / follow-up

Keep it under 300 words. Professional but warm tone.

Document content:
{context}

Email:
"""

    def run(
        self,
        retriever,
        user_name: str = "",
        mood: str = "Neutral",
        user_whom: str = "me",
        smtp_config: dict | None = None,
        recipient_email: str = "",
        **_,
    ) -> str:
        context = _require_context(
            retriever,
            "diagnosis findings lab results medications recommendations follow-up",
            self.NAME,
        )
        if context is None:
            return "⚠️ No document content found. Please upload your documents first."

        whom_map = {"me": "Myself", "parent": "My parent", "child": "My child", "other": "Friend/Other"}
        composed = _invoke(
            self.COMPOSE_PROMPT,
            {
                "context": context,
                "user_name": user_name or "Patient",
                "mood": mood,
                "user_whom": whom_map.get(user_whom, "Myself"),
            },
            max_tokens=600,
        )

        # Parse subject line
        lines = composed.strip().split("\n")
        subject = "Medical Report Summary — MediChat"
        body_lines = lines
        for i, line in enumerate(lines):
            clean_line = line.replace("**", "").replace("*", "").strip()
            if clean_line.lower().startswith("subject:"):
                subject = clean_line[8:].strip()
                body_lines = lines[i + 1:]
                break
        body = "\n".join(body_lines).strip()

        return (
            f"**📧 Draft Email Ready**\n\n"
            f"**Subject:** {subject}\n\n"
            f"---\n\n{body}"
        )

# ── 6. Medical Image Explainer Agent ─────────────────────────

class ImageExplainerAgent:
    """Explains medical images (X-rays, scans) in simple language."""

    NAME = "🩻 Image Explainer"
    DESCRIPTION = "Upload an X-ray or scan and get a plain-language explanation."

    def run(self, retriever, summaries: dict | None = None, **_) -> str:
        # Use the stored summary/context from the image if it was uploaded
        if summaries:
            image_summaries = {
                name: summary for name, summary in summaries.items()
                if Path(name).suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
            }
            if not image_summaries:
                return "⚠️ No medical images found. Please upload an X-ray or scan (JPG, PNG, WEBP) and process it first."

            results = []
            for img_name, img_text in image_summaries.items():
                prompt = f"""You are MediChat's Image Explainer. A patient has uploaded a medical image called "{img_name}".

Below is the extracted content from that image. Your job is to explain it in two versions:

**🧑 Patient Version**
Write 3-4 casual, warm sentences a non-medical person can understand. No jargon. Use simple analogies. Be reassuring but honest. Don't diagnose — just explain what you see described.

**👨‍⚕️ Doctor Version**
Write a brief clinical summary in 3-4 sentences using proper medical terminology. Include findings, any notable observations, and suggest what follow-up might be relevant.

**⚠️ Important Disclaimer**
Always end with: "This is an AI-assisted explanation only. Always consult a qualified radiologist or doctor for an accurate diagnosis."

Image content:
{img_text}
"""
                result = _invoke(prompt, {}, max_tokens=800)
                results.append(f"### {img_name}\n\n{result}")

            return "\n\n---\n\n".join(results)

        return "⚠️ No image content found. Please upload and process a medical image first."


# ── Orchestrator ──────────────────────────────────────────────────────────────

class AgentOrchestrator:
    """
    Routes user selection to the correct agent and returns the result.
    All agents share the same retriever from the vectorstore.
    """

    AGENTS: list[type] = [
        SummarizerAgent,
        LabAnalyzerAgent,
        MedicationAgent,
        AppointmentAgent,
        EmailAgent,
        ImageExplainerAgent,  # ← add this
    ]

    def __init__(self):
        self._instances: dict[str, Any] = {
            cls.NAME: cls() for cls in self.AGENTS
        }

    @property
    def agent_names(self) -> list[str]:
        return [cls.NAME for cls in self.AGENTS]

    @property
    def agent_descriptions(self) -> dict[str, str]:
        return {cls.NAME: cls.DESCRIPTION for cls in self.AGENTS}

    def run(self, agent_name: str, retriever, **kwargs) -> str:
        """
        Run the selected agent.

        kwargs are passed through to the agent's run() method —
        include user_name, mood, user_whom, smtp_config, recipient_email, summaries etc.
        """
        agent = self._instances.get(agent_name)
        if agent is None:
            return f"❌ Unknown agent: {agent_name}"
        return agent.run(retriever=retriever, **kwargs)
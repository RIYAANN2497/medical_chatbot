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
import smtplib
import textwrap
from collections import defaultdict
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Any

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq


# ── Shared LLM factory ────────────────────────────────────────────────────────

def _llm(max_tokens: int = 800) -> ChatGroq:
    return ChatGroq(
        model_name="llama-3.3-70b-versatile",
        temperature=0.1,
        max_tokens=max_tokens,
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


# ── 1. Summarizer Agent ───────────────────────────────────────────────────────

class SummarizerAgent:
    """Produces a structured, easy-to-read summary of all uploaded documents."""

    NAME = "📋 Summarizer"
    DESCRIPTION = "Get a full structured summary of all your uploaded documents."

    PROMPT = """
You are MediChat's Summarizer Agent. Create a clear, structured summary of the medical document(s) below.

Format your summary EXACTLY like this:

**Patient / Subject**
- Name, age, gender if mentioned

**Key Findings**
- Bullet each important finding in plain language

**Abnormal Values**
- List ONLY values outside normal range; explain what each means simply

**Diagnosis / Impression**
- What the doctor concluded (if present)

**Recommendations / Next Steps**
- What the doctor recommends

**Documents Covered**
- List document names

Use plain English. Define any medical term in brackets immediately after using it.
Keep each bullet to one sentence.

Document content:
{context}

Summary:
"""

    def run(self, retriever, summaries: dict | None = None, **_) -> str:
        context = _get_context(retriever, "patient diagnosis findings recommendations medications")
        if not context.strip():
            return "⚠️ No document content found. Please upload and process your documents first."
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
        context = _get_context(
            retriever,
            "lab results blood test urine creatinine glucose hemoglobin CBC LFT KFT vitals",
        )
        if not context.strip():
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
        context = _get_context(
            retriever,
            "medication prescription tablet capsule dose mg ml twice daily morning tablet drug",
        )
        if not context.strip():
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

Format your output as:

**Upcoming / Recommended Appointments**
For each one:
📅 [Date or timeframe] — [Type of appointment / test]
   → Reason: [why this is needed, in plain language]
   → Bring: [what to bring if mentioned — reports, empty stomach, etc.]

**Tests / Investigations Ordered**
List any tests the doctor has ordered that haven't been done yet.

**Repeat Medications / Refills**
Note any medication refill dates or pharmacy instructions.

**Reminder Tips**
- Save these dates in your phone's calendar
- Bring all previous reports to follow-up visits
- Fast for blood tests unless told otherwise

If no dates are found, state that no specific appointments were mentioned
and list any general follow-up recommendations instead.

Document content:
{context}

Appointments and Reminders:
"""

    def run(self, retriever, **_) -> str:
        context = _get_context(
            retriever,
            "appointment follow-up date review next visit schedule test repeat",
        )
        if not context.strip():
            return "⚠️ No document content found. Please upload your documents first."
        return _invoke(self.PROMPT, {"context": context}, max_tokens=800)


# ── 5. Email Agent ────────────────────────────────────────────────────────────

class EmailAgent:
    """Composes a professional medical summary email and sends it via SMTP."""

    NAME = "📧 Email Report"
    DESCRIPTION = "Send a medical summary report to a doctor or hospital via email."

    COMPOSE_PROMPT = """
You are MediChat's Email Agent. Write a professional, concise medical summary email
that a patient would send to their doctor or hospital.

Patient name: {user_name}
Mood / context: {mood}
Whom the report is for: {user_whom}

Write a professional email with:
- Subject line (first line, prefixed with "Subject: ")
- Polite greeting
- Brief introduction (1-2 sentences about why they're writing)
- Key findings summary (5-8 bullet points from the document)
- Any abnormal values worth flagging
- A polite closing requesting review / follow-up

Keep it under 300 words. Professional but human tone.

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
        context = _get_context(
            retriever,
            "diagnosis findings lab results medications recommendations follow-up",
        )
        if not context.strip():
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

        # Parse subject line from composed email
        lines = composed.strip().split("\n")
        subject = "Medical Report Summary — MediChat"
        body_lines = lines
        for i, line in enumerate(lines):
            if line.lower().startswith("subject:"):
                subject = line[8:].strip()
                body_lines = lines[i + 1 :]
                break
        body = "\n".join(body_lines).strip()

        # If no SMTP config or recipient, just return the composed draft
        if not smtp_config or not recipient_email:
            return (
                f"**📧 Draft Email Ready**\n\n"
                f"**To:** {recipient_email or '(no recipient set)'}\n"
                f"**Subject:** {subject}\n\n"
                f"---\n\n{body}\n\n"
                f"---\n\n"
                f"_To send this email, configure SMTP in the sidebar._"
            )

        # Send via SMTP
        try:
            msg = MIMEMultipart("alternative")
            msg["Subject"] = subject
            msg["From"] = smtp_config["sender_email"]
            msg["To"] = recipient_email

            footer = (
                "\n\n---\nThis report was generated by MediChat. "
                "For informational purposes only. Always consult your doctor."
            )
            msg.attach(MIMEText(body + footer, "plain"))

            with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
                server.login(smtp_config["sender_email"], smtp_config["app_password"])
                server.sendmail(smtp_config["sender_email"], recipient_email, msg.as_string())

            return (
                f"✅ **Email sent successfully!**\n\n"
                f"**To:** {recipient_email}\n"
                f"**Subject:** {subject}\n\n"
                f"---\n\n{body}"
            )
        except Exception as e:
            return (
                f"❌ **Failed to send email:** {e}\n\n"
                f"**Draft (copy this manually):**\n\n"
                f"**Subject:** {subject}\n\n{body}"
            )


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
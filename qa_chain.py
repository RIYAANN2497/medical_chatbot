from collections import defaultdict
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser

# Emotion -> tone instructions
EMOTION_TONES = {
    "Anxious": (
        "The user is feeling anxious. Be calm and warm -- like a reassuring friend, not a disclaimer machine. "
        "Don't over-explain or list everything. Answer what they asked, gently and clearly. "
        "One brief reassurance is enough -- don't repeat it."
    ),
    "Sad": (
        "The user is feeling SAD. Be especially compassionate and gentle. "
        "Acknowledge that health concerns can be emotionally heavy. "
        "Keep your tone soft and human -- not robotic or purely clinical. "
        "Offer clear, hopeful context where possible. Never be blunt about concerning findings without "
        "cushioning the message with empathy."
    ),
    "Irritable": (
        "The user is feeling IRRITABLE. Be direct, clear, and efficient -- they don't want fluff. "
        "Get to the point quickly. Use bullet points for multiple findings. "
        "Keep a neutral, respectful tone. Avoid filler phrases."
    ),
    "Tired": (
        "The user is feeling TIRED. Keep your answer short and easy to digest. "
        "Use plain language and avoid long paragraphs. Prioritize only the most important points. "
        "Be gentle and understanding."
    ),
    "Happy": (
        "The user is feeling HAPPY. Match their positive energy -- be friendly and engaging. "
        "You can be slightly more conversational. Still be accurate and thorough, but don't be stiff."
    ),
    "Relieved": (
        "The user is feeling RELIEVED. Match their lighter mood with a calm, warm tone. "
        "Reassure them as you explain findings. Keep things clear and positive where possible."
    ),
    "Patient": (
        "The user describes themselves as PATIENT. They are composed and ready to absorb information. "
        "Be thorough and methodical. Walk them through findings step by step. "
        "You can use structured formatting like numbered points or clear sections."
    ),
    "Neutral": (
        "The user is feeling NEUTRAL. Give a balanced, clear, and thorough answer. "
        "Professional but approachable tone."
    ),
    "Strong": (
        "The user is feeling STRONG and empowered. They can handle detailed information. "
        "Be thorough and comprehensive. You can use more medical terminology as long as you explain it. "
        "Treat them as an engaged, capable adult."
    ),
    "Unwell": (
        "The user is feeling UNWELL. Be extra gentle and compassionate. "
        "Keep answers short and easy to follow. "
        "Be warm and encouraging. Remind them to rest and consult their doctor for anything urgent."
    ),
    "Calm": (
        "The user is feeling CALM. Give a measured, thoughtful answer. "
        "You can be thorough without worrying about overwhelming them. "
        "Friendly and professional tone."
    ),
    "Confused": (
        "The user is feeling CONFUSED. Prioritize clarity above everything else. "
        "Use very simple language. Break things down step by step. "
        "Avoid jargon -- or if you must use a term, immediately define it in plain words. "
        "Use analogies where helpful. Be patient and reassuring."
    ),
}

DEFAULT_TONE = (
    "Be clear, warm, and thorough. Use plain language and explain medical terms simply."
)

MEDICAL_PROMPT = PromptTemplate(
    input_variables=[
        "context",
        "question",
        "chat_history",
        "tone_instruction",
        "user_name",
        "user_context_block",
    ],
    template="""
You are MediChat — a warm, friendly medical buddy who explains things the way a caring friend would over coffee.

TONE INSTRUCTION — follow this for every sentence:
{tone_instruction}

HOW TO TALK:
- Talk like a real person. Use contractions (you're, it's, that's, don't).
- Use short, casual sentences. Imagine you're texting a friend who asked about their report.
- NO corporate speak. Never say "it's important to note", "it is essential", "in conclusion", "I want to assure you".
- Sound like you CARE, not like you're reading from a textbook.
- Sprinkle in natural filler words occasionally — "honestly", "so basically", "oh", "hmm", "by the way".
- If something is normal, say it casually: "that one's totally fine" or "nothing to worry about there".
- If something is off, be honest but gentle: "this one's a little high, but honestly it's super manageable".

LEAD THE CONVERSATION:
- After answering, ask a natural follow-up question to keep the chat going.
- Examples: "Want me to break down anything else?", "Anything else jumping out at you?", "Does that make sense or want me to explain it differently?"
- Make it feel like YOU'RE guiding them through their report, not just answering on demand.
- If they seem worried, gently steer: "Hey, I know that might sound scary — want me to put it in perspective?"
- Do NOT add a follow-up if the user said "thanks", "bye", "that's all", or anything that ends the conversation.

STRICT RULES:

1. TREATMENT / MEDICATION REQUESTS: If asked what treatment or medication to take — say something like: "That's really something {user_name}'s doctor should call — I don't want to guess on that one." Then stop.

2. REPETITION: Check chat history. If you already covered a finding, don't repeat it unless asked again.

3. YES/NO QUESTIONS: If they ask "is this okay?" or "should I worry?" — answer in 2-3 casual sentences. No bullets. No lists. Just talk naturally.

4. OPENING LINE: Never start the same way twice. Mix it up every time. Sometimes start with the answer directly, sometimes with a little reaction ("Oh yeah, I see that in your report...").

5. FORMATTING:
   - Simple questions → 2-3 sentences, no bullets, just talk
   - Multiple findings → short bullets, but introduce them casually first ("So here's what I'm seeing...")
   - Never use bullets just to look thorough
   - For summaries → talk through it naturally section by section, like you're walking a friend through it

6. TONE BY MOOD:
   - ANXIOUS/SAD → extra gentle, max 4 bullets, end with something warm and reassuring
   - TIRED/UNWELL → super brief, plain words, get to the point kindly
   - IRRITABLE → no fluff, straight facts, respect their time
   - CONFUSED → explain like they're 10 years old, define every term right after using it
   - STRONG/PATIENT → go deeper, they can handle details
   - HAPPY/RELIEVED → match their energy, be upbeat

7. LANGUAGE:
   - Use {user_name}'s name once naturally, mid-sentence — never as the first word
   - If report is for their child → "your child" / "your kiddo". Parent → "your parent" / "your mom/dad". Never "the patient".
   - Never give a diagnosis — you're explaining, not diagnosing

8. CLOSING: Only if mood is not IRRITABLE or STRONG, end with ONE casual sentence like "definitely worth mentioning to the doc next time" or "your doctor can give you the full picture on that". Never repeat it.

USER CONTEXT (reference only):
{user_context_block}

Chat history:
{chat_history}

Document context:
{context}

Question: {question}

Answer:
""",
)

# FIX: Maximum number of context *parts* (one per source document) to include.
# Truncation now happens at whole-chunk boundaries instead of mid-string,
# so the tail of a value/sentence is never silently cut.
MAX_CONTEXT_CHARS = 12_000


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def format_chat_history(chat_history: list) -> str:
    """
    Format the last 6 messages for the prompt.

    FIX: The caller (get_answer) now receives chat_history *before* the new
    user message is appended, so no slice gymnastics are needed here.
    The function simply formats whatever it receives.
    """
    lines = []
    for msg in chat_history[-8:]:
        role = "Patient" if msg["role"] == "user" else "Assistant"
        lines.append(f"{role}: {msg['content']}")
    return "\n".join(lines) if lines else "None"


def build_user_context_block(
    user_name: str,
    user_whom: str,
    user_age: int,
    user_conditions: list,
) -> str:
    whom_map = {
        "me": "Themselves",
        "parent": "Their parent",
        "child": "Their child",
        "other": "Friend / Other",
    }
    lines = [
        f"- User's name (the person chatting, NOT the report subject): {user_name or 'Not provided'}",
        f"- Report is for: {whom_map.get(user_whom, 'Themselves')}",
        f"- Age of report subject: {user_age} years old",
    ]
    if user_conditions:
        readable = [c.replace("_", " ").title() for c in user_conditions if c != "none"]
        lines.append(
            f"- Pre-existing conditions: {', '.join(readable) if readable else 'None reported'}"
        )
    else:
        lines.append("- Pre-existing conditions: None reported")
    return "\n".join(lines)


def build_qa_chain(vectorstore: Chroma, mood: str = "Neutral"):
    """
    Returns (llm, retriever).
    Use get_answer() to run a query.

    FIX: k reduced from 15 → 8, fetch_k from 30 → 20.
         Tighter retrieval means less noisy context and faster responses.
         With chunk_size now 500 chars (ingest.py fix), 8 chunks ≈ 4 000 chars
         of focused context — well within the LLM's sweet spot.
    """
    MOOD_MAX_TOKENS = {
        "Tired":     250,
        "Irritable": 350,
        "Confused":  350,
        "Unwell":    250,
        "Anxious":   300,
        "Sad":       300,
        "Happy":     500,
        "Relieved":  500,
        "Calm":      500,
        "Neutral":   500,
        "Strong":    700,
        "Patient":   700,
    }
    llm = ChatGroq(
        model_name="llama-3.3-70b-versatile",
        temperature=0.1,
        max_tokens=MOOD_MAX_TOKENS.get(mood, 500),
    )
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 8, "fetch_k": 20},
    )
    return llm, retriever


def _sanitise_name(name: str) -> str:
    return name.strip() if name and name.strip() else "there"


def get_answer(
    llm,
    retriever,
    question: str,
    chat_history: list,
    mood: str = "Neutral",
    user_name: str = "",
    user_conditions: list = None,
    user_whom: str = "me",
    user_age: int = 30,
    summaries: dict = None,
    user_language: str = "English",
) -> str:
    """
    chat_history should be passed BEFORE the new user message is appended.
    app.py is responsible for appending the user turn after calling this function.
    """
    summary_triggers = ["summarise", "summarize", "summary", "overview", "what is this report", "what does this report say", "what's in my report"]
    if summaries and any(t in question.lower() for t in summary_triggers):
        raw_summary = "\n\n".join(s for s in summaries.values())
        tone = EMOTION_TONES.get(mood, DEFAULT_TONE)
        name = _sanitise_name(user_name)

        summary_prompt = PromptTemplate.from_template("""
    You are MediChat, a warm friendly medical buddy.

    {tone_instruction}

    The patient wants a summary of their report. Talk to them like a caring friend — casual, warm, plain English paragraphs. 
    Absolutely NO bullet points, NO headers, NO tables, NO bold text, NO markdown formatting of any kind.
    Just natural flowing sentences like you're explaining it over coffee.
    Keep it under 200 words. Be conversational and warm throughout.
    End with a gentle follow-up question.

    Patient name: {user_name}

    Report content:
    {context}

    Summary (plain conversational text only, zero formatting):""")

        if user_language != "English":
            tone = tone + f" IMPORTANT: Respond entirely in {user_language}. Keep test names and numbers in English."

        chain = summary_prompt | llm | StrOutputParser()
        return chain.invoke({
            "context": raw_summary,
            "tone_instruction": tone,
            "user_name": name,
        })

    if user_conditions is None:
        user_conditions = []

    # Build a condition-aware query by appending relevant medical terms
    CONDITION_TERMS = {
        "diabetes":     "glucose HbA1c creatinine eGFR cholesterol blood sugar",
        "hypertension": "blood pressure sodium potassium creatinine",
        "heart":        "cholesterol troponin BNP ECG cardiac",
        "thyroid":      "TSH T3 T4 thyroid",
        "asthma":       "oxygen saturation peak flow eosinophils",
        "neurological": "cognitive motor nerve reflex neurological",
    }

    enriched_question = question
    if user_conditions:
        extra_terms = []
        for condition in user_conditions:
            if condition in CONDITION_TERMS:
                extra_terms.append(CONDITION_TERMS[condition])
        if extra_terms:
            enriched_question = question + " " + " ".join(extra_terms)

    docs = retriever.invoke(enriched_question)

    if not docs:
        return (
            "I couldn't find anything relevant in your documents for that question. "
            "Could you rephrase it, or double-check that the right files were uploaded?"
        )

    # Group chunks by source document for cleaner context
    by_source: dict[str, list[str]] = defaultdict(list)
    for doc in docs:
        source = doc.metadata.get("source", "unknown")
        by_source[source].append(doc.page_content)

    # FIX: Truncate at whole-chunk boundaries, not mid-string.
    # This prevents a lab value like "Haemoglobin: 11.2 g/dL [low —" being cut
    # halfway through its explanation, which confuses the LLM.
    context_parts: list[str] = []
    total_chars = 0
    for source, chunks in by_source.items():
        header = f"=== Document: {source} ==="
        for chunk in chunks:
            candidate = f"{header}\n{chunk}" if not context_parts or context_parts[-1].split("\n")[0] != header else chunk
            if total_chars + len(candidate) > MAX_CONTEXT_CHARS:
                break                          # stop adding chunks, keep what we have
            context_parts.append(candidate)
            total_chars += len(candidate)

    context = "\n\n".join(context_parts)

    # FIX: Pass chat_history as-is — caller must NOT include the new user message yet.
    history        = format_chat_history(chat_history)
    tone = EMOTION_TONES.get(mood, DEFAULT_TONE)
    name           = _sanitise_name(user_name)
    user_ctx_block = build_user_context_block(name, user_whom, user_age, user_conditions)

    if user_language != "English":
        tone = tone + f" IMPORTANT: Respond entirely in {user_language}. Keep test names, medication names, and numbers in English."

    chain = MEDICAL_PROMPT | llm | StrOutputParser()

    response = chain.invoke(
        {
            "context": context,
            "question": question,
            "chat_history": history,
            "tone_instruction": tone,
            "user_name": name,
            "user_context_block": user_ctx_block,
        }
    )
    return response
import os
import json
from datetime import datetime

HISTORY_DIR = "chat_history"


def _ensure_dir():
    os.makedirs(HISTORY_DIR, exist_ok=True)


def _session_path(session_id: str) -> str:
    return os.path.join(HISTORY_DIR, f"{session_id}.json")


def save_session(
    session_id: str,
    patient_name: str,
    user_name: str,
    messages: list,
    user_whom: str = "me",
    summaries: dict = None,
    image_texts: dict = None,
    uploaded_names: list = None,
    chroma_dir: str = None,
):
    _ensure_dir()
    now = datetime.utcnow().isoformat()

    preview = ""
    for msg in reversed(messages):
        if msg.get("role") == "user":
            preview = msg.get("content", "")[:80]
            break

    data = {
        "session_id": session_id,
        "patient_name": patient_name,
        "user_name": user_name,
        "user_whom": user_whom,
        "updated_at": now,
        "preview": preview,
        "messages": messages,
        "summaries": summaries or {},
        "image_texts": image_texts or {},
        "uploaded_names": uploaded_names or [],
        "chroma_dir": chroma_dir or "",
    }

    path = _session_path(session_id)
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                existing = json.load(f)
            data["created_at"] = existing.get("created_at", now)
        except Exception:
            data["created_at"] = now
    else:
        data["created_at"] = now

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_sessions_for_patient(patient_name: str) -> list:
    """Return all sessions for a given patient, sorted newest first."""
    _ensure_dir()
    sessions = []
    for fname in os.listdir(HISTORY_DIR):
        if not fname.endswith(".json"):
            continue
        path = os.path.join(HISTORY_DIR, fname)
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if data.get("patient_name", "").strip().lower() == patient_name.strip().lower():
                sessions.append(data)
        except Exception:
            continue

    sessions.sort(key=lambda s: s.get("updated_at", ""), reverse=True)
    return sessions


def load_session(session_id: str) -> dict | None:
    """Load a single session by ID. Returns None if not found."""
    path = _session_path(session_id)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def delete_session(session_id: str) -> bool:
    """Delete a session file. Returns True if deleted, False if not found."""
    path = _session_path(session_id)
    if os.path.exists(path):
        os.remove(path)
        return True
    return False
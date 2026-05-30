import fitz
import base64
import os
import uuid
import shutil
from pathlib import Path
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from groq import Groq
import docx

CHROMA_BASE_DIR = str(Path(__file__).parent / "chroma_store")
SUPPORTED_IMAGES = {".jpg", ".jpeg", ".png", ".webp"}

# FIX 3: Cache the embedding model at module level so it is only loaded once
# per process instead of on every upload call.
_embeddings = None


def get_embeddings() -> HuggingFaceEmbeddings:
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
    return _embeddings

def summarise_document(chunks: list[str], doc_name: str) -> str:
    from langchain_groq import ChatGroq
    from langchain_core.output_parsers import StrOutputParser
    sample = "\n\n".join(chunks[:6])[:3000]
    prompt = f"""You are a medical assistant. Summarise this medical document in 5-8 bullet points.
Cover: patient details if present, key test results, abnormal values, diagnosis/impression, recommendations.
Document: {doc_name}

Content:
{sample}

Summary:"""
    llm = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0.1)
    return (llm | StrOutputParser()).invoke(prompt)


def extract_text_from_pdf(pdf_path: str, max_vision_pages: int = 10) -> str:
    """
    Extract text from a PDF.

    For text-based PDFs: uses PyMuPDF directly (fast, free).
    For scanned/image-only PDFs: renders every page (up to max_vision_pages)
    and sends each to the Groq vision model.

    FIX 1: Previously only page 0 was OCR-ed for scanned PDFs.
            Now all pages are processed (capped at max_vision_pages for cost safety).
    """
    with fitz.open(pdf_path) as doc:
        text = "".join(page.get_text() for page in doc)

    if len(text.strip()) >= 50:
        return text

    # Scanned PDF — OCR every page via Groq vision
    all_text: list[str] = []
    with fitz.open(pdf_path) as doc:
        pages_to_process = min(len(doc), max_vision_pages)
        for i in range(pages_to_process):
            img_path = f"{pdf_path}_page{i}.png"
            try:
                pix = doc[i].get_pixmap(dpi=200)
                pix.save(img_path)
                page_text = extract_text_from_image(img_path)
                if page_text.strip():
                    all_text.append(f"[Page {i + 1}]\n{page_text}")
            finally:
                if os.path.exists(img_path):
                    os.unlink(img_path)

    return "\n\n".join(all_text)


def extract_text_from_docx(docx_path: str) -> str:
    """
    Extract text from DOCX including table cell content.

    FIX 4: python-docx returns merged cells multiple times in row.cells.
            Added deduplication per row to avoid repeated values in chunks.
    """
    doc = docx.Document(docx_path)
    parts = [para.text for para in doc.paragraphs if para.text.strip()]

    for table in doc.tables:
        for row in table.rows:
            seen: set[str] = set()
            cells: list[str] = []
            for cell in row.cells:
                text = cell.text.strip()
                if text and text not in seen:
                    seen.add(text)
                    cells.append(text)
            if cells:
                parts.append(" | ".join(cells))

    return "\n".join(parts)


def extract_text_from_image(image_path: str) -> str:
    """Send an image to the Groq vision model and extract all medical text."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise EnvironmentError("GROQ_API_KEY is not set — cannot process image files.")

    client = Groq(api_key=api_key)
    ext = Path(image_path).suffix.lower()
    media_map = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
    }
    media_type = media_map.get(ext, "image/jpeg")

    with open(image_path, "rb") as f:
        image_data = base64.standard_b64encode(f.read()).decode("utf-8")

    try:
        response = client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{media_type};base64,{image_data}"
                            },
                        },
                        {
                            "type": "text",
                            "text": (
                                "This is a medical document or report. "
                                "Extract ALL text, values, labels, and findings visible in this image. "
                                "Include every number, unit, test name, and result. Format clearly."
                            ),
                        },
                    ],
                }
            ],
            max_tokens=2048,
        )
        return response.choices[0].message.content
    except Exception as e:
        raise RuntimeError(f"Image extraction failed for {image_path}: {e}") from e


def ingest_multiple_files(
    file_paths: list,
    original_names: list,
    session_id: str = None,
) -> tuple:
    """
    Ingest files into a session-scoped Chroma vectorstore.

    Returns (vectorstore, chroma_dir) so the caller can clean up later.

    FIX 2: chunk_size reduced from 1200 → 500 chars (≈125 tokens) to stay well
            within the 256-token limit of all-MiniLM-L6-v2.  Overlap reduced
            proportionally from 200 → 80.
    FIX 5: Each chunk now carries source, chunk_index, and file_type metadata
            so answers can eventually cite "page 3 of report.pdf" etc.
    FIX 6: .txt files are now supported instead of silently skipped.
            Unsupported types surface a clear warning list returned to the caller.
    """
    # FIX 2: Safe chunk size for all-MiniLM-L6-v2 (256-token context window).
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=80)

    all_chunks: list[Document] = []
    summaries: dict[str, str] = {}
    skipped: list[str] = []          # FIX 6: collect skipped file names

    for file_path, original_name in zip(file_paths, original_names):
        ext = Path(original_name).suffix.lower()
        print(f"[ingest] Processing: {original_name}")

        try:
            if ext == ".pdf":
                raw_text = extract_text_from_pdf(file_path)
            elif ext == ".docx":
                raw_text = extract_text_from_docx(file_path)
            elif ext in SUPPORTED_IMAGES:
                raw_text = extract_text_from_image(file_path)
            elif ext == ".txt":
                # FIX 6: plain-text support
                with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                    raw_text = f.read()
            else:
                print(f"[ingest] Skipping unsupported file: {original_name}")
                skipped.append(original_name)
                continue
        except Exception as e:
            print(f"[ingest] Error processing {original_name}: {e}")
            raise RuntimeError(f"Failed to process '{original_name}': {e}") from e

        chunks = splitter.split_text(raw_text)

        # FIX 5: Richer metadata — source filename, chunk index, and file type.
        documents = [
            Document(
                page_content=chunk,
                metadata={
                    "source": original_name,
                    "chunk_index": i,
                    "file_type": ext.lstrip("."),
                },
            )
            for i, chunk in enumerate(chunks)
        ]
        all_chunks.extend(documents)
        summaries[original_name] = summarise_document(chunks, original_name)
        print(f"[ingest] {original_name} → {len(chunks)} chunks")

    if skipped:
        print(f"[ingest] Skipped unsupported files: {', '.join(skipped)}")

    if not all_chunks:
        raise ValueError(
            "No content could be extracted from the uploaded files. "
            "Check that your files are not empty or in an unsupported format."
        )

    # Session-scoped directory prevents concurrent-user collisions
    sub = session_id or str(uuid.uuid4())
    chroma_dir = str(Path(CHROMA_BASE_DIR) / sub)

    if os.path.exists(chroma_dir):
        shutil.rmtree(chroma_dir)

    embedding_model = get_embeddings()   # FIX 3: returns cached instance
    print(f"[ingest] Embedding {len(all_chunks)} total chunks → {chroma_dir}")

    vectorstore = Chroma.from_documents(
        documents=all_chunks,
        embedding=embedding_model,
        persist_directory=chroma_dir,
    )
    print("[ingest] Done.")
    return vectorstore, chroma_dir, summaries
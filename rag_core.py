from __future__ import annotations

import hashlib
import os
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import chromadb
import pymupdf as fitz
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

DATA_DIR = Path(__file__).parent / "data"
PDF_DIR = DATA_DIR / "pdfs"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 800 
CHUNK_OVERLAP = 150


@dataclass
class Chunk:
    id: str
    text: str
    doc_id: str
    filename: str
    page: int          # 1-indexed
    score: float = 0.0


# Load
def extract_pages(pdf_path: Path, ocr: bool = True) -> list[tuple[int, str]]:
    # Return [(page_number, text)]. Falls back to OCR for pages with no text layer.
    pages = []
    with fitz.open(pdf_path) as doc:
        for i, page in enumerate(doc):
            text = page.get_text("text")
            if ocr and len(text.strip()) < 20:
                text = _ocr_page(page) or text
            pages.append((i + 1, text))
    return _clean_pages(pages)


def _ocr_page(page) -> str:
    # OCR a scanned page with Tesseract (optional: needs `tesseract` installed).
    try:
        tp = page.get_textpage_ocr(dpi=200, full=True)
        return page.get_text("text", textpage=tp)
    except Exception:
        return ""

# Clean
def _clean_pages(pages: list[tuple[int, str]]) -> list[tuple[int, str]]:
    # Strip repeated headers/footers and page numbers, fix hyphenation and whitespace.
    def norm(line: str) -> str:
        return re.sub(r"\d+", "#", line.strip().lower())

    # Lines (digits normalised) that recur on many pages are headers/footers.
    counts = Counter()
    for _, text in pages:
        counts.update({norm(l) for l in text.splitlines() if l.strip()})
    threshold = max(3, int(len(pages) * 0.5))
    boilerplate = {l for l, c in counts.items() if c >= threshold and len(l) < 100}

    cleaned = []
    for num, text in pages:
        lines = [
            l for l in text.splitlines()
            if l.strip()
            and norm(l) not in boilerplate
            and not re.fullmatch(r"\s*(page\s*)?\d+(\s*(of|/)\s*\d+)?\s*", l, re.I)
        ]
        t = "\n".join(lines)
        t = re.sub(r"(\w)-\n(\w)", r"\1\2", t)        # re-join hyphenated words
        t = re.sub(r"(?<![.!?:\n])\n(?!\n)", " ", t)   # unwrap hard-wrapped lines
        t = re.sub(r"[ \t]+", " ", t).strip()
        cleaned.append((num, t))
    return cleaned


# Chunk
def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    # Split on sentence boundaries into ~`size`-char chunks with `overlap` chars carried over.
    sentences = re.split(r"(?<=[.!?])\s+|\n+", text)
    chunks, current = [], ""
    for sent in sentences:
        sent = sent.strip()
        if not sent:
            continue
        while len(sent) > size:                      # very long "sentence": hard split
            chunks.append(sent[:size])
            sent = sent[size - overlap:]
        if len(current) + len(sent) + 1 > size and current:
            chunks.append(current)
            tail = current[-overlap:]
            current = tail[tail.find(" ") + 1:] if " " in tail else tail
        current = f"{current} {sent}".strip()
    if current:
        chunks.append(current)
    return chunks


# Embed + store
class RAGIndex:
    def __init__(self, persist_dir: Path = DATA_DIR / "chroma"):
        PDF_DIR.mkdir(parents=True, exist_ok=True)
        self.embedder = SentenceTransformer(EMBED_MODEL)
        self.client = chromadb.PersistentClient(path=str(persist_dir))
        self.col = self.client.get_or_create_collection(
            "chunks", metadata={"hnsw:space": "cosine"}
        )
        self._build_bm25()

    def _embed(self, texts: list[str]) -> list[list[float]]:
        return self.embedder.encode(texts, normalize_embeddings=True).tolist()

    def _build_bm25(self):
        data = self.col.get(include=["documents", "metadatas"])
        self._all = [
            Chunk(id=i, text=d, doc_id=m["doc_id"], filename=m["filename"], page=m["page"])
            for i, d, m in zip(data["ids"], data["documents"], data["metadatas"])
        ]
        self._bm25 = BM25Okapi([_tokenize(c.text) for c in self._all]) if self._all else None

    def documents(self) -> dict[str, str]:
        # doc_id -> filename for every indexed PDF.
        return {c.doc_id: c.filename for c in self._all}

    def add_pdf(self, filename: str, data: bytes) -> tuple[str, int]:
        # Index a PDF. Returns (doc_id, n_chunks). Re-uploading the same file is a no-op.
        doc_id = hashlib.sha1(data).hexdigest()[:16]
        if doc_id in self.documents():
            return doc_id, 0
        path = PDF_DIR / f"{doc_id}.pdf"
        path.write_bytes(data)

        ids, texts, metas = [], [], []
        for page_num, page_text in extract_pages(path):
            for j, chunk in enumerate(chunk_text(page_text)):
                ids.append(f"{doc_id}-p{page_num}-{j}")
                texts.append(chunk)
                metas.append({"doc_id": doc_id, "filename": filename, "page": page_num})
        if not texts:
            raise ValueError("No text found in PDF (scanned PDF without Tesseract installed?)")

        for s in range(0, len(texts), 256):           # batch to keep memory flat
            self.col.add(ids=ids[s:s+256], documents=texts[s:s+256],
                         metadatas=metas[s:s+256], embeddings=self._embed(texts[s:s+256]))
        self._build_bm25()
        return doc_id, len(texts)

    def remove_pdf(self, doc_id: str):
        self.col.delete(where={"doc_id": doc_id})
        (PDF_DIR / f"{doc_id}.pdf").unlink(missing_ok=True)
        self._build_bm25()

    # Retrieve (hybrid)
    def search(self, query: str, k: int = 5, doc_ids: list[str] | None = None,
               hybrid: bool = True, pool: int = 20) -> list[Chunk]:
        # Vector search + BM25 keyword search, merged with Reciprocal Rank Fusion.
        if not self._all:
            return []
        where = {"doc_id": {"$in": doc_ids}} if doc_ids else None
        res = self.col.query(query_embeddings=self._embed([query]), n_results=min(pool, len(self._all)),
                             where=where, include=["documents", "metadatas", "distances"])
        vector_ranked = [
            Chunk(id=i, text=d, doc_id=m["doc_id"], filename=m["filename"], page=m["page"], score=1 - dist)
            for i, d, m, dist in zip(res["ids"][0], res["documents"][0],
                                     res["metadatas"][0], res["distances"][0])
        ]
        if not hybrid or self._bm25 is None:
            return vector_ranked[:k]

        scores = self._bm25.get_scores(_tokenize(query))
        order = sorted(range(len(self._all)), key=lambda i: scores[i], reverse=True)
        keyword_ranked = [self._all[i] for i in order
                          if scores[i] > 0 and (not doc_ids or self._all[i].doc_id in doc_ids)][:pool]

        fused: dict[str, float] = {}
        by_id: dict[str, Chunk] = {}
        for ranked in (vector_ranked, keyword_ranked):
            for rank, c in enumerate(ranked):
                fused[c.id] = fused.get(c.id, 0) + 1 / (60 + rank)
                by_id.setdefault(c.id, c)
        top = sorted(fused, key=fused.get, reverse=True)[:k]
        return [Chunk(**{**by_id[i].__dict__, "score": fused[i]}) for i in top]


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+(?:[-_.][a-z0-9]+)*", text.lower())


def render_page(doc_id: str, page: int, highlight: str | None = None, zoom: float = 1.5) -> bytes:
    # PNG of a PDF page, with the cited passage highlighted where it can be located.
    with fitz.open(PDF_DIR / f"{doc_id}.pdf") as doc:
        p = doc[page - 1]
        if highlight:
            # Highlight sentence by sentence; fall back to the first words if cleaning changed the text enough that the full sentence isn't found.
            for sent in re.split(r"(?<=[.!?])\s+", highlight):
                if len(sent) < 15:
                    continue
                quads = p.search_for(sent, quads=True) or p.search_for(" ".join(sent.split()[:8]), quads=True)
                if quads:
                    p.add_highlight_annot(quads)
        return p.get_pixmap(matrix=fitz.Matrix(zoom, zoom)).tobytes("png")


# Generate
SYSTEM_PROMPT = """You answer questions about the user's documents using ONLY the numbered context passages provided.

Rules:
- Base every claim on the passages. Do not use outside knowledge, even if you know the answer.
- Cite the passages you used inline with their numbers in square brackets, e.g. [1] or [2][4].
- If the passages do not contain the answer, say "I couldn't find that in the uploaded documents." and, if useful, mention what related information they do contain.
- Be concise. Use short paragraphs or bullet points where it helps."""


def build_context(chunks: list[Chunk]) -> str:
    return "\n\n".join(
        f"[{n}] (source: {c.filename}, page {c.page})\n{c.text}" for n, c in enumerate(chunks, 1)
    )


def retrieval_query(question: str, history: list[dict]) -> str:
    # Follow-ups like 'what about the second one?' retrieve poorly alone, so add the previous question.
    prev = [m["content"] for m in history if m["role"] == "user"]
    return f"{prev[-1]} {question}" if prev and len(question.split()) < 12 else question

PROVIDERS = {
    "Ollama": {"env": None, "key_url": "https://ollama.com/download",
               "default_models": ["llama3.2", "llama3.1:8b", "qwen2.5:7b"]},
    "Groq": {"env": "GROQ_API_KEY", "key_url": "https://console.groq.com/keys",
             "default_models": ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"]},
    "Gemini": {"env": "GEMINI_API_KEY", "key_url": "https://aistudio.google.com/apikey",
               "default_models": ["gemini-2.5-flash", "gemini-2.5-flash-lite"]},
}
GROQ_URL = "https://api.groq.com/openai/v1"
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta"
OLLAMA_URL = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/")
if not OLLAMA_URL.startswith("http"):
    OLLAMA_URL = f"http://{OLLAMA_URL}"


def _build_messages(question: str, chunks: list[Chunk], history: list[dict]) -> list[dict]:
    messages = [{"role": m["role"], "content": m["content"]} for m in history[-6:]]
    messages.append({"role": "user", "content":
                     f"<context>\n{build_context(chunks)}\n</context>\n\nQuestion: {question}"})
    return messages


def list_models(provider: str, api_key: str) -> list[str]:
    # Ask the provider which models are available (installed ones, for Ollama).
    import requests
    try:
        if provider == "Ollama":
            r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=2)
            r.raise_for_status()
            names = [m["name"] for m in r.json().get("models", [])
                     if "embed" not in m["name"]]
            names = [n.removesuffix(":latest") for n in names]
        elif provider == "Groq":
            r = requests.get(f"{GROQ_URL}/models", headers={"Authorization": f"Bearer {api_key}"}, timeout=10)
            r.raise_for_status()
            names = [m["id"] for m in r.json()["data"]
                     if not any(x in m["id"] for x in ("whisper", "tts", "guard", "embed"))]
        else:
            r = requests.get(f"{GEMINI_URL}/models", headers={"x-goog-api-key": api_key}, timeout=10)
            r.raise_for_status()
            names = [m["name"].removeprefix("models/") for m in r.json().get("models", [])
                     if "generateContent" in m.get("supportedGenerationMethods", [])
                     and m["name"].startswith("models/gemini")]
    except Exception:
        return []
    preferred = [m for m in PROVIDERS[provider]["default_models"] if m in names]
    return preferred + sorted(n for n in names if n not in preferred)


def ollama_running() -> bool:
    import requests
    try:
        return requests.get(f"{OLLAMA_URL}/api/tags", timeout=1).status_code == 200
    except requests.RequestException:
        return False


def stream_answer(provider: str, model: str, api_key: str, question: str,
                  chunks: list[Chunk], history: list[dict]) -> Iterator[str]:
    # Stream an answer from a free LLM (local Ollama or a free-tier API), yielding text pieces.
    import requests

    if provider != "Ollama" and not api_key:
        info = PROVIDERS[provider]
        yield (f"**No {provider} API key.** Get a free one at {info['key_url']}, then paste it in the "
               f"sidebar or set `{info['env']}`.")
        return
    messages = _build_messages(question, chunks, history)
    try:
        if provider == "Ollama":
            yield from _stream_ollama(model, messages)
        elif provider == "Groq":
            yield from _stream_groq(model, api_key, messages)
        else:
            yield from _stream_gemini(model, api_key, messages)
    except requests.HTTPError as e:
        code = e.response.status_code if e.response is not None else None
        if provider == "Ollama" and code == 404:
            yield f"**Model `{model}` isn't installed.** Run `ollama pull {model}` and try again."
        elif code in (401, 403):
            yield f"**Invalid {provider} API key.** Check it and try again."
        elif code == 429:
            yield f"**{provider} free-tier rate limit reached.** Wait a minute, or switch provider/model in the sidebar."
        elif code == 404:
            yield f"**Model `{model}` isn't available on {provider}.** Pick another model in the sidebar."
        else:
            detail = e.response.text[:300] if e.response is not None else str(e)
            yield f"**{provider} error {code}:** {detail}"
    except requests.ConnectionError as e:
        if provider == "Ollama":
            yield ("**Can't reach Ollama.** Install it from https://ollama.com/download, make sure it's "
                   f"running, then run `ollama pull {model}`.")
        else:
            yield f"**Could not reach {provider}:** {e}"
    except requests.RequestException as e:
        yield f"**{provider} error:** {e}"


def _sse_data(resp) -> Iterator[dict]:
    # Parse a Server-Sent Events stream into JSON payloads.
    import json
    for line in resp.iter_lines(decode_unicode=True):
        if line and line.startswith("data:"):
            data = line[5:].strip()
            if data and data != "[DONE]":
                yield json.loads(data)


def _stream_ollama(model: str, messages: list[dict]) -> Iterator[str]:
    import json
    import requests
    body = {"model": model, "stream": True, "options": {"temperature": 0.1, "num_ctx": 8192},
            "messages": [{"role": "system", "content": SYSTEM_PROMPT}] + messages}
    with requests.post(f"{OLLAMA_URL}/api/chat", json=body, stream=True, timeout=600) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if line:
                event = json.loads(line)
                if event.get("error"):
                    yield f"\n\n**Ollama error:** {event['error']}"
                    return
                yield event.get("message", {}).get("content", "")


def _stream_groq(model: str, api_key: str, messages: list[dict]) -> Iterator[str]:
    import requests
    body = {"model": model, "stream": True, "temperature": 0.1, "max_tokens": 2048,
            "messages": [{"role": "system", "content": SYSTEM_PROMPT}] + messages}
    with requests.post(f"{GROQ_URL}/chat/completions", json=body, stream=True, timeout=120,
                       headers={"Authorization": f"Bearer {api_key}"}) as r:
        r.raise_for_status()
        for event in _sse_data(r):
            for choice in event.get("choices", []):
                yield choice.get("delta", {}).get("content") or ""


def _stream_gemini(model: str, api_key: str, messages: list[dict]) -> Iterator[str]:
    import requests
    body = {
        "systemInstruction": {"parts": [{"text": SYSTEM_PROMPT}]},
        "contents": [{"role": "model" if m["role"] == "assistant" else "user",
                      "parts": [{"text": m["content"]}]} for m in messages],
        "generationConfig": {"temperature": 0.1, "maxOutputTokens": 2048},
    }
    with requests.post(f"{GEMINI_URL}/models/{model}:streamGenerateContent?alt=sse", json=body,
                       stream=True, timeout=120, headers={"x-goog-api-key": api_key}) as r:
        r.raise_for_status()
        for event in _sse_data(r):
            for cand in event.get("candidates", []):
                for part in cand.get("content", {}).get("parts", []):
                    if not part.get("thought"):
                        yield part.get("text", "")
                if cand.get("finishReason") == "SAFETY":
                    yield "\n\n_Gemini blocked this answer for safety reasons._"


def cited_numbers(answer: str) -> list[int]:
    return sorted({int(n) for n in re.findall(r"\[(\d+)\]", answer)})

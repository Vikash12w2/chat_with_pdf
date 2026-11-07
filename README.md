# chat_with_pdf

Streamlit app to ask questions about PDFs. It pulls the relevant passages out of the uploaded files and has an LLM answer only from those, with `[n]` citations pointing back to the page. Built without LangChain/LlamaIndex.

LLM options (all free): **Ollama** (local), **Groq** or **Gemini** (free-tier APIs).

## Setup

Needs Python 3.10+.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

The embedding model (`all-MiniLM-L6-v2`, ~90 MB) downloads on the first run.

### LLM

Pick one (or set up more and switch in the sidebar):

- **Ollama** – install from https://ollama.com/download, then `ollama pull llama3.2`. The app picks Ollama by default if it's running. Nothing leaves the machine.
- **Groq** – key from https://console.groq.com/keys
- **Gemini** – key from https://aistudio.google.com/apikey

API keys can be pasted in the sidebar, or set beforehand so they're picked up automatically:

```bash
export GROQ_API_KEY="..."
export GEMINI_API_KEY="..."
```

or in `.streamlit/secrets.toml` (copy `.streamlit/secrets.toml.example`; this file is gitignored).

If Ollama isn't on the default `http://localhost:11434`, set `OLLAMA_HOST`.

### OCR (optional)

Pages with no text layer (scanned PDFs) are OCR'd through PyMuPDF, which needs [Tesseract](https://github.com/tesseract-ocr/tesseract) installed. Without it, scanned pages just come out empty.

## Run

```bash
streamlit run app.py
```

Opens at http://localhost:8501.

1. Upload PDFs in the sidebar → **Index uploaded files**.
2. Ask questions in the chat.
3. Expand **Sources** under an answer to see the passages; **Show page** renders the page with the passage highlighted.

Quick test: index `samples/handbook.pdf` and ask "Which form do I use to request equipment?"

Sidebar also has: tick/untick which docs to search, delete a doc, model picker, `k` (passages retrieved, default 5), hybrid search toggle, clear chat.

## How it works

**Indexing** (`RAGIndex.add_pdf`)
1. Extract text per page with PyMuPDF (OCR fallback for pages with < 20 chars).
2. Clean: drop lines repeated on ≥ 50% of pages (headers/footers), page numbers, fix hyphenation and hard line wraps.
3. Chunk each page on sentence boundaries, ~800 chars with 150 overlap. Chunks never cross pages, so every chunk has one page number.
4. Embed with sentence-transformers and store in ChromaDB (cosine). The doc ID is a SHA-1 of the file bytes, so re-uploading the same PDF is skipped.

**Answering** (`RAGIndex.search` + `stream_answer`)
1. Short follow-up questions (< 12 words) get the previous question prepended for retrieval.
2. Hybrid search: vector search + BM25 keyword search (top 20 each), merged with Reciprocal Rank Fusion, top `k` kept. BM25 helps with exact terms like form codes (`IT-4471`).
3. Passages go to the LLM numbered `[1]..[k]` along with the last 6 chat messages. System prompt: answer only from the passages, cite them, say "I couldn't find that…" otherwise. Temperature 0.1.
4. Answer is streamed; cited passages are listed first in the Sources panel.

## Evaluation

`evaluate.py` measures retrieval only (no LLM). The eval file is a list of questions with the page(s) that answer them:

```json
{"question": "Which form do I use to request equipment?", "pdf": "handbook.pdf", "pages": [2]}
```

It reads the existing index, so **index the PDFs in the app first**, then:

```bash
python evaluate.py samples/eval_set.json --k 5
```

Prints hit@k (correct page anywhere in top k) and MRR (how high the first correct one ranks) for vector-only vs hybrid, plus any misses.

## Files

```
app.py                 Streamlit UI
rag_core.py            everything else: extraction, cleaning, chunking, index, search, LLM calls
evaluate.py            retrieval eval (hit@k, MRR)
samples/               handbook.pdf + eval_set.json
.streamlit/            secrets.toml.example
data/                  created at runtime: chroma/ (vector DB) and pdfs/ (stored copies), gitignored
```

## Things to tweak

Top of `rag_core.py`:

| Setting | Default | |
|---|---|---|
| `CHUNK_SIZE` | 800 | biggest effect on answer quality |
| `CHUNK_OVERLAP` | 150 | |
| `EMBED_MODEL` | `all-MiniLM-L6-v2` | e.g. `BAAI/bge-small-en-v1.5`, `all-mpnet-base-v2` |

Default model lists per provider are in `PROVIDERS`; the system prompt is `SYSTEM_PROMPT`.

**After changing chunking or the embedding model, delete `data/` and re-index.** Deleting `data/` is also how to reset everything.

## Notes

- Ollama only works when running locally. On a deployed version (e.g. Streamlit Cloud), use Groq or Gemini with keys in secrets.
- With Groq/Gemini, the retrieved passages (not the whole PDF) are sent to the provider.
- Free-tier rate limits (429) show up as a message in the chat; wait or switch model/provider.
- BM25 is rebuilt in memory from Chroma on startup and after every add/remove, which is fine for a handful of PDFs but would get slow with a lot of them.
- Tables and images aren't handled specially; table text comes through however PyMuPDF extracts it.
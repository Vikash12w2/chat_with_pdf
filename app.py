import os

import streamlit as st

from chat_store import ChatStore, auto_title
from rag_core import (PROVIDERS, RAGIndex, cited_numbers, list_models, ollama_running,
                      render_page, retrieval_query, stream_answer)

st.set_page_config(page_title="Chat with your PDFs", page_icon="📄", layout="wide")


def _secret(name: str) -> str | None:
    """Read from .streamlit/secrets.toml (used when deployed), if present."""
    try:
        return st.secrets.get(name)
    except Exception:
        return None


@st.cache_data(ttl=3600, show_spinner=False)
def cached_models(provider: str, api_key: str) -> list[str]:
    return list_models(provider, api_key)


@st.cache_resource(show_spinner="Loading embedding model…")
def get_index() -> RAGIndex:
    return RAGIndex()


@st.cache_resource
def get_store() -> ChatStore:
    return ChatStore()


index = get_index()
store = get_store()


# ── Session-state init ──────────────────────────────────────────────────
def _init_state():
    if "conv_id" not in st.session_state:
        st.session_state.conv_id = None
    if "messages" not in st.session_state:
        st.session_state.messages = []     # [{role, content, sources?}]
    if "search_query" not in st.session_state:
        st.session_state.search_query = ""

_init_state()


# ── Chat management helpers ─────────────────────────────────────────────
def _load_conversation(conv_id: str):
    """Switch to an existing conversation."""
    conv = store.get_conversation(conv_id)
    if not conv:
        return
    st.session_state.conv_id = conv.id
    st.session_state.messages = [
        {"role": m.role, "content": m.content, "sources": m.sources}
        for m in conv.messages
    ]
    # Restore the document selections for this conversation
    saved_docs = set(conv.doc_ids or [])
    for d_id in index.documents():
        st.session_state[f"sel-{d_id}"] = (d_id in saved_docs) if saved_docs else True


def _new_chat():
    """Start a fresh conversation."""
    st.session_state.conv_id = None
    st.session_state.messages = []
    for d_id in index.documents():
        st.session_state[f"sel-{d_id}"] = True


def _ensure_conversation() -> str:
    """Create a DB conversation if one doesn't exist yet (lazy creation on
    first message, so empty 'New chat' windows don't litter the history)."""
    if st.session_state.conv_id is None:
        conv_id = store.create_conversation()
        st.session_state.conv_id = conv_id
    return st.session_state.conv_id


# ── Sidebar ──────────────────────────────────────────────────────────────
with st.sidebar:
    # ── Chat history section ────────────────────────────────────────────
    st.header("💬 Chats")

    # New chat button
    if st.button("➕  New chat", use_container_width=True, type="primary"):
        _new_chat()
        st.rerun()

    # Search bar
    search_q = st.text_input(
        "Search chats…", value=st.session_state.search_query,
        key="chat_search_input", label_visibility="collapsed",
        placeholder="🔍 Search chats…",
    )
    st.session_state.search_query = search_q

    # Chat list
    convos = store.list_conversations(search=search_q or None)
    if convos:
        for conv in convos:
            col1, col2 = st.columns([5, 1])
            is_active = conv.id == st.session_state.conv_id
            label = f"**{conv.title}**" if is_active else conv.title
            if col1.button(label, key=f"conv-{conv.id}", use_container_width=True,
                           disabled=is_active):
                _load_conversation(conv.id)
                st.rerun()
            if col2.button("🗑", key=f"del-conv-{conv.id}", help=f"Delete '{conv.title}'"):
                store.delete_conversation(conv.id)
                if st.session_state.conv_id == conv.id:
                    _new_chat()
                st.rerun()
    elif search_q:
        st.caption("No chats match your search.")

    st.divider()

    # ── Documents section ───────────────────────────────────────────────
    st.header("📚 Documents")
    uploads = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)
    if uploads and st.button("Index uploaded files", type="primary", use_container_width=True):
        for f in uploads:
            with st.spinner(f"Indexing {f.name}…"):
                try:
                    _, n = index.add_pdf(f.name, f.getvalue())
                    st.success(f"{f.name}: {n} chunks" if n else f"{f.name}: already indexed")
                except Exception as e:
                    st.error(f"{f.name}: {e}")

    docs = index.documents()
    selected = []
    if docs:
        st.caption("Search in:")
        for doc_id, name in docs.items():
            c1, c2 = st.columns([5, 1])
            if c1.checkbox(name, value=True, key=f"sel-{doc_id}"):
                selected.append(doc_id)
            if c2.button("🗑", key=f"del-{doc_id}", help=f"Remove {name}"):
                index.remove_pdf(doc_id)
                st.rerun()
    else:
        st.info("No documents yet. Upload a PDF to start.")

    st.divider()

    # Settings section
    st.header("⚙️ Settings")
    local = ollama_running()
    provider = st.radio("Free LLM", list(PROVIDERS), horizontal=True,
                        index=0 if local else 1,
                        help="Ollama runs on your computer (private, unlimited). "
                             "Groq and Gemini are free online APIs (faster, bigger models).")
    info = PROVIDERS[provider]
    api_key = None
    if provider == "Ollama":
        models = list_models("Ollama", "") if local else []
        if not local:
            st.warning("Ollama isn't running. [Install it](https://ollama.com/download), "
                       "run `ollama pull llama3.2`, then reload this page.")
        elif not models:
            st.warning("No models installed yet. Run `ollama pull llama3.2`, then reload this page.")
    else:
        api_key = os.environ.get(info["env"]) or _secret(info["env"])
        if not api_key:
            api_key = st.text_input(f"{provider} API key", type="password",
                                    help=f"Free key: {info['key_url']}. Or set {info['env']} before launching.")
            st.caption(f"[Get a free {provider} key]({info['key_url']}), no credit card needed.")
        models = cached_models(provider, api_key) if api_key else []
    model = st.selectbox("Model", models or info["default_models"])
    top_k = st.slider("Passages to retrieve (k)", 2, 10, 5)
    hybrid = st.toggle("Hybrid search (keywords + embeddings)", value=True)
    if st.button("Clear chat", use_container_width=True,
                 help="Clear all messages in this chat"):
        if st.session_state.conv_id:
            store.clear_messages(st.session_state.conv_id)
        st.session_state.messages = []
        st.rerun()


# Sources panel
def show_sources(sources: list[dict], answer: str):
    """Cited passages first, each expandable to the exact text and the highlighted page."""
    cited = set(cited_numbers(answer))
    order = sorted(range(len(sources)), key=lambda i: (i + 1) not in cited)
    with st.expander(f"📎 Sources ({len(cited)} cited, {len(sources)} retrieved)"):
        for i in order:
            s = sources[i]
            n = i + 1
            label = f"**[{n}]** {s['filename']} — page {s['page']}"
            if n not in cited:
                label += "  _(retrieved, not cited)_"
            st.markdown(label)
            st.caption(s["text"])
            if st.toggle("Show page", key=f"pg-{s['id']}-{id(sources)}-{n}"):
                try:
                    st.image(render_page(s["doc_id"], s["page"], highlight=s["text"]))
                except Exception as e:
                    st.error(f"Could not render page: {e}")
            st.divider()


# Chat
st.title("📄 Chat with your PDFs")
st.caption("Answers come only from your documents, with page citations like [1].")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            show_sources(msg["sources"], msg["content"])

if question := st.chat_input("Ask a question about your documents…", disabled=not docs):
    history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        chunks = index.search(retrieval_query(question, history), k=top_k,
                              doc_ids=selected or None, hybrid=hybrid)
        if not chunks:
            answer = "No passages found. Select at least one document in the sidebar."
            st.markdown(answer)
        else:
            answer = st.write_stream(stream_answer(provider, model, api_key, question, chunks, history))
        sources = [c.__dict__ for c in chunks]
        if chunks:
            show_sources(sources, answer)

    # Persist to chat history
    conv_id = _ensure_conversation()

    # Auto-title from first user question
    if not history:
        store.update_title(conv_id, auto_title(question))

    # Save the selected documents
    store.update_doc_ids(conv_id, selected)

    # Save both messages
    store.add_message(conv_id, "user", question)
    store.add_message(conv_id, "assistant", answer,
                      sources=sources if chunks else None)

    st.session_state.messages += [
        {"role": "user", "content": question},
        {"role": "assistant", "content": answer, "sources": sources if chunks else None},
    ]

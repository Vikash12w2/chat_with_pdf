import os

import streamlit as st

from chat_store import ChatStore, auto_title
from rag_core import (PROVIDERS, RAGIndex, cited_numbers, list_models, ollama_running,
                      render_page, retrieval_query, stream_answer)

st.set_page_config(page_title="Chat with your PDFs", page_icon="📄", layout="wide")


def _secret(name: str) -> str | None:
    try:
        return st.secrets.get(name)
    except Exception:
        return None


@st.cache_data(ttl=3600, show_spinner=False)
def cached_models(provider: str, api_key: str) -> list[str]:
    return list_models(provider, api_key)

@st.cache_data(ttl=15, show_spinner=False)
def cached_ollama_running() -> bool:
    return ollama_running()


@st.cache_data(ttl=15, show_spinner=False)
def cached_ollama_models() -> list[str]:
    return list_models("Ollama", "")


@st.cache_data(show_spinner=False)
def cached_page(doc_id: str, page: int, highlight: str) -> bytes:
    return render_page(doc_id, page, highlight=highlight)


@st.cache_resource(show_spinner="Loading embedding model…")
def get_index() -> RAGIndex:
    return RAGIndex()


@st.cache_resource
def get_store() -> ChatStore:
    return ChatStore()


index = get_index()
store = get_store()


# Session-state init
def _init_state():
    st.session_state.setdefault("conv_id", None)
    st.session_state.setdefault("messages", [])     # [{role, content, sources?, attachments?}]
    st.session_state.setdefault("chat_docs", [])    # doc_ids attached to the current chat
    st.session_state.setdefault("search_query", "")

_init_state()


# Chat management helpers
def _load_conversation(conv_id: str):
    # Switch to an existing conversation.
    conv = store.get_conversation(conv_id)
    if not conv:
        return
    st.session_state.conv_id = conv.id
    st.session_state.messages = [
        {"role": m.role, "content": m.content, "sources": m.sources,
         "attachments": m.attachments}
        for m in conv.messages
    ]
    known = index.documents()
    st.session_state.chat_docs = [d for d in conv.doc_ids if d in known]
    st.session_state.lib_pick = []


def _new_chat():
    # Start a fresh conversation. 
    st.session_state.conv_id = None
    st.session_state.messages = []
    st.session_state.chat_docs = []
    st.session_state.lib_pick = []


def _ensure_conversation(title: str = "New chat") -> str:
    # Create a DB conversation if one doesn't exist yet (lazy creation on first message/attachment, so empty 'New chat' windows don't litter the history).
    if st.session_state.conv_id is None:
        st.session_state.conv_id = store.create_conversation(title=auto_title(title))
    return st.session_state.conv_id


def _cleanup_docs(doc_ids: list[str]):
    # Delete PDFs (file + chunks) that no chat uses anymore.
    for d in doc_ids:
        if not store.doc_in_use(d):
            index.remove_pdf(d)


def _link_docs(doc_ids: list[str], title_hint: str):
    # Attach already-indexed PDFs to the current chat.
    new = [d for d in doc_ids if d not in st.session_state.chat_docs]
    if not new:
        return
    conv_id = _ensure_conversation(title_hint)
    st.session_state.chat_docs = st.session_state.chat_docs + new
    store.update_doc_ids(conv_id, st.session_state.chat_docs)


def _unlink_doc(doc_id: str):
    # Remove a PDF from the current chat; delete it if no other chat uses it.
    st.session_state.chat_docs = [d for d in st.session_state.chat_docs if d != doc_id]
    if st.session_state.conv_id:
        store.update_doc_ids(st.session_state.conv_id, st.session_state.chat_docs)
    _cleanup_docs([doc_id])


def _add_from_library():
    picked = st.session_state.get("lib_pick") or []
    if picked:
        _link_docs(picked, title_hint=index.documents().get(picked[0], "New chat"))
    st.session_state.lib_pick = []


def _delete_conversation(conv_id: str):
    doc_ids = store.get_doc_ids(conv_id)
    store.delete_conversation(conv_id)
    if st.session_state.conv_id == conv_id:
        _new_chat()
    _cleanup_docs(doc_ids)


def _clear_chat():
    # Clear the messages but keep the chat and its documents.
    if st.session_state.conv_id:
        store.clear_messages(st.session_state.conv_id)
    st.session_state.messages = []


# Sidebar 
with st.sidebar:
    # Chat history section 
    st.header("💬 Chats")

    # New chat button
    st.button("➕  New chat", use_container_width=True, type="primary", on_click=_new_chat)

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
            col1.button(label, key=f"conv-{conv.id}", use_container_width=True,
                        disabled=is_active, on_click=_load_conversation, args=(conv.id,))
            col2.button("🗑", key=f"del-conv-{conv.id}", help=f"Delete '{conv.title}'",
                        on_click=_delete_conversation, args=(conv.id,))
    elif search_q:
        st.caption("No chats match your search.")

    st.divider()

    # Documents in this chat
    st.header("📎 Documents")
    docs = index.documents()                       # whole library: doc_id -> filename
    chat_docs = [d for d in st.session_state.chat_docs if d in docs]
    if chat_docs:
        for doc_id in chat_docs:
            c1, c2 = st.columns([5, 1])
            c1.markdown(f"📄 {docs[doc_id]}")
            c2.button("✕", key=f"unlink-{doc_id}",
                      help="Remove from this chat (deleted if no other chat uses it)",
                      on_click=_unlink_doc, args=(doc_id,))
    else:
        st.caption("No PDFs in this chat yet. Attach some with the 📎 in the chat box.")

    others = {d: n for d, n in docs.items() if d not in chat_docs}
    if others:
        with st.expander("Add from library"):
            st.multiselect("PDFs", list(others), format_func=others.get, key="lib_pick",
                           label_visibility="collapsed",
                           placeholder="PDFs uploaded in other chats")
            st.button("Add to this chat", use_container_width=True,
                      on_click=_add_from_library,
                      disabled=not st.session_state.get("lib_pick"))

    st.divider()

    # Settings section
    st.header("⚙️ Settings")
    local = cached_ollama_running()
    provider = st.radio("Free LLM", list(PROVIDERS), horizontal=True,
                        index=0 if local else 1,
                        help="Ollama runs on your computer (private, unlimited). "
                             "Groq and Gemini are free online APIs (faster, bigger models).")
    info = PROVIDERS[provider]
    api_key = None
    if provider == "Ollama":
        models = cached_ollama_models() if local else []
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
    st.button("Clear chat", use_container_width=True,
              help="Clear all messages in this chat", on_click=_clear_chat)


# Sources panel
def show_sources(sources: list[dict], answer: str):
    # Cited passages first, each expandable to the exact text and the highlighted page.
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
                    st.image(cached_page(s["doc_id"], s["page"], s["text"]))
                except Exception as e:
                    st.error(f"Could not render page: {e}")
            st.divider()


# Chat
st.title("📄 Chat with your PDFs")
st.caption("Attach PDFs with 📎 and ask questions. Answers come only from this chat's documents, "
           "with page citations like [1].")


def show_user_message(content: str, attachments: list[str] | None):
    if content:
        st.markdown(content)
    if attachments:
        st.caption("📎 " + ", ".join(attachments))


for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "user":
            show_user_message(msg["content"], msg.get("attachments"))
        else:
            st.markdown(msg["content"])
            if msg.get("sources"):
                show_sources(msg["sources"], msg["content"])

prompt = st.chat_input("Ask a question, or attach PDFs with 📎",
                       accept_file="multiple", file_type=["pdf"])
if prompt:
    question = (prompt.text or "").strip()
    files = prompt.files or []
    # Attachment-only messages have no text, so they're left out of the LLM history.
    history = [{"role": m["role"], "content": m["content"]}
               for m in st.session_state.messages if m["content"]]
    first_question = not any(m["role"] == "user" and m["content"]
                             for m in st.session_state.messages)
    was_new_chat = st.session_state.conv_id is None

    # Index attachments and link them to this chat
    attached = []
    with st.chat_message("user"):
        if question:
            st.markdown(question)
        for f in files:
            with st.spinner(f"Indexing {f.name}…"):
                try:
                    doc_id, _ = index.add_pdf(f.name, f.getvalue())
                    _link_docs([doc_id], title_hint=f.name)
                    attached.append(f.name)
                except Exception as e:
                    st.error(f"{f.name}: {e}")
        if attached:
            st.caption("📎 " + ", ".join(attached))

    chat_docs = st.session_state.chat_docs
    answer, sources = None, None
    if question and not chat_docs:
        # Nothing to search: show a hint but don't save anything.
        with st.chat_message("assistant"):
            st.info("This chat has no PDFs yet. Attach one with the 📎 in the chat box, "
                    "or add one from the library in the sidebar.")
    elif question:
        with st.chat_message("assistant"):
            chunks = index.search(retrieval_query(question, history), k=top_k,
                                  doc_ids=chat_docs, hybrid=hybrid)
            if not chunks:
                answer = "No passages found in this chat's documents."
                st.markdown(answer)
            else:
                answer = st.write_stream(stream_answer(provider, model, api_key,
                                                       question, chunks, history))
                sources = [c.__dict__ for c in chunks]
                show_sources(sources, answer)

    # Persist to chat history
    if answer is not None or attached:
        conv_id = _ensure_conversation(question or attached[0])
        if answer is not None and first_question:
            store.update_title(conv_id, auto_title(question))   # title = first question
        store.add_message(conv_id, "user", question, attachments=attached or None)
        st.session_state.messages.append(
            {"role": "user", "content": question, "attachments": attached or None})
        if answer is not None:
            store.add_message(conv_id, "assistant", answer, sources=sources)
            st.session_state.messages.append(
                {"role": "assistant", "content": answer, "sources": sources})

        # The sidebar was drawn before this; refresh it so the new chat / PDFs show up.
        if was_new_chat or attached:
            st.rerun()
"""
Streamlit frontend for the Multimodal Study Assistant.

Talks to the FastAPI backend at localhost:8000.
Run with: streamlit run frontend/app.py
"""

import streamlit as st
import requests
import time

API_URL = "http://localhost:8000"

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Study Assistant",
    page_icon="📚",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------
# Chat history and uploaded-file lists are keyed BY MODULE so switching modules
# here resets when the Streamlit process restarts.
if "messages" not in st.session_state:
    st.session_state.messages = {}  # {module: [{role, content, sources}]}

if "session_id" not in st.session_state:
    st.session_state.session_id = None  # currently selected module

if "ingested_files" not in st.session_state:
    st.session_state.ingested_files = {}  # {module: [str]}

if "created_modules" not in st.session_state:
    # A module created here has no chunks until something is ingested, so it
    # won't come back from GET /sessions yet. Remember it locally so it stays
    # selectable in the meantime.
    st.session_state.created_modules = []


# ---------------------------------------------------------------------------
# Helper — call backend
# ---------------------------------------------------------------------------

def check_health():
    """Return (status_ok, ollama_ok) booleans."""
    try:
        r = requests.get(f"{API_URL}/health", timeout=3)
        data = r.json()
        return True, data.get("ollama", False)
    except Exception:
        return False, False


def fetch_sessions() -> list:
    """GET /sessions -> [{"session_id": ..., "chunks": n}]. [] if unreachable."""
    try:
        r = requests.get(f"{API_URL}/sessions", timeout=5)
        r.raise_for_status()
        return r.json()
    except Exception:
        return []


def ingest_file(uploaded_file, session_id: str) -> dict:
    """POST a file to /ingest and return the JSON response."""
    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
    data = {"session_id": session_id}
    r = requests.post(f"{API_URL}/ingest", files=files, data=data, timeout=300)
    r.raise_for_status()
    return r.json()


def query_backend(question: str, session_id: str) -> dict:
    """POST to /query and return {answer, sources}."""
    payload = {"question": question, "session_id": session_id}
    r = requests.post(f"{API_URL}/query", json=payload, timeout=600)
    r.raise_for_status()
    return r.json()


# ---------------------------------------------------------------------------
# Sidebar — status + upload + session
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("📚 Study Assistant")
    st.divider()

    # --- Health status ---
    api_ok, ollama_ok = check_health()
    col1, col2 = st.columns(2)
    col1.metric("API", "✅ Online" if api_ok else "❌ Offline")
    col2.metric("Ollama", "✅ Ready" if ollama_ok else "❌ Down")

    if not api_ok:
        st.error("FastAPI is not running. Start it with:\nuvicorn backend.main:app --reload")
    if api_ok and not ollama_ok:
        st.warning("Ollama is not running. Start it with:\nollama serve")

    st.divider()

    # --- Module switcher ---
    st.subheader("Study Module")

    server_sessions = fetch_sessions()
    chunk_counts = {s["session_id"]: s["chunks"] for s in server_sessions}
    # Union the server's modules with any created this run but not yet ingested.
    modules = sorted(set(chunk_counts) | set(st.session_state.created_modules))

    if modules:
        # Keep the selection valid if the previous module vanished server-side.
        if st.session_state.session_id not in modules:
            st.session_state.session_id = modules[0]

        selected = st.selectbox(
            "Active module",
            modules,
            index=modules.index(st.session_state.session_id),
            format_func=lambda m: f"{m}  ({chunk_counts.get(m, 0)} chunks)",
            help="Queries and uploads apply only to the selected module.",
        )
        st.session_state.session_id = selected
    else:
        # Empty store and nothing created yet — no dropdown, just the create box.
        st.info("No modules yet. Create one below to get started.")
        st.session_state.session_id = None

    new_module = st.text_input(
        "New module name",
        placeholder="e.g. CM3060",
        help="Each module keeps its own material and its own conversation.",
    )
    if st.button("Create Module", use_container_width=True):
        name = new_module.strip()
        if not name:
            st.warning("Enter a module name first.")
        elif name in modules:
            st.warning(f"Module '{name}' already exists.")
        else:
            st.session_state.created_modules.append(name)
            st.session_state.session_id = name
            st.session_state.messages.setdefault(name, [])
            st.session_state.ingested_files.setdefault(name, [])
            st.rerun()

    # The selected module scopes everything below.
    module = st.session_state.session_id

    st.divider()

    # --- File upload ---
    st.subheader("Upload Study Material")

    # A successful ingest reruns the script so the dropdown re-fetches its chunk
    # counts, which discards the st.success() from that pass. The message is
    # parked in session state instead and shown once here, after the rerun.
    notice = st.session_state.pop("ingest_notice", None)
    if notice:
        st.success(notice)

    uploaded = st.file_uploader(
        "Drop a file to ingest",
        type=["pdf", "pptx", "mp3", "wav", "png", "jpg", "jpeg"],
        help="PDF, PPTX, audio, or image files are all supported.",
    )

    if uploaded and not module:
        st.warning("Create or select a module before ingesting.")
    elif uploaded:
        if st.button("Ingest File", type="primary", use_container_width=True):
            ingest_ok = False
            with st.spinner(f"Ingesting {uploaded.name} into {module}..."):
                try:
                    result = ingest_file(uploaded, module)
                    st.session_state.ingested_files.setdefault(module, []).append(
                        f"{result['filename']} ({result['chunks_stored']} chunks)"
                    )
                    st.session_state.ingest_notice = (
                        f"✅ **{result['filename']}** — "
                        f"{result['chunks_stored']} chunks stored in **{module}**"
                    )
                    ingest_ok = True
                except requests.exceptions.HTTPError as e:
                    st.error(f"Ingest failed: {e.response.text}")
                except Exception as e:
                    st.error(f"Error: {e}")

            # Rerun OUTSIDE the try: st.rerun() works by raising, so calling it
            # above would be caught by `except Exception` and shown as an error.
            # The rerun re-runs fetch_sessions(), refreshing the dropdown count.
            if ingest_ok:
                st.rerun()

    # --- Ingested files list (this module only) ---
    module_files = st.session_state.ingested_files.get(module, []) if module else []
    if module_files:
        st.divider()
        st.subheader("Ingested Files")
        for f in module_files:
            st.markdown(f"📄 {f}")

    st.divider()

    # --- Clear chat (this module only) ---
    if st.button("Clear Chat", use_container_width=True, disabled=not module):
        st.session_state.messages[module] = []
        st.rerun()


# ---------------------------------------------------------------------------
# Main area — chat interface
# ---------------------------------------------------------------------------
st.title("💬 Ask Your Notes")

if not module:
    st.info("👈 Create a module in the sidebar to get started.")
    st.stop()

st.caption(
    f"Module: **{module}** — answers are grounded only in this module's material."
)

# This module's history. setdefault returns the live list, so appends below
# write straight back into st.session_state.messages[module].
messages = st.session_state.messages.setdefault(module, [])

# Render existing chat history
for msg in messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        # Show sources below assistant messages if available
        if msg["role"] == "assistant" and msg.get("sources"):
            with st.expander(f"📎 Sources ({len(msg['sources'])} chunks)"):
                for i, src in enumerate(msg["sources"], 1):
                    source_file = src.get("source_file", "unknown")
                    source_type = src.get("source_type", "")
                    page = src.get("page_or_slide")
                    page_str = f" — page/slide {page}" if page else ""
                    st.markdown(f"**{i}.** `{source_file}` ({source_type}){page_str}")

# Chat input
if question := st.chat_input("Ask a question about your notes..."):

    # Show the user's message immediately
    messages.append({"role": "user", "content": question, "sources": []})
    with st.chat_message("user"):
        st.markdown(question)

    # Call backend and stream a spinner while waiting
    with st.chat_message("assistant"):
        with st.spinner("Thinking... (this may take 1-2 minutes on CPU)"):
            try:
                result = query_backend(question, module)
                answer = result.get("answer", "No answer returned.")
                sources = result.get("sources", [])

                st.markdown(answer)

                # Show sources inline
                if sources:
                    with st.expander(f"📎 Sources ({len(sources)} chunks)"):
                        for i, src in enumerate(sources, 1):
                            source_file = src.get("source_file", "unknown")
                            source_type = src.get("source_type", "")
                            page = src.get("page_or_slide")
                            page_str = f" — page/slide {page}" if page else ""
                            st.markdown(f"**{i}.** `{source_file}` ({source_type}){page_str}")

                # Save to history
                messages.append(
                    {"role": "assistant", "content": answer, "sources": sources}
                )

            except requests.exceptions.HTTPError as e:
                error_msg = f"❌ Query failed: {e.response.status_code} — {e.response.text}"
                st.error(error_msg)
                messages.append(
                    {"role": "assistant", "content": error_msg, "sources": []}
                )
            except requests.exceptions.Timeout:
                msg = "❌ Request timed out. The model is taking too long — try a shorter question."
                st.error(msg)
                messages.append(
                    {"role": "assistant", "content": msg, "sources": []}
                )
            except Exception as e:
                msg = f"❌ Unexpected error: {e}"
                st.error(msg)
                messages.append(
                    {"role": "assistant", "content": msg, "sources": []}
                )

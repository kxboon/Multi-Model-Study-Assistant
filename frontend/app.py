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
if "messages" not in st.session_state:
    st.session_state.messages = []  # chat history: [{role, content, sources}]

if "session_id" not in st.session_state:
    st.session_state.session_id = "default"

if "ingested_files" not in st.session_state:
    st.session_state.ingested_files = []  # track what has been uploaded


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

    # --- Session ID ---
    st.subheader("Study Session")
    session_id = st.text_input(
        "Session ID",
        value=st.session_state.session_id,
        help="Give each study topic its own ID so queries only search that material.",
    )
    st.session_state.session_id = session_id

    st.divider()

    # --- File upload ---
    st.subheader("Upload Study Material")
    uploaded = st.file_uploader(
        "Drop a file to ingest",
        type=["pdf", "pptx", "mp3", "wav", "png", "jpg", "jpeg"],
        help="PDF, PPTX, audio, or image files are all supported.",
    )

    if uploaded:
        if st.button("Ingest File", type="primary", use_container_width=True):
            with st.spinner(f"Ingesting {uploaded.name}..."):
                try:
                    result = ingest_file(uploaded, st.session_state.session_id)
                    st.success(
                        f"✅ **{result['filename']}** — {result['chunks_stored']} chunks stored"
                    )
                    st.session_state.ingested_files.append(
                        f"{result['filename']} ({result['chunks_stored']} chunks)"
                    )
                except requests.exceptions.HTTPError as e:
                    st.error(f"Ingest failed: {e.response.text}")
                except Exception as e:
                    st.error(f"Error: {e}")

    # --- Ingested files list ---
    if st.session_state.ingested_files:
        st.divider()
        st.subheader("Ingested Files")
        for f in st.session_state.ingested_files:
            st.markdown(f"📄 {f}")

    st.divider()

    # --- Clear chat ---
    if st.button("Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# ---------------------------------------------------------------------------
# Main area — chat interface
# ---------------------------------------------------------------------------
st.title("💬 Ask Your Notes")
st.caption(f"Session: **{st.session_state.session_id}** — answers are grounded only in your uploaded material.")

# Render existing chat history
for msg in st.session_state.messages:
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
    st.session_state.messages.append({"role": "user", "content": question, "sources": []})
    with st.chat_message("user"):
        st.markdown(question)

    # Call backend and stream a spinner while waiting
    with st.chat_message("assistant"):
        with st.spinner("Thinking... (this may take 1-2 minutes on CPU)"):
            try:
                result = query_backend(question, st.session_state.session_id)
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
                st.session_state.messages.append(
                    {"role": "assistant", "content": answer, "sources": sources}
                )

            except requests.exceptions.HTTPError as e:
                error_msg = f"❌ Query failed: {e.response.status_code} — {e.response.text}"
                st.error(error_msg)
                st.session_state.messages.append(
                    {"role": "assistant", "content": error_msg, "sources": []}
                )
            except requests.exceptions.Timeout:
                msg = "❌ Request timed out. The model is taking too long — try a shorter question."
                st.error(msg)
                st.session_state.messages.append(
                    {"role": "assistant", "content": msg, "sources": []}
                )
            except Exception as e:
                msg = f"❌ Unexpected error: {e}"
                st.error(msg)
                st.session_state.messages.append(
                    {"role": "assistant", "content": msg, "sources": []}
                )

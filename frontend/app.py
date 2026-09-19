"""
Streamlit frontend for the Multimodal Study Assistant.

Talks to the FastAPI backend at 127.0.0.1:8000.
Run with: streamlit run frontend/app.py
"""

import streamlit as st
import requests
import time

# 127.0.0.1, not localhost: "localhost" resolves to ::1 first on this machine,
# and uvicorn binds IPv4 only, so every call waits ~2s for the IPv6 attempt to
# fail before falling back. That cost was paid on every rerun.
API_URL = "http://127.0.0.1:8000"

# Shown under each per-subject breakdown. Subjects are free text typed per quiz
# or deck, and the backend only matches them case-insensitively, so closely
# related wordings stay in separate rows. Saying so beats letting a split
# breakdown read as a bug.
SUBJECT_MATCH_NOTE = (
    "Subjects are matched by exact wording (ignoring case). Differently worded "
    "subjects stay separate even when they cover the same ground."
)

# Dev/test/evaluation modules that live in the same ChromaDB collection as real
# study material (verification/*.py harnesses and ad hoc manual testing write
# to the same store) but must never appear in a participant's module dropdown.
# Exact session_id match — these are never shown, but they are not deleted, so
# existing signals/vectorstore data tied to them stays intact.
HIDDEN_MODULES = {
    "Kai_Xiang_Test",
    "eval_set",
    "verify_audio",
    "verify_image",
    "migration_check",
    "C3015 ML",
    # Reads like a real module, but it is 51 chunks of test_notes.pdf — the
    # same fixture already hidden above as "C3015 ML" and "migration_check".
    "Machine Learning",
}

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

if "quiz" not in st.session_state:
    # Keyed by module, same as messages and ingested_files, so switching
    # modules never shows a quiz generated from another module's material.
    # {module: {"topic": str, "items": [...], "meta": {...}, "results": [...]}}
    st.session_state.quiz = {}

if "flashcards" not in st.session_state:
    # Keyed by module for the same reason as quiz. In-memory only — a deck
    # lasts as long as the session. Spaced repetition is out of scope: nothing
    # here schedules cards or survives a new deck.
    # {module: {"topic", "cards", "nonce", "index", "revealed", "ratings"}}
    st.session_state.flashcards = {}


# ---------------------------------------------------------------------------
# Helper — call backend
# ---------------------------------------------------------------------------

# Both helpers below run on EVERY rerun — including each flashcard reveal and
# rating click — and each is an HTTP round trip. Uncached they dominated the
# cost of an interaction, so both are cached with a short TTL.

@st.cache_data(ttl=15, show_spinner=False)
def check_health():
    """Return (status_ok, ollama_ok) booleans.

    TTL is deliberately the shorter of the two: this drives a live status
    indicator, so a stale "Ollama ready" after Ollama has died is actively
    misleading. 15s bounds how long the display can lie.
    """
    try:
        r = requests.get(f"{API_URL}/health", timeout=3)
        data = r.json()
        return True, data.get("ollama", False)
    except Exception:
        return False, False


@st.cache_data(ttl=30, show_spinner=False)
def fetch_sessions() -> list:
    """GET /sessions -> [{"session_id": ..., "chunks": n}]. [] if unreachable.

    Longer TTL than check_health: the only change this app makes to the session
    list is an ingest, and that path clears this cache explicitly. The TTL is
    just a backstop for changes made outside this session.
    """
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


def generate_quiz(topic: str, session_id: str, n_questions: int) -> dict:
    """POST to /quiz and return the generated items plus their metadata."""
    payload = {
        "topic": topic,
        "session_id": session_id,
        "n_questions": n_questions,
    }
    r = requests.post(f"{API_URL}/quiz", json=payload, timeout=900)
    r.raise_for_status()
    return r.json()


def post_quiz_signals(session_id: str, topic: str, results: list) -> dict:
    """POST marked outcomes to /quiz/signals so each becomes a learning signal.

    Callers must treat a failure here as non-fatal — the student's score is
    already computed and displayed by the time this runs.
    """
    payload = {
        "session_id": session_id,
        "topic": topic,
        "results": [
            {
                "question": r["question"],
                "correct": r["correct"],
                "source_file": r.get("source_file"),
            }
            for r in results
        ],
    }
    resp = requests.post(f"{API_URL}/quiz/signals", json=payload, timeout=60)
    resp.raise_for_status()
    return resp.json()


def generate_flashcards(topic: str, session_id: str, n_cards: int) -> dict:
    """POST to /flashcards and return the generated deck plus its metadata."""
    payload = {"topic": topic, "session_id": session_id, "n_cards": n_cards}
    r = requests.post(f"{API_URL}/flashcards", json=payload, timeout=900)
    r.raise_for_status()
    return r.json()


def post_flashcard_signals(session_id: str, topic: str, results: list) -> dict:
    """POST self-rated cards to /flashcards/signals.

    Callers must treat a failure as non-fatal — the student has already seen
    the card, and a logging problem must not block moving to the next one.
    """
    payload = {
        "session_id": session_id,
        "topic": topic,
        "results": [
            {
                "term": r["term"],
                "known": r["known"],
                "source_file": r.get("source_file"),
            }
            for r in results
        ],
    }
    resp = requests.post(f"{API_URL}/flashcards/signals", json=payload, timeout=60)
    resp.raise_for_status()
    return resp.json()


def fetch_confidence(session_id: str) -> dict:
    """GET /confidence for one module. Deliberately uncached — it must reflect
    the quiz or deck the student just finished."""
    r = requests.get(f"{API_URL}/confidence",
                     params={"session_id": session_id}, timeout=30)
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
    # Union the server's modules with any created this run but not yet ingested,
    # excluding dev/test/evaluation modules that share the same store.
    modules = sorted(
        (set(chunk_counts) | set(st.session_state.created_modules)) - HIDDEN_MODULES
    )

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
                # MUST clear first. The ingest just changed the chunk counts, but
                # fetch_sessions() is cached — without this the rerun would be
                # served the pre-ingest counts and the dropdown would show a
                # stale number until the TTL expired, reintroducing exactly the
                # bug the rerun above was added to fix.
                fetch_sessions.clear()
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
st.title("📖 Study Workspace")

if not module:
    st.info("👈 Create a module in the sidebar to get started.")
    st.stop()

st.caption(
    f"Module: **{module}** — answers are grounded only in this module's material."
)

tab_chat, tab_quiz, tab_cards, tab_progress = st.tabs(
    ["💬 Chat", "📝 Quiz", "🗂️ Flashcards", "📊 Progress"]
)

# ---------------------------------------------------------------------------
# Chat tab — behaviour unchanged, only relocated inside the tab
# ---------------------------------------------------------------------------
with tab_chat:
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


# ---------------------------------------------------------------------------
# Quiz tab — generate MCQs from this module, mark them locally
# ---------------------------------------------------------------------------
with tab_quiz:
    quiz = st.session_state.quiz.get(module)

    # --- Config form ---
    with st.form("quiz_config"):
        topic_input = st.text_input(
            "Topic",
            value=(quiz or {}).get("topic", ""),
            placeholder="e.g. tokenization",
            help="Questions are retrieved and written about this topic.",
        )
        n_questions = st.radio(
            "Number of questions", [3, 5, 10], index=1, horizontal=True
        )
        generate = st.form_submit_button(
            "Generate Quiz", type="primary", use_container_width=True
        )

    if generate:
        topic_clean = topic_input.strip()
        if not topic_clean:
            st.warning("Enter a topic first.")
        else:
            with st.spinner(
                f"Generating {n_questions} questions on '{topic_clean}'... "
                "(this may take a few minutes on CPU)"
            ):
                try:
                    data = generate_quiz(topic_clean, module, n_questions)
                    st.session_state.quiz[module] = {
                        "topic": topic_clean,
                        "items": data.get("items", []),
                        "requested": data.get("requested"),
                        "parsed": data.get("parsed"),
                        "generation_time_s": data.get("generation_time_s"),
                        "warnings": data.get("warnings", []),
                        # A fresh nonce namespaces this quiz's radio widgets, so
                        # answers from a previous quiz cannot bleed into it.
                        "nonce": str(time.time()),
                        "results": None,
                    }
                    quiz = st.session_state.quiz[module]
                except requests.exceptions.HTTPError as e:
                    st.error(
                        f"Quiz generation failed: {e.response.status_code} — "
                        f"{e.response.text}"
                    )
                except requests.exceptions.Timeout:
                    st.error("Quiz generation timed out — try fewer questions.")
                except Exception as e:
                    st.error(f"Error: {e}")

    # --- Generated quiz ---
    if quiz is not None and not quiz.get("items"):
        st.error(
            "No questions could be parsed from the model's output. "
            "Try a different topic, or fewer questions."
        )
        if quiz.get("warnings"):
            with st.expander("Parser notes"):
                for w in quiz["warnings"]:
                    st.markdown(f"- {w}")

    elif quiz:
        items = quiz["items"]
        results = quiz.get("results")

        if quiz.get("parsed") != quiz.get("requested"):
            st.warning(
                f"Generated {quiz['parsed']} of {quiz['requested']} requested "
                "questions — the rest could not be parsed."
            )
        if quiz.get("warnings"):
            with st.expander("Parser notes"):
                for w in quiz["warnings"]:
                    st.markdown(f"- {w}")

        # Submitting and starting a new quiz redraw this slot in place rather
        # than calling st.rerun(). An explicit rerun re-creates the tab strip,
        # losing the client-side "which tab is open" state and dropping the
        # user back on Chat — mid-quiz, that means never seeing your score.
        quiz_slot = st.empty()

        def draw_form() -> bool:
            """Draw the answer form into the slot. Returns True if submitted."""
            with quiz_slot.container():
                with st.form(f"quiz_answers_{quiz['nonce']}"):
                    for i, item in enumerate(items):
                        st.markdown(f"**Q{i + 1}. {item['question']}**")
                        st.radio(
                            "Choose one",
                            options=list(range(len(item["options"]))),
                            format_func=lambda k, it=item: it["options"][k],
                            key=f"quiz_{module}_{quiz['nonce']}_{i}",
                            index=None,
                            label_visibility="collapsed",
                        )
                        st.divider()
                    return st.form_submit_button(
                        "Submit Quiz", type="primary", use_container_width=True
                    )

        def draw_results() -> bool:
            """Draw the marked results into the slot.

            Returns True if "New Quiz" was clicked. The reset itself happens in
            the caller: clearing the slot from inside its own container would
            destroy the container mid-write and corrupt the element tree.
            """
            results = quiz["results"]
            score = sum(1 for r in results if r["correct"])
            with quiz_slot.container():
                st.subheader(f"Score: {score} / {len(results)}")
                st.progress(score / len(results) if results else 0.0)
                st.divider()

                for i, (item, result) in enumerate(zip(items, results), 1):
                    icon = "✅" if result["correct"] else "❌"
                    st.markdown(f"{icon} **Q{i}. {item['question']}**")

                    selected = result["selected_index"]
                    your_answer = (
                        item["options"][selected] if selected is not None
                        else "_(not answered)_"
                    )
                    st.markdown(f"- Your answer: {your_answer}")
                    st.markdown(
                        f"- Correct answer: **{item['options'][item['correct_index']]}**"
                    )

                    # Provenance: the chunk this question was drawn from.
                    chunk_text = item.get("source_chunk_text")
                    if chunk_text:
                        label = item.get("source_file") or "source"
                        page = item.get("page_or_slide")
                        if page:
                            label += f" — page/slide {page}"
                        with st.expander(f"📎 From {label}"):
                            st.markdown(chunk_text)
                    else:
                        st.caption("No source chunk recorded for this question.")
                    st.divider()

                return st.button("New Quiz", use_container_width=True,
                                 key=f"quiz_new_{module}_{quiz['nonce']}")

        def reset_quiz():
            st.session_state.quiz.pop(module, None)
            quiz_slot.empty()
            st.info("Quiz cleared — generate a new one above.")

        if results is None:
            if draw_form():
                # Marked in Python by comparing indices — the LLM is never
                # asked whether an answer is right. The radio values survive in
                # session_state after the form is replaced below.
                marked = []
                for i, item in enumerate(items):
                    selected = st.session_state.get(
                        f"quiz_{module}_{quiz['nonce']}_{i}"
                    )
                    marked.append({
                        "question": item["question"],
                        "selected_index": selected,
                        "correct_index": item["correct_index"],
                        "correct": selected == item["correct_index"],
                        "source_file": item.get("source_file"),
                    })
                quiz["results"] = marked

                # Signal logging is best-effort: the score is already computed,
                # so a failure here must not lose the student's marking.
                try:
                    post_quiz_signals(module, quiz["topic"], marked)
                except Exception as e:
                    st.warning(f"Score recorded locally, but signal logging failed: {e}")

                # Swap the form for the results in the same run, so the open
                # tab stays put.
                if draw_results():
                    reset_quiz()
        else:
            if draw_results():
                reset_quiz()


# ---------------------------------------------------------------------------
# Flashcards tab — term on the front, definition on the back, one at a time
# ---------------------------------------------------------------------------
with tab_cards:
    deck = st.session_state.flashcards.get(module)

    # --- Config form ---
    with st.form("flashcard_config"):
        card_topic_input = st.text_input(
            "Topic",
            value=(deck or {}).get("topic", ""),
            placeholder="e.g. NLP terminology",
            help="Cards are retrieved and written about this topic.",
            key="flashcard_topic_input",
        )
        n_cards = st.radio(
            "Number of cards", [5, 10, 20], index=1, horizontal=True
        )
        generate_deck = st.form_submit_button(
            "Generate Flashcards", type="primary", use_container_width=True
        )

    if generate_deck:
        card_topic = card_topic_input.strip()
        if not card_topic:
            st.warning("Enter a topic first.")
        else:
            with st.spinner(
                f"Generating {n_cards} cards on '{card_topic}'... "
                "(this may take a few minutes on CPU)"
            ):
                try:
                    data = generate_flashcards(card_topic, module, n_cards)
                    st.session_state.flashcards[module] = {
                        "topic": card_topic,
                        "cards": data.get("cards", []),
                        "requested": data.get("requested"),
                        "parsed": data.get("parsed"),
                        "generation_time_s": data.get("generation_time_s"),
                        "warnings": data.get("warnings", []),
                        # Namespaces this deck's widget keys so a previous
                        # deck's reveal state cannot bleed into it.
                        "nonce": str(time.time()),
                        "index": 0,
                        "revealed": False,
                        "ratings": [],
                    }
                    deck = st.session_state.flashcards[module]
                except requests.exceptions.HTTPError as e:
                    st.error(
                        f"Flashcard generation failed: {e.response.status_code} "
                        f"— {e.response.text}"
                    )
                except requests.exceptions.Timeout:
                    st.error("Flashcard generation timed out — try fewer cards.")
                except Exception as e:
                    st.error(f"Error: {e}")

    # --- Deck ---
    if deck is not None and not deck.get("cards"):
        st.error(
            "No cards could be parsed from the model's output. "
            "Try a different topic, or fewer cards."
        )
        if deck.get("warnings"):
            with st.expander("Parser notes"):
                for w in deck["warnings"]:
                    st.markdown(f"- {w}")

    elif deck:
        cards = deck["cards"]
        total = len(cards)
        idx = deck["index"]

        if deck.get("parsed") != deck.get("requested"):
            st.warning(
                f"Generated {deck['parsed']} of {deck['requested']} requested "
                "cards — the rest could not be parsed."
            )
        if deck.get("warnings"):
            with st.expander("Parser notes"):
                for w in deck["warnings"]:
                    st.markdown(f"- {w}")

        # Reveals and ratings redraw these slots in place instead of calling
        # st.rerun(). An explicit rerun re-creates the tab strip, which loses
        # the client-side "which tab is open" state and drops the user back on
        # Chat — so going through a deck kicked you out on every click.
        card_slot = st.empty()
        summary_slot = st.empty()

        def draw_card(i: int, revealed: bool):
            """Draw card i into card_slot. Returns (reveal, got_it, missed)."""
            card = cards[i]
            with card_slot.container():
                st.caption(f"Card {i + 1} of {total}")
                st.progress(i / total)
                st.markdown(f"### {card['term']}")

                if not revealed:
                    clicked = st.button(
                        "Reveal", type="primary", use_container_width=True,
                        key=f"fc_reveal_{module}_{deck['nonce']}_{i}",
                    )
                    return clicked, False, False

                st.success(card["definition"])

                chunk_text = card.get("source_chunk_text")
                if chunk_text:
                    label = card.get("source_file") or "source"
                    page = card.get("page_or_slide")
                    if page:
                        label += f" — page/slide {page}"
                    with st.expander(f"📎 From {label}"):
                        st.markdown(chunk_text)
                else:
                    st.caption("No source chunk recorded for this card.")

                st.divider()
                st.caption("How well did you know this?")
                col_known, col_unknown = st.columns(2)
                return (
                    False,
                    col_known.button(
                        "✅ Got it", use_container_width=True,
                        key=f"fc_known_{module}_{deck['nonce']}_{i}",
                    ),
                    col_unknown.button(
                        "❌ Didn't know", use_container_width=True,
                        key=f"fc_unknown_{module}_{deck['nonce']}_{i}",
                    ),
                )

        def draw_summary() -> bool:
            """Draw the end-of-deck summary. Returns True if "New Deck" was
            clicked; the reset happens in the caller, because clearing the slot
            from inside its own container corrupts the element tree."""
            ratings = deck["ratings"]
            known = sum(1 for r in ratings if r["known"])
            unknown = len(ratings) - known
            with summary_slot.container():
                st.subheader(f"Deck complete — {known} known, {unknown} unknown")
                st.progress(1.0)
                col_a, col_b = st.columns(2)
                col_a.metric("✅ Got it", known)
                col_b.metric("❌ Didn't know", unknown)
                st.divider()
                for i, rating in enumerate(ratings, 1):
                    icon = "✅" if rating["known"] else "❌"
                    st.markdown(f"{icon} **{i}. {rating['term']}**")
                st.divider()
                return st.button("New Deck", use_container_width=True,
                                 key=f"fc_new_{module}_{deck['nonce']}")

        def reset_deck():
            st.session_state.flashcards.pop(module, None)
            summary_slot.empty()
            st.info("Deck cleared — generate a new one above.")

        if idx < total:
            reveal, got_it, missed = draw_card(idx, deck["revealed"])

            if reveal:
                deck["revealed"] = True
                # Redraw the same card, now showing its back. Same run, so the
                # tab the user is looking at stays put.
                draw_card(idx, True)

            elif got_it or missed:
                rating = {
                    "term": cards[idx]["term"],
                    "known": bool(got_it),
                    "source_file": cards[idx].get("source_file"),
                }
                deck["ratings"].append(rating)
                deck["index"] = idx + 1
                deck["revealed"] = False

                # Best-effort: the rating is already recorded locally, so a
                # logging failure must not stop the student moving on.
                try:
                    post_flashcard_signals(module, deck["topic"], [rating])
                except Exception as e:
                    st.warning(
                        f"Rating kept locally, but signal logging failed: {e}"
                    )

                if deck["index"] < total:
                    draw_card(deck["index"], False)
                else:
                    card_slot.empty()
                    if draw_summary():
                        reset_deck()
        else:
            if draw_summary():
                reset_deck()


# ---------------------------------------------------------------------------
# Progress tab — per-signal-type confidence, deliberately NOT one blended score
# ---------------------------------------------------------------------------
with tab_progress:
    # Refreshing redraws this slot in place rather than calling st.rerun(),
    # which would re-create the tab strip and bounce the user back to Chat.
    progress_slot = st.empty()

    def draw_progress(pass_id: int = 0) -> bool:
        """Render the confidence panel. Returns True if Refresh was clicked.

        `pass_id` namespaces the Refresh button's key. A redraw happens in the
        same script run as the first draw, and reusing the key there raises
        DuplicateWidgetID.
        """
        with progress_slot.container():
            try:
                data = fetch_confidence(module)
            except Exception as e:
                st.error(f"Could not load progress: {e}")
                return False

            if data["total_signals"] == 0:
                st.info(
                    f"No activity recorded for **{module}** yet. Take a quiz or "
                    "work through a flashcard deck, and your results will "
                    "appear here."
                )
                return st.button("Refresh", use_container_width=True,
                                 key=f"prog_refresh_{module}_{pass_id}")

            st.caption(
                f"{data['total_signals']} signals recorded for **{module}**. "
                "The three measures are kept separate on purpose — they are not "
                "equally reliable, so a single blended score would be misleading."
            )

            # --- Quiz: measured ---
            quiz_agg = data["quiz"]
            st.subheader("📝 Quiz — measured")
            st.caption("Marked automatically by comparing your answer to the key.")
            if quiz_agg["count"] == 0:
                st.markdown("_No quiz questions answered yet._")
            else:
                c1, c2, c3 = st.columns(3)
                c1.metric("Questions", quiz_agg["count"])
                c2.metric("Correct", quiz_agg["correct"])
                c3.metric("Accuracy", f"{quiz_agg['accuracy_pct']}%")
                st.progress((quiz_agg["accuracy_pct"] or 0) / 100)

                if data["by_quiz_topic"]:
                    with st.expander("By quiz subject"):
                        for row in data["by_quiz_topic"]:
                            st.markdown(
                                f"**{row['subject']}** — {row['correct']}/"
                                f"{row['count']} correct ({row['accuracy_pct']}%)"
                            )
                        st.caption(SUBJECT_MATCH_NOTE)

            st.divider()

            # --- Flashcards: self-reported ---
            fc_agg = data["flashcard"]
            st.subheader("🗂️ Flashcards — self-reported")
            st.caption(
                "Based on your own 'Got it' / 'Didn't know' ratings, not on any "
                "check of your answer."
            )
            if fc_agg["count"] == 0:
                st.markdown("_No flashcards rated yet._")
            else:
                c1, c2, c3 = st.columns(3)
                c1.metric("Cards rated", fc_agg["count"])
                c2.metric("Known", fc_agg["known"])
                c3.metric("Known %", f"{fc_agg['known_pct']}%")
                st.progress((fc_agg["known_pct"] or 0) / 100)

                if data["by_flashcard_topic"]:
                    with st.expander("By deck subject"):
                        for row in data["by_flashcard_topic"]:
                            st.markdown(
                                f"**{row['subject']}** — {row['known']}/"
                                f"{row['count']} known ({row['known_pct']}%)"
                            )
                        st.caption(SUBJECT_MATCH_NOTE)

            st.divider()

            # --- Sentiment: inferred ---
            s_agg = data["sentiment"]
            st.subheader("💬 Question tone — inferred")
            st.caption(
                "Guessed by a sentiment model from how your questions are "
                "phrased. The weakest of the three: it never sees whether you "
                "understood anything."
            )
            if s_agg["count"] == 0:
                st.markdown("_No questions asked yet._")
            else:
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Questions", s_agg["count"])
                c2.metric("Negative", s_agg["negative"])
                c3.metric("Neutral", s_agg["neutral"])
                c4.metric("Positive", s_agg["positive"])
                if s_agg["inconclusive"]:
                    st.caption(
                        f"{s_agg['inconclusive']} question(s) scored below the "
                        f"{s_agg['score_threshold']} confidence threshold and "
                        "are not counted, whatever label they were given — at "
                        "that confidence the model does not reliably tell "
                        "frustration from a plainly worded question."
                    )

            st.divider()
            return st.button("Refresh", use_container_width=True,
                             key=f"prog_refresh_{module}_{pass_id}")

    if draw_progress(0):
        # Redraw in place with fresh data; no st.rerun(), so the tab stays put.
        draw_progress(1)

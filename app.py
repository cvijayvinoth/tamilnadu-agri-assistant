"""Tamil Nadu Agriculture Schemes Assistant - Streamlit entry point.

Run:  streamlit run app.py
"""

from __future__ import annotations

import streamlit as st

import config
from src import catalogue as cat
from src import chain, guardrails, ingestion, retrieval, taxonomy, voice
from src.ui import logo, speaker, theme
import streamlit.components.v1 as components

st.set_page_config(
    page_title=config.APP_TITLE,
    page_icon=logo.favicon_data_uri(),
    layout="wide",
    initial_sidebar_state="expanded",
)
theme.apply_theme()
taxonomy.init_db()  # idempotent: creates the vocabulary tables on first run


# ------------------------------------------------------------- session state
def init_state() -> None:
    """Chat history lives here, so it survives every rerun until a page refresh."""
    defaults = {
        "messages": [],          # [{role, content, sources, meta}]
        "lang_history": [],      # LangChain message objects for the chain
        "pending_question": "",  # text pushed in from the mic, editable
        "last_audio_id": None,   # stops the same clip being transcribed twice
        "voice_note": None,
        "followups": [],
        "feedback": {},
        "speak_id": None,        # which message index to read aloud this rerun
        "tts_on": False,         # auto-read newly generated answers
        "total_tokens": 0,
        "turn_count": 0,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


init_state()


@st.cache_resource(show_spinner=False)
def _load_store(version: str):
    """FAISS store, cached per on-disk index version.

    The `version` argument is part of the cache key, not used in the body: when
    scripts/bootstrap.py rebuilds the index, the fingerprint changes and the
    running app picks up the new store on the next rerun - no restart needed.
    """
    return ingestion.load_index()


def get_store():
    return _load_store(ingestion.index_version())


def refresh_store():
    _load_store.clear()
    return get_store()


# -------------------------------------------------------------------- header
store = get_store()
stats = ingestion.index_stats()
kb_ready = store is not None and stats.get("chunk_count", 0) > 0

theme.hero(
    logo.logo_html(40) + " " + config.APP_TITLE,
    config.APP_TAGLINE,
    chips=[
        "🎙️ Live voice — no file upload",
        "⚡ Streaming answers",
        "🔎 Vector search (MMR)",
        "📄 Page-level citations",
        f"v{config.APP_VERSION}",
    ],
)

# ------------------------------------------------------------------- sidebar
with st.sidebar:
    st.markdown("### ⚙️ Assistant settings")

    mode = config.DEFAULT_RETRIEVAL_MODE
    st.caption(
        "🔎 **Vector search (MMR)** — semantic retrieval with diversity "
        "re-ranking, so the passages you get are relevant *and* "
        "non-repetitive."
    )
    top_k = st.slider("Passages to retrieve (k)", 2, 10, config.DEFAULT_TOP_K)

    st.markdown("#### 🔍 Narrow the context")
    district_counts = taxonomy.get_terms_with_counts("district")
    category_counts = taxonomy.get_terms_with_counts("category")
    district_values = [v for v, _ in district_counts]
    category_values = [v for v, _ in category_counts]
    district_labels = {v: f"{v} ({n})" if v != "All" else v
                        for v, n in district_counts}
    category_labels = {v: f"{v} ({n})" if v != "All" else v
                        for v, n in category_counts}
    district = st.selectbox(
        "District", district_values, format_func=lambda v: district_labels[v],
    )
    category = st.selectbox(
        "Category", category_values, format_func=lambda v: category_labels[v],
    )
    if district != "All" or category != "All":
        st.caption(
            "Filters apply to documents tagged at ingestion time. A "
            "**(0)** count means no indexed document currently carries that "
            "tag — the assistant will fall back to an unfiltered search."
        )

    theme.divider()

    st.markdown("### 📊 Knowledge base")
    c1, c2 = st.columns(2)
    with c1:
        theme.metric_card(stats.get("chunk_count", 0), "Chunks")
    with c2:
        theme.metric_card(stats.get("file_count", 0), "Documents")

    status_pills = []
    status_pills.append(("Index ready", "ok") if kb_ready else ("Index empty", "warn"))
    status_pills.append(
        ("API key loaded", "ok") if config.api_key_present() else ("No API key", "err")
    )
    if config.langsmith_enabled():
        status_pills.append(("LangSmith tracing on", "info"))
    theme.pills(status_pills)

    theme.divider()

    st.markdown("### 🎙️ Voice engine")
    voice_ok, voice_msg = voice.engine_ready()
    theme.pills([("Whistle ready (on-device)", "ok") if voice_ok
                 else ("Voice unavailable", "err")])
    if not voice_ok:
        st.caption(voice_msg)
    else:
        st.caption("Runs locally — no audio leaves this machine, zero API cost.")
    st.caption("🗣️ Speech input is **English only**. Type in Tamil for Tamil answers.")

    st.markdown("### 🔊 Read answers aloud")
    if not config.TTS_ENABLED:
        st.caption("Read-aloud is disabled in config.")
        tts_rate = config.TTS_RATE
    else:
        st.session_state.tts_on = st.toggle(
            "Auto-read new answers",
            value=st.session_state.tts_on,
            help="Speaks each answer as soon as it finishes generating.",
        )
        tts_rate = st.slider(
            "Speaking speed", min_value=0.6, max_value=1.4,
            value=float(config.TTS_RATE), step=0.05,
        )
        st.caption(
            "Spoken by your **browser** — the text is never sent anywhere, "
            "so this stays free and private. Tamil answers need a ta-IN "
            "voice installed in Windows."
        )
        with st.expander("Installed voices", expanded=False):
            components.html(speaker.voice_probe_html(), height=90)

    theme.divider()
    if st.session_state.messages:
        st.caption(
            f"💬 {st.session_state.turn_count} turn(s) this session · "
            f"~{st.session_state.total_tokens:,} tokens"
        )
    if st.button("🧹 Clear conversation", width="stretch"):
        st.session_state.messages = []
        st.session_state.lang_history = []
        st.session_state.followups = []
        st.session_state.speak_id = None   # else a stale index re-speaks later
        st.session_state.turn_count = 0
        st.session_state.total_tokens = 0
        st.rerun()

# ---------------------------------------------------------------------- tabs
tab_chat, tab_catalogue, tab_sources, tab_status = st.tabs([
    "💬 Assistant",
    "📚 Scheme Catalogue",
    "📥 Sources & Ingestion",
    "📊 Knowledge Base",
])


def render_sources(sources: list[dict], note: str = "") -> None:
    """Compact citation line plus an expander with the exact passages used."""
    if not sources:
        return
    if note:
        st.caption(f"ℹ️ {note}")
    st.markdown(
        f'<div class="agri-sources"><b>Sources:</b> '
        f'{retrieval.group_sources(sources)}</div>',
        unsafe_allow_html=True,
    )
    with st.expander(f"📄 View the {len(sources)} passage(s) behind this answer"):
        for s in sources:
            meta = f"[{s['n']}] {s['file']} — page {s['page']}"
            if s.get("scheme"):
                meta += f" · {s['scheme']}"
            theme.passage(meta, s["text"][:1500])
            if s.get("url"):
                st.caption(f"🔗 {s['url']}")


def ask(question: str) -> None:
    """One full turn: guard -> rewrite -> retrieve -> stream -> remember."""
    ok, msg = guardrails.validate_question(question)
    if not ok:
        st.warning(msg)
        return

    if not config.api_key_present():
        st.error("No OpenAI API key found. Add OPENAI_API_KEY to your .env file.")
        return

    current_store = get_store()
    if current_store is None:
        st.warning(
            "The knowledge base is empty. Open **📥 Sources & Ingestion** and index "
            "a document first."
        )
        return

    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user", avatar="🧑‍🌾"):
        st.markdown(question)

    with st.chat_message("assistant", avatar=logo.favicon_data_uri()):
        # Follow-ups are rewritten standalone, else they embed poorly.
        with st.status("Thinking…", expanded=False) as status:
            status.update(label="Understanding the question…")
            search_query = chain.rewrite_query(question, st.session_state.lang_history)
            if search_query != question:
                st.caption(f"🔁 Searched as: _{search_query}_")

            status.update(label=f"Searching with {mode} retrieval…")
            result = retrieval.retrieve(
                current_store, search_query, mode=mode, k=top_k,
                district=district, category=category,
            )
            status.update(label="Composing a grounded answer…", state="complete")

        if result.empty:
            st.warning(
                "I could not find anything relevant in the indexed documents. "
                "Try rephrasing, widening the filters, or ingesting more sources."
            )
            st.session_state.messages.append({
                "role": "assistant",
                "content": "No relevant passages were found in the knowledge base.",
                "sources": [],
            })
            return

        context = retrieval.format_context(
            [d for d in result.docs]
        )
        context = guardrails.sanitize_passage(context, limit=12000)

        try:
            answer = st.write_stream(
                chain.stream_answer(question, context, st.session_state.lang_history)
            )
        except Exception as exc:
            st.error(f"The answer could not be generated: {exc}")
            return

        sources = retrieval.to_sources(result.docs)
        render_sources(sources, result.note)

        report = guardrails.grounding_report(answer, len(sources))
        badges = []
        if report["is_refusal"]:
            badges.append(("Not found in sources", "warn"))
        elif report["is_grounded"]:
            badges.append((f"{report['citation_count']} citation(s)", "ok"))
        else:
            badges.append(("Uncited answer — verify manually", "warn"))
        if report["invalid_citations"]:
            badges.append(("Invalid citation number", "err"))
        badges.append((f"{mode} · k={top_k}", "info"))
        theme.pills(badges)

        # Read-aloud for the turn that was just generated. Auto-play is
        # opt-in via the sidebar toggle; otherwise the Play button starts it.
        if config.TTS_ENABLED:
            components.html(
                speaker.speak_html(
                    answer, rate=tts_rate,
                    autoplay=bool(st.session_state.tts_on),
                    nonce=f"live-{len(st.session_state.messages)}",
                ),
                height=78,
            )

        st.download_button(
            "⬇️ Download answer (.md)",
            data=f"# {question}\n\n{answer}\n\n## Sources\n"
                 + "\n".join(f"- [{s['n']}] {s['file']} p.{s['page']}" for s in sources),
            file_name="tn-agri-answer.md",
            mime="text/markdown",
            key=f"dl_{len(st.session_state.messages)}",
        )

    # Persist the turn so it replays on every rerun until the page is refreshed.
    from langchain_core.messages import AIMessage, HumanMessage

    st.session_state.lang_history.extend([
        HumanMessage(content=question),
        AIMessage(content=answer),
    ])
    st.session_state.messages.append({
        "role": "assistant", "content": answer,
        "sources": sources, "note": result.note,
    })
    st.session_state.turn_count += 1
    st.session_state.total_tokens += (len(context) + len(answer)) // 4
    st.session_state.followups = chain.suggest_followups(question, answer)


SAMPLE_QUESTIONS = [
    "What is the subsidy for drip irrigation?",
    "Tell me about the Tamil Nadu Millet Mission",
    "Which schemes support oilseed farmers?",
    "What does the KAVIADP programme cover?",
]

with tab_chat:
    if not st.session_state.messages:
        st.markdown("#### 👋 Ask about any Tamil Nadu agriculture scheme")
        st.caption(
            "Type below, or press the mic and **speak** — your words appear in the "
            "box as editable text before anything is sent."
        )
        chip_cols = st.columns(len(SAMPLE_QUESTIONS))
        for col, sample in zip(chip_cols, SAMPLE_QUESTIONS):
            with col:
                if st.button(sample, key=f"s_{sample}", width="stretch"):
                    st.session_state.pending_question = sample
                    st.rerun()

    # ---- replay the conversation (survives reruns until a page refresh) ----
    for i, m in enumerate(st.session_state.messages):
        avatar = "🧑‍🌾" if m["role"] == "user" else logo.favicon_data_uri()
        with st.chat_message(m["role"], avatar=avatar):
            st.markdown(m["content"])
            if m["role"] == "assistant" and m.get("sources"):
                render_sources(m["sources"], m.get("note", ""))
            # Replayed answers get a manual Play button only -- never autoplay,
            # or every rerun would re-speak the whole conversation.
            if m["role"] == "assistant" and config.TTS_ENABLED and m.get("content"):
                if st.button("🔊 Read aloud", key=f"tts_{i}"):
                    st.session_state.speak_id = i
                if st.session_state.speak_id == i:
                    components.html(
                        speaker.speak_html(
                            m["content"], rate=tts_rate,
                            autoplay=True, nonce=f"hist-{i}",
                        ),
                        height=78,
                    )

    # ------------------------- voice status line -------------------------
    # The microphone now lives *inside* st.chat_input (accept_audio=True), so
    # there is no separate recorder widget here. Only its status is shown.
    if st.session_state.voice_note:
        st.caption(st.session_state.voice_note)
    elif voice_ok:
        st.caption(
            "🔒 Use the 🎙️ mic inside the message box below. Audio never leaves "
            "your computer — the transcript appears here for review first."
        )

    # The transcript lands here as editable text, never auto-submitted.
    if st.session_state.pending_question:
        with st.form("voice_form", clear_on_submit=False):
            edited = st.text_area(
                "📝 Review and edit before sending",
                value=st.session_state.pending_question,
                height=90,
            )
            fc1, fc2 = st.columns([1, 5])
            with fc1:
                send = st.form_submit_button("Ask ▶", type="primary")
            with fc2:
                discard = st.form_submit_button("Discard")
        if send and edited.strip():
            st.session_state.pending_question = ""
            st.session_state.voice_note = None
            ask(edited.strip())
            st.rerun()
        if discard:
            st.session_state.pending_question = ""
            st.session_state.voice_note = None
            st.rerun()

    # accept_audio puts the mic *inside* the input bar, and st.chat_input is
    # pinned to the bottom of the viewport -- so typing and speaking are always
    # visible together. 16 kHz mono is exactly what Whistle wants, no resampling.
    submitted = st.chat_input(
        "Ask about a scheme — in English or Tamil…",
        accept_audio=voice_ok,
        audio_sample_rate=config.VOICE_SAMPLE_RATE,
        disabled=not kb_ready,
    )
    if not kb_ready:
        st.info(
            "📥 The knowledge base is empty. Go to **Sources & Ingestion** and index "
            "the bundled PDFs to start asking questions."
        )

    if submitted is not None:
        # With accept_audio=True this is a ChatInputValue, not a str. It is a
        # mapping that always carries a "text" key, so truthiness is useless --
        # the fields must be inspected explicitly.
        if isinstance(submitted, str):
            text_in, audio_in = submitted, None
        else:
            text_in = submitted.text or ""
            audio_in = submitted.audio if voice_ok else None

        if audio_in is not None:
            raw = audio_in.getvalue()
            clip_id = voice.fingerprint(raw)
            # Streamlit reruns on every interaction; never re-transcribe a clip.
            if clip_id != st.session_state.last_audio_id:
                st.session_state.last_audio_id = clip_id
                with st.spinner("🎧 Transcribing on-device…"):
                    tr = voice.transcribe_bytes(raw)
                if tr.ok:
                    # Anything typed alongside the clip is kept as a prefix.
                    spoken = tr.text if not text_in.strip() else f"{text_in.strip()} {tr.text}"
                    st.session_state.pending_question = spoken
                    st.session_state.voice_note = (
                        f"✅ Heard: “{tr.text}” · {tr.elapsed_s:.1f}s"
                        + (f" · {tr.decode_tps:.0f} tok/s" if tr.decode_tps else "")
                    )
                else:
                    st.session_state.voice_note = f"⚠️ {tr.error}"
                st.rerun()
        elif text_in.strip():
            ask(text_in.strip())
            st.rerun()

    # ------------------------- follow-up chips -------------------------
    if st.session_state.followups and st.session_state.messages:
        st.caption("💡 Suggested follow-ups")
        fcols = st.columns(len(st.session_state.followups))
        for col, fu in zip(fcols, st.session_state.followups):
            with col:
                if st.button(fu, key=f"fu_{fu}", width="stretch"):
                    st.session_state.followups = []
                    ask(fu)
                    st.rerun()


# ------------------------------------------------------- scheme catalogue tab
with tab_catalogue:
    st.markdown("### 📚 Curated scheme catalogue")
    st.caption(
        "Browse the vetted scheme registry from `data/sources.csv`, then send any "
        "official source straight to the ingestion pipeline."
    )

    cat_df = cat.load_catalogue()
    if cat_df.empty:
        st.warning("No catalogue found at data/sources.csv.")
    else:
        cstats = cat.catalogue_stats(cat_df)
        m1, m2, m3 = st.columns(3)
        with m1:
            theme.metric_card(cstats["schemes"], "Schemes")
        with m2:
            theme.metric_card(cstats["departments"], "Departments")
        with m3:
            theme.metric_card(cstats["urls"], "Source URLs")

        theme.divider()
        f1, f2, f3 = st.columns([2, 2, 2])
        with f1:
            search = st.text_input("Search schemes", placeholder="millet, drip, subsidy…")
        with f2:
            dept = st.selectbox("Department", cat.departments(cat_df))
        with f3:
            stype = st.selectbox("Scheme type", cat.scheme_types(cat_df))

        view = cat.filter_catalogue(cat_df, search, dept, stype)
        st.caption(f"Showing **{len(view)}** of {len(cat_df)} schemes")
        st.dataframe(
            view[["scheme", "scheme_type", "what_it_covers", "evidence_level"]],
            width="stretch",
            hide_index=True,
        )

        theme.divider()
        st.markdown("#### 🔗 Ingest an official source")
        urls = cat.candidate_urls(view if not view.empty else cat_df)
        if urls:
            labels = [
                f"{'📕' if u['is_pdf'] else '🌐'} {u['scheme'][:60]} — {u['url'][:60]}…"
                for u in urls
            ]
            picked = st.selectbox(
                "Choose a source", range(len(urls)), format_func=lambda i: labels[i]
            )
            chosen = urls[picked]
            st.caption(f"Evidence level: **{chosen['evidence_level'] or 'n/a'}**")
            if st.button("📥 Ingest this source", type="primary"):
                if not config.api_key_present():
                    st.error("An OpenAI API key is required to embed documents.")
                else:
                    with st.status("Fetching and indexing…", expanded=True) as s:
                        try:
                            s.write("Downloading source…")
                            docs = ingestion.load_url(chosen["url"])
                            s.write(f"Extracted {len(docs)} page(s). Embedding…")
                            rep = ingestion.build_or_update_index(
                                docs, scheme=chosen["scheme"]
                            )
                            if rep.ok:
                                s.update(label="Indexed successfully", state="complete")
                                st.success(
                                    f"Added {rep.added_chunks} chunks in "
                                    f"{rep.elapsed_s:.1f}s."
                                )
                                refresh_store()
                            else:
                                s.update(label="Ingestion failed", state="error")
                                for e in rep.errors:
                                    st.error(e)
                        except Exception as exc:
                            s.update(label="Ingestion failed", state="error")
                            st.error(f"Could not ingest that URL: {exc}")
        else:
            st.info("No ingestible URLs in the current selection.")


# -------------------------------------------------------- sources/ingest tab
with tab_sources:
    st.markdown("### 📥 Build the knowledge base")
    st.caption("Everything indexed here becomes citable evidence for the assistant.")

    if not config.api_key_present():
        st.error("Add `OPENAI_API_KEY` to your `.env` file before indexing.")

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("#### 📄 Upload PDFs")
        ups = st.file_uploader(
            "Scheme documents, Government Orders, budget speeches",
            type=["pdf"],
            accept_multiple_files=True,
        )
        tag_d = st.selectbox("Tag district", taxonomy.get_districts(), key="tag_d")
        tag_c = st.selectbox("Tag category", taxonomy.get_categories(), key="tag_c")

        if ups and st.button("Index uploaded PDFs", type="primary"):
            with st.status("Indexing…", expanded=True) as s:
                docs, warns = [], []
                for f in ups:
                    dest = config.UPLOAD_DIR / f.name
                    dest.write_bytes(f.getvalue())
                    try:
                        pages = ingestion.load_pdf(dest)
                    except RuntimeError as exc:
                        st.error(str(exc))
                        continue
                    scanned = ingestion.ocr_page_count(pages)
                    if scanned:
                        warns.append(
                            f"{f.name}: {scanned} page(s) look scanned — "
                            "text extraction may be incomplete."
                        )
                    docs.extend(pages)
                    s.write(f"✓ {f.name}: {len(pages)} page(s)")

                if docs:
                    rep = ingestion.build_or_update_index(
                        docs, district=tag_d, category=tag_c
                    )
                    for w in warns + rep.warnings:
                        st.warning(w)
                    if rep.ok:
                        s.update(label="Indexed", state="complete")
                        st.success(f"Added {rep.added_chunks} chunks.")
                        refresh_store()
                    else:
                        s.update(label="Failed", state="error")
                        for e in rep.errors:
                            st.error(e)
                else:
                    s.update(label="Nothing indexed", state="error")

    with col_b:
        st.markdown("#### 🌐 Ingest a URL")
        url = st.text_input("Public PDF or web page", placeholder="https://…")
        if url and st.button("Fetch & index URL"):
            with st.status("Fetching…", expanded=True) as s:
                try:
                    docs = ingestion.load_url(url.strip())
                    s.write(f"Extracted {len(docs)} page(s).")
                    rep = ingestion.build_or_update_index(docs)
                    if rep.ok:
                        s.update(label="Indexed", state="complete")
                        st.success(f"Added {rep.added_chunks} chunks.")
                        refresh_store()
                    else:
                        s.update(label="Failed", state="error")
                        for e in rep.errors:
                            st.error(e)
                except Exception as exc:
                    s.update(label="Failed", state="error")
                    st.error(f"Could not fetch that URL: {exc}")

        theme.divider()
        st.markdown("#### 🌱 Bundled documents")
        seeds = sorted(config.RAW_DATA_DIR.glob("*.pdf"))
        if seeds:
            for p in seeds:
                st.caption(f"• {p.name} ({p.stat().st_size // 1024} KB)")
            if st.button("Index bundled PDFs from data/raw"):
                with st.status("Indexing bundled PDFs…", expanded=True) as s:
                    rep = ingestion.ingest_seed_pdfs()
                    for w in rep.warnings:
                        st.warning(w)
                    if rep.ok:
                        s.update(label="Indexed", state="complete")
                        st.success(
                            f"Added {rep.added_chunks} chunks from "
                            f"{len(rep.files)} file(s)."
                        )
                        refresh_store()
                    else:
                        s.update(label="Failed", state="error")
                        for e in rep.errors:
                            st.error(e)
        else:
            st.caption("No PDFs in data/raw.")


# ---------------------------------------------------------------- status tab
with tab_status:
    st.markdown("### 📊 Knowledge base status")
    live = ingestion.index_stats()

    s1, s2, s3, s4 = st.columns(4)
    with s1:
        theme.metric_card(live.get("chunk_count", 0), "Chunks")
    with s2:
        theme.metric_card(live.get("file_count", 0), "Documents")
    with s3:
        theme.metric_card(live.get("chunk_size", config.CHUNK_SIZE), "Chunk size")
    with s4:
        theme.metric_card(live.get("embedding_dim", config.EMBEDDING_DIM), "Vector dims")

    theme.divider()
    if live.get("per_file"):
        st.markdown("#### Indexed documents")
        rows = [
            {"Document": k, "Chunks": v}
            for k, v in sorted(live["per_file"].items(), key=lambda x: -x[1])
        ]
        st.dataframe(rows, width="stretch", hide_index=True)
    else:
        st.info("Nothing indexed yet — add documents from the Sources tab.")

    st.markdown("#### Configuration")
    st.json({
        "llm_model": config.LLM_MODEL,
        "embedding_model": config.EMBEDDING_MODEL,
        "embedding_dim": config.EMBEDDING_DIM,
        "chunk_size": config.CHUNK_SIZE,
        "chunk_overlap": config.CHUNK_OVERLAP,
        "retrieval_modes": list(config.RETRIEVAL_MODES),
        "speech_engine": "Whistle (on-device, English only)",
        "last_updated": live.get("updated_at", "never"),
    })

    theme.divider()
    with st.expander("⚠️ Danger zone"):
        st.caption(
            "Deleting the index removes all embeddings. Your original PDFs in "
            "data/raw and data/uploads are NOT touched, so you can always re-index."
        )
        confirm = st.text_input("Type DELETE to confirm")
        if st.button("Delete vector index", disabled=confirm != "DELETE"):
            ingestion.reset_index()
            refresh_store()
            st.success("Index deleted. Re-ingest to rebuild.")
            st.rerun()

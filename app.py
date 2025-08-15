import os, time
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import streamlit as st
from typing import List, Dict, Any
from rag_pipeline.transcribe import transcribe_audio
from rag_pipeline.diarize import diarize
from rag_pipeline.chunking import chunk_by_time
from rag_pipeline.index import Indexer
from rag_pipeline.search import hybrid_rank
from rag_pipeline.generate import generate_answer
from rag_pipeline.utils import mmss, ensure_dir, hash_file

st.set_page_config(page_title="Podcast RAG Search", page_icon="🎙️", layout="wide")

st.title("🎙️ Audio‑to‑Text RAG for Podcast Search")
st.caption("Upload podcasts → Transcribe with timestamps → Build a searchable index → Ask topic questions (multi‑episode).")

with st.sidebar:
    st.header("Settings")
    collection_name = st.text_input("Collection name", value="podcasts")
    model_size = st.selectbox("Whisper model", options=["tiny", "base", "small"], index=1)
    window_s = st.slider("Chunk window (sec)", 20, 120, 45, step=5)
    overlap_s = st.slider("Chunk overlap (sec)", 0, 20, 7, step=1)
    top_k = st.slider("Top-k results", 3, 20, 8, step=1)
    st.divider()
    st.write("**LLM (optional):** set `GROQ_API_KEY` or `OPENAI_API_KEY` as env vars to enable generation.")
    st.write("**Diarization (optional):** set `HUGGINGFACE_TOKEN` to enable speaker diarization.")

uploads_dir = "uploads"
ensure_dir(uploads_dir)

indexer = Indexer(persist_dir="chroma")

st.subheader("1) Upload episodes")
files = st.file_uploader("Drop multiple podcast files (.mp3/.wav/.m4a)", type=["mp3","wav","m4a"], accept_multiple_files=True)

if "ingested_meta" not in st.session_state:
    st.session_state.ingested_meta = {}

if st.button("Ingest & Index", type="primary") and files:
    with st.spinner("Transcribing & indexing..."):
        t0 = time.time()
        corpus_texts = []
        corpus_ids = []
        from rank_bm25 import BM25Okapi

        for f in files:
            path = os.path.join(uploads_dir, f.name)
            with open(path, "wb") as out:
                out.write(f.read())
            ep_id = f"{f.name}-{hash_file(path)}"
            st.write(f"Processing **{f.name}** (id: `{ep_id}`)")

            tr = transcribe_audio(path, model_size=model_size, vad_filter=False)
            st.write(f"- Language: `{tr['language']}` | Duration: {tr['duration']:.1f}s | STT time: {tr['runtime_sec']:.1f}s")

            dia = diarize(path)
            if dia:
                st.write(f"- Diarization enabled: found {len(dia)} speaker segments")

            chunks = chunk_by_time(tr["segments"], window_s=window_s, overlap_s=overlap_s)
            st.write(f"- Created {len(chunks)} chunks with {window_s}s windows ({overlap_s}s overlap)")

            meta = {"filename": f.name, "has_diarization": bool(dia)}
            indexer.add_chunks(collection_name, chunks, episode_id=ep_id, meta_extra=meta)

            for i, c in enumerate(chunks):
                doc_id = f"{ep_id}:{i}"
                corpus_ids.append(doc_id)
                corpus_texts.append(c["text"])

            st.session_state.ingested_meta[ep_id] = {
                "filename": f.name,
                "duration": tr["duration"],
                "num_chunks": len(chunks),
            }

        bm25 = BM25Okapi([t.lower().split() for t in corpus_texts])
        st.session_state["bm25_model"] = bm25
        st.session_state["bm25_ids"] = corpus_ids
        st.success(f"✅ Done in {time.time()-t0:.1f}s. Indexed {len(corpus_ids)} chunks across {len(files)} episodes.")

st.subheader("2) Ask your question")
query = st.text_input("e.g., What did they say about transformers architecture?")

colA, colB = st.columns([1,1])
with colA:
    topk = st.slider("Retrieve top‑k", 3, 12, 8, step=1, key="retrieve_k")
with colB:
    gen = st.toggle("Generate answer (uses LLM if configured)", value=True)

if query:
    with st.spinner("Searching..."):
        t0 = time.time()
        res = indexer.query(collection_name, query, top_k=topk)
        ids = res.get("ids", [[]])[0]
        docs = res.get("documents", [[]])[0]
        metas = res.get("metadatas", [[]])[0]

        bm25 = st.session_state.get("bm25_model")
        corpus_ids = st.session_state.get("bm25_ids")
        if bm25 and corpus_ids and ids:
            ranked_ids = hybrid_rank(res, bm25, corpus_ids, query, top_k=topk)
            id_to_idx = {id_: i for i, id_ in enumerate(ids)}
            order = [id_to_idx[r] for r in ranked_ids if r in id_to_idx]
            if order:
                ids = [ids[i] for i in order]
                docs = [docs[i] for i in order]
                metas = [metas[i] for i in order]

        hits = [{"id": ids[i], "document": docs[i], "metadata": metas[i]} for i in range(len(ids))]

        st.write(f"🔎 Query latency: **{time.time()-t0:.2f}s**")
        for h in hits:
            ep = h["metadata"].get("episode_id", "episode")
            filename = h["metadata"].get("filename", "")
            stt = h["metadata"].get("start", 0.0)
            ent = h["metadata"].get("end", 0.0)
            with st.expander(f"{filename} [{mmss(stt)}–{mmss(ent)}] — {ep}"):
                st.write(h["document"])
                st.caption(f"Timecode: {stt:.1f}s – {ent:.1f}s | Episode ID: {ep}")
        st.divider()

        if gen and hits:
            answer = generate_answer(query, hits)
            st.subheader("Answer")
            st.write(answer)

st.divider()
st.subheader("Indexed episodes")
if st.session_state.ingested_meta:
    for ep, m in st.session_state.ingested_meta.items():
        st.write(f"- `{m['filename']}` — duration {m['duration']:.1f}s — chunks: {m['num_chunks']} — id: `{ep}`")
else:
    st.write("No episodes indexed yet.")

st.caption("Tip: Provide a Groq or OpenAI API key via environment variables to enable LLM answers. Without keys, the app returns extractive summaries.")

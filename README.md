# 🎙️ Audio‑to‑Text RAG for Podcast Search

A lightweight multimodal RAG system that ingests audio podcasts, transcribes them with timestamps, builds a searchable index across multiple episodes, and answers topic queries with time‑coded references.

**Demo stack:** Streamlit + faster‑whisper + Sentence‑Transformers + ChromaDB (+ optional pyannote diarization, + optional Groq/OpenAI LLM).

## Features

- Upload multiple podcast episodes (`.mp3/.wav/.m4a`).
- High‑quality speech‑to‑text via `faster-whisper` (CPU‑friendly).
- Word‑level timestamps → smart time‑window chunking (with overlap).
- Multi‑episode semantic search using `all-MiniLM-L6-v2` embeddings in ChromaDB.
- Hybrid retrieval: BM25 + dense with Reciprocal Rank Fusion.
- Returns top hits with timestamps and episode IDs; jump-to-time hints.
- (Optional) Speaker diarization with `pyannote.audio` (requires HF token).
- (Optional) Generative answers using Groq (Llama‑3.x) or OpenAI—fallback to extractive answer if no API key.
- Basic metrics in UI: ingestion time, indexing time, query latency.

## Project structure

```
podcast-rag/
├── app.py
├── rag_pipeline/
│   ├── __init__.py
│   ├── transcribe.py
│   ├── diarize.py
│   ├── chunking.py
│   ├── index.py
│   ├── search.py
│   ├── generate.py
│   └── utils.py
├── requirements.txt
├── packages.txt           # for Hugging Face Spaces (installs ffmpeg)
├── README.md
└── .gitignore
```

## Quickstart (local)

```bash
git clone <your-fork-url> podcast-rag
cd podcast-rag
python -m venv .venv && source .venv/bin/activate   # on Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Then open the Streamlit URL, upload a few `.mp3` files, click **Ingest & Index**, and start asking questions like:
- "When did they discuss reinforcement learning?"
- "What did the guest say about funding in 2024?"

## Environment variables (optional)

Create a `.env` (or set in your hosting environment):

- `GROQ_API_KEY` – to use Groq's LLM (e.g., llama-3.1-8b-instant).
- `OPENAI_API_KEY` – to use OpenAI models instead.
- `HUGGINGFACE_TOKEN` – to enable pyannote.audio speaker diarization.

Without these keys, the app still works in extractive mode.

## Deploy to Hugging Face Spaces

1. Create a new Space → Streamlit template.
2. Upload this repo (or connect to your GitHub).
3. Ensure `packages.txt` and `requirements.txt` are included (Spaces will install ffmpeg and Python deps).
4. Set optional secrets for APIs in the Space settings.
5. The app launches automatically and persists index in `./chroma` (ephemeral on free tier).

## Evaluation (lightweight)

- Retrieval@k: The app shows top‑k chunk hits per query and their scores.
- Latency: Ingestion and query runtimes are printed in the UI.
- RAGAS (optional): You can export Q/A/context triplets from the UI and run RAGAS offline.

## License

MIT

# WhatsApp Reel Knowledge Base

Simple prototype to save WhatsApp / Instagram reels as a searchable knowledge base using:

- FastAPI backend
- OpenAI (chat + embeddings)
- Pinecone vector database
- Streamlit UI

## Setup

1. Create a virtual environment and install dependencies:

```bash
cd "Whatsapp Reel Knowledge Base"
python -m venv .venv
source .venv/bin/activate  # on Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

2. Set your OpenAI API key in a `.env` file:

```bash
echo "OPENAI_API_KEY=sk-..." > .env
```

> Make sure the `.env` file is NOT committed to source control.

## Run the backend (FastAPI)

```bash
uvicorn backend.main:app --reload --port 8000
```

The API docs will be available at `http://localhost:8000/docs`.

## Run the UI (Streamlit)

In another terminal (with the same virtualenv activated):

```bash
streamlit run ui/app.py
```

Open the printed URL (usually `http://localhost:8501`) in your browser.

## Usage

- **Save new reel**:
  - Paste a reel URL.
  - Optionally add manual tags like `goa, restaurant, street food`.
  - The backend:
    - (Stub) generates a fake transcript.
    - Summarizes it using the LLM.
    - Generates auto-tags.
    - Stores everything in Chroma with an embedding.

- **Ask questions**:
  - Example queries:
    - `show restaurants we saved in Goa`
    - `any reels about Bali?`
    - `cheap street food ideas?`
  - The system embeds your query, searches similar reels, and shows the best matches.

## Next steps / ideas

- Replace the `fake_transcript_from_reel` stub with real:
  - Reel downloader → audio file.
  - Whisper (local or via OpenAI) to get the true transcript.
- Add user accounts and share collections between you and your wife.
- Move metadata to Postgres and keep embeddings in Chroma / Pinecone.


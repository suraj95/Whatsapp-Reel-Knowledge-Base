## Travel Reels Knowledge Base 🎥🧠

A small AI project that turns **Travel reels into a searchable knowledge base**.

## The Problem 🤔

My wife and I constantly share reels for:

- 🌍 travel destinations  
- 🍜 restaurants  
- 🏝 hidden beaches  
- 🍣 food spots  

They get buried across multiple WhatsApp groups, and after a while we forget things like:

> “What was that Goa restaurant reel?”  
> “Didn't we save a Bali beach video?”

Scrolling through chat history becomes impossible.

---

## The Idea 💡

Turn reels into a **personal AI memory**.

Paste a reel URL and the system will:

1. Download the reel  
2. Extract frames from the video  
3. Analyze the frames with an AI vision model  
4. Generate a summary + tags  
5. Store embeddings in a vector database  

Later you can **search it using natural language**.

Example queries:

- show restaurants we saved in Goa  
- any reels about Bali beaches?  
- cheap street food ideas  

---

## Demo 🚀

### Copy a Reel URL

![Reel Copy Placeholder](./docs/images/reel.png)

### Paste the Reel URL and get AI Generated Summary

![Summary Placeholder](./docs/images/reel_summary_placeholder.png)

### Search the Knowledge Base

![Search Placeholder](./docs/images/reel_search_placeholder.png)

---

## Tech Stack ⚙️

**Backend**
- FastAPI

**AI**
- OpenAI (vision summarization + embeddings)

**Vector Database**
- Pinecone

**Frontend**
- Streamlit

**Video Processing**
- yt-dlp (reel download)  
- ffmpeg (frame extraction)

---

## Architecture 🏗

Reel URL  
↓  
Download reel (yt-dlp)  
↓  
Extract frames (ffmpeg)  
↓  
Vision model summary  
↓  
Embeddings  
↓  
Pinecone vector database  
↓  
Natural language search  

---

## Setup 🛠

Create a virtual environment:

```bash
cd "Whatsapp Reel Knowledge Base"

python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Create `.env` file:

```bash
OPENAI_API_KEY=sk-...
PINECONE_API_KEY=...
```

⚠️ Make sure `.env` is not committed to source control.

---

## Run Backend

```bash
uvicorn backend.main:app --reload --port 8000
```

API docs:

`http://localhost:8000/docs`

---

## Run UI

```bash
streamlit run ui/app.py
```

Open in browser:

`http://localhost:8501`

---

## Usage 🧭

### Save a Reel

Paste a reel URL and optionally add tags like:

- goa  
- restaurant  
- street food  

The backend will:

- download the reel  
- extract frames  
- generate an AI summary  
- create embeddings  
- store everything in Pinecone  

### Ask Questions

Example queries:

- show restaurants we saved in Goa  
- any reels about Bali?  
- cheap street food ideas  

The system embeds the query, searches Pinecone, and returns the most relevant reels.


# AI-Chatbot (Upgraded) — RAG + Persistent Memory + Streamlit Deployment

This is an upgraded customer-support chatbot built with **LangChain + OpenAI API + ChromaDB** and deployed via **Streamlit**.
It supports:
- Retrieval-Augmented Generation (RAG) with ChromaDB (upload PDFs / txt into `support_docs/`)
- Persistent chat memory using SQLite (simple user-session based storage)
- Streamlit UI with sidebar settings and live chat
- GitHub + Streamlit Cloud deployment ready (see steps below)

## Live Demo
Deploy on Streamlit Cloud (instructions below) and add your `OPENAI_API_KEY` in Secrets.

## Quick start (local)
```bash
git clone <your-repo-url>
cd AI-Chatbot-Upgraded
python -m venv venv
source venv/bin/activate   # mac/linux
# or `venv\Scripts\activate` on Windows
pip install -r requirements.txt
export OPENAI_API_KEY="sk-..."
streamlit run app.py
```

## Files
- `app.py` — Streamlit application (UI + glue)
- `chatbot.py` — Core chatbot logic: LangChain + Chroma + persistent memory
- `data_loader.py` — Load PDFs / text files from `support_docs/` into Chroma
- `requirements.txt` — Python dependencies
- `Dockerfile` — Optional containerization
- `support_docs/` — drop PDFs or text docs here for the bot to ingest
- `db/` — sqlite file will be created here for conversation memory (committed as empty here)

## How to deploy on Streamlit Cloud
1. Push this repository to GitHub.
2. Go to https://share.streamlit.io and create a new app.
3. Select your repo, branch, and set the main file as `app.py`.
4. Add a secret named `OPENAI_API_KEY` with your OpenAI API key.
5. Deploy and open the shareable link.

## Notes / Limitations
- ChromaDB is used here in local persistent mode (it stores embeddings locally).
- OpenAI API usage costs may apply.
- For production, consider more robust vector DB (Pinecone), authentication, HTTPS, rate-limiting, and data privacy.

## Future enhancements
- Add user authentication (login)
- Host ChromaDB on managed vector DB for scale
- Build a React frontend and FastAPI backend for multi-user production
- Add analytics and feedback loop for model improvements

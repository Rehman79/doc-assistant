# 📄 Doc Assistant

> Chat with your company documents — every answer cites its exact source, and it never makes things up.

Doc Assistant is an AI document assistant built on **Retrieval-Augmented Generation (RAG)**. Upload one or more PDFs (an employee handbook, a policy manual, product docs), ask questions in plain English, and get answers grounded in your files complete with the source file and page number. Ask something that isn't in the documents and it honestly replies *"I couldn't find that in the documents"* instead of guessing.

🔗 **Live demo:** https://doc-assistant-l.streamlit.app

---

## ✨ Features

- **Upload & chat** — drop in PDFs and start asking questions immediately.
- **Cited answers** — every response shows the exact source passages (file + page) it used.
- **No hallucinations** — answers are restricted to your documents; off-topic questions are refused.
- **Multi-document search** — ask across several PDFs at once.
- **Clean, responsive UI** — a polished dark theme that works on desktop and mobile.

---

## 🧠 How it works
1. **Load & chunk** — PDFs are read and split into overlapping ~1,000-character chunks.
2. **Embed** — each chunk is converted to a vector with OpenAI `text-embedding-3-small`.
3. **Retrieve** — the question is embedded and the 4 most relevant chunks are pulled from an in-memory vector store.
4. **Generate** — those chunks are passed to `gpt-4o-mini`, which answers using only that context and cites it.

---

## 🛠️ Tech stack

| Layer | Tool |
|---|---|
| UI | [Streamlit](https://streamlit.io) |
| Orchestration | [LangChain](https://www.langchain.com) |
| Embeddings & LLM | [OpenAI](https://platform.openai.com) (`text-embedding-3-small`, `gpt-4o-mini`) |
| Vector store | LangChain `InMemoryVectorStore` |
| PDF parsing | `pypdf` |

---

## 🚀 Run it locally

**Prerequisites:** Python 3.11–3.12 and an [OpenAI API key](https://platform.openai.com/api-keys).

```bash
# 1. Clone the repo
git clone https://github.com/<your-username>/doc-assistant.git
cd doc-assistant

# 2. Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
streamlit run app.py
```

The app opens at `http://localhost:8501`. Paste your OpenAI key when prompted, upload a PDF, and start asking.

---

## 📁 Project structure

```
doc-assistant/
├── app.py                  # The full Streamlit + RAG app
├── requirements.txt        # Python dependencies
├── .streamlit/
│   └── config.toml         # Dark theme configuration
├── .gitignore
└── README.md
```

---

## 🔒 Security

- Your OpenAI key is never committed — `.gitignore` blocks `secrets.toml` and `.env`.
- Uploaded documents are processed in memory for the session and are not persisted.

---

## 📝 License

Released under the MIT License. Feel free to use and adapt it.

---

_Built as a portfolio project demonstrating a production-style RAG pipeline with source citations._

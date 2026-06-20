"""
Doc Assistant — a RAG (Retrieval-Augmented Generation) chat app.

What it does:
  1. You upload one or more PDFs.
  2. The app splits them into chunks and turns each chunk into a vector (embedding).
  3. You chat with your docs. It finds the most relevant chunks and sends them to
     GPT, which answers using ONLY those chunks — and shows you the source.
  4. If the answer isn't in the documents, it says so instead of making something up.

Run it locally with:  streamlit run app.py
"""

import os
import html as html_lib
import tempfile

import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# ----------------------------------------------------------------------------
# Page setup + styling
# ----------------------------------------------------------------------------
st.set_page_config(page_title="Doc Assistant", page_icon="📄", layout="wide")

st.html("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap" rel="stylesheet">
<style>
  /* ===== Obsidian Flow theme ===== */
  :root {
    --bg:#13131b; --primary:#c0c1ff; --primary2:#8083ff;
    --violet:#ddb7ff; --pink:#ec4899; --tertiary:#ffb783;
    --ink:#e4e1ed; --ink-dim:#c7c4d7; --line:rgba(255,255,255,.10);
    --card:rgba(255,255,255,.03);
  }
  html, body, [class*="css"] { font-family:'Plus Jakarta Sans', sans-serif; }
  #MainMenu, footer, header [data-testid="stToolbar"] { visibility:hidden; }

  /* Atmospheric background */
  .stApp {
    background:
      radial-gradient(1000px 520px at 10% -10%, rgba(128,131,255,.16), transparent 60%),
      radial-gradient(900px 520px at 92% -6%, rgba(236,72,153,.12), transparent 55%),
      radial-gradient(700px 600px at 50% 120%, rgba(192,193,255,.10), transparent 60%),
      var(--bg);
  }
  .block-container { padding-top:2rem; max-width:1120px; }

  /* ===== Hero ===== */
  .hero {
    position:relative; overflow:hidden;
    background: linear-gradient(115deg,#4b3fb0 0%,#7c3aed 38%,#b5468f 70%,#d97721 105%);
    background-size:200% 200%; animation:heroShift 12s ease infinite;
    padding:40px 44px; border-radius:22px; margin-bottom:22px; color:#fff;
    border:1px solid rgba(255,255,255,.12);
    box-shadow:0 24px 60px rgba(76,46,160,.40), inset 0 1px 0 rgba(255,255,255,.18);
  }
  @keyframes heroShift {0%{background-position:0% 50%}50%{background-position:100% 50%}100%{background-position:0% 50%}}
  .hero::after { content:""; position:absolute; inset:0;
    background:radial-gradient(460px 220px at 80% 10%, rgba(255,255,255,.22), transparent 60%); }
  .hero .badge {
    display:inline-block; background:rgba(255,255,255,.16); backdrop-filter:blur(8px);
    border:1px solid rgba(255,255,255,.30); color:#fff; font-size:.68rem; font-weight:700;
    letter-spacing:.14em; text-transform:uppercase; padding:5px 13px; border-radius:999px; margin-bottom:14px;
  }
  .hero h1 { margin:0; font-size:2.7rem; font-weight:800; letter-spacing:-1.2px; }
  .hero p  { margin:10px 0 0; opacity:.95; font-size:1.05rem; max-width:560px; line-height:1.6; }
  .hero .pills { margin-top:18px; display:flex; gap:10px; flex-wrap:wrap; }
  .hero .pill {
    background:rgba(255,255,255,.14); backdrop-filter:blur(8px);
    border:1px solid rgba(255,255,255,.26); color:#fff;
    font-size:.8rem; font-weight:600; padding:7px 14px; border-radius:999px;
  }

  /* ===== Custom metric cards (icon tile + value) ===== */
  .metric-row { display:flex; gap:16px; margin:4px 0 18px; flex-wrap:wrap; }
  .metric {
    flex:1; min-width:180px; display:flex; align-items:center; gap:14px;
    background:var(--card); border:1px solid var(--line); border-radius:16px;
    padding:16px 18px; box-shadow:0 10px 32px rgba(0,0,0,.30);
    transition:transform .15s ease, border-color .15s ease;
  }
  .metric:hover { transform:translateY(-3px); border-color:rgba(128,131,255,.5); }
  .metric .tile {
    width:46px; height:46px; border-radius:13px; display:grid; place-items:center;
    font-size:1.25rem; background:linear-gradient(140deg, rgba(128,131,255,.30), rgba(236,72,153,.22));
    border:1px solid rgba(255,255,255,.14);
  }
  .metric .lbl { font-size:.7rem; font-weight:700; letter-spacing:.12em; text-transform:uppercase; color:var(--ink-dim); }
  .metric .val { font-size:1.5rem; font-weight:800; color:var(--ink); line-height:1.1; }

  /* ===== Chat bubbles ===== */
  [data-testid="stChatMessage"] {
    background:var(--card); border:1px solid var(--line);
    border-radius:18px; padding:8px 16px; margin-bottom:8px;
    box-shadow:0 8px 24px rgba(0,0,0,.22);
  }

  /* ===== Source citation cards ===== */
  .src-card {
    background:linear-gradient(180deg, rgba(236,72,153,.12), rgba(236,72,153,.04));
    border:1px solid rgba(236,72,153,.30); border-left:4px solid var(--pink);
    border-radius:14px; padding:12px 16px; margin:8px 0; font-size:.92rem; color:var(--ink-dim);
    transition:transform .15s ease, box-shadow .15s ease;
  }
  .src-card:hover { transform:translateX(3px); box-shadow:0 10px 26px rgba(236,72,153,.22); }
  .src-tag {
    display:inline-block; background:linear-gradient(90deg,#8083ff,#ddb7ff,#ec4899);
    color:#1b1b23; font-size:.72rem; font-weight:800; padding:3px 12px;
    border-radius:999px; margin-bottom:8px; letter-spacing:.2px;
  }

  /* ===== Sidebar branding ===== */
  section[data-testid="stSidebar"] {
    background:linear-gradient(180deg,#191925,#13131b);
    border-right:1px solid var(--line);
  }
  .brand { display:flex; align-items:center; gap:12px; padding:6px 4px 2px; }
  .brand .logo {
    width:40px; height:40px; border-radius:12px; display:grid; place-items:center; font-size:1.3rem;
    background:linear-gradient(140deg,#8083ff,#ec4899); box-shadow:0 6px 18px rgba(128,131,255,.45);
  }
  .brand .name { font-weight:800; font-size:1.08rem; color:var(--ink); line-height:1; }
  .brand .tag  { font-size:.62rem; font-weight:700; letter-spacing:.16em; text-transform:uppercase; color:var(--ink-dim); margin-top:4px; }
  .navlist { margin:16px 0 6px; }
  .navlist .item {
    display:flex; align-items:center; gap:11px; padding:9px 12px; border-radius:11px;
    color:var(--ink-dim); font-weight:600; font-size:.92rem; margin-bottom:3px;
  }
  .navlist .item.active { background:rgba(128,131,255,.16); color:#fff; border:1px solid rgba(128,131,255,.30); }

  /* ===== Buttons ===== */
  div[data-testid="stButton"] button {
    width:100%; text-align:left; border-radius:12px; border:1px solid var(--line);
    background:var(--card); color:var(--ink); font-weight:600; transition:all .15s ease;
  }
  div[data-testid="stButton"] button:hover {
    border-color:var(--primary2);
    background:linear-gradient(90deg, rgba(128,131,255,.18), rgba(236,72,153,.16));
    transform:translateY(-2px); color:#fff;
  }

  /* ===== Chat input ===== */
  [data-testid="stChatInput"] textarea { font-size:1rem; }
  [data-testid="stChatInput"] { border-radius:16px; }

  /* ===== Chat bubbles (custom) ===== */
  .row { display:flex; gap:11px; margin:14px 0; align-items:flex-end; animation:rise .35s ease both; }
  .row.user { justify-content:flex-end; }
  @keyframes rise { from{opacity:0; transform:translateY(8px)} to{opacity:1; transform:none} }
  .bubble { max-width:76%; padding:13px 17px; border-radius:18px; line-height:1.62; font-size:.97rem; }
  .ub { background:linear-gradient(135deg,#6f5cff,#9a6cff); color:#fff;
        border-bottom-right-radius:6px; box-shadow:0 10px 26px rgba(128,131,255,.40); }
  .ab { background:rgba(255,255,255,.045); border:1px solid var(--line); color:var(--ink);
        border-bottom-left-radius:6px; box-shadow:0 8px 24px rgba(0,0,0,.22); }
  .av { width:38px; height:38px; border-radius:12px; display:grid; place-items:center;
        font-size:1.1rem; flex:0 0 auto; }
  .uav { background:linear-gradient(140deg,#8083ff,#ec4899); }
  .aav { background:rgba(255,255,255,.06); border:1px solid var(--line); }
  .src-wrap { margin:2px 0 6px 49px; }
  .src-head { font-size:.74rem; font-weight:700; letter-spacing:.1em; text-transform:uppercase;
              color:var(--ink-dim); margin:6px 0; }

  /* ===== Welcome / empty state ===== */
  .welcome { text-align:center; margin:26px auto 8px; max-width:560px; }
  .welcome .big { font-size:2.6rem; }
  .welcome h2 { margin:6px 0 4px; font-weight:800; letter-spacing:-.5px; }
  .welcome p { color:var(--ink-dim); font-size:1rem; }
  .feat-grid { display:flex; gap:16px; flex-wrap:wrap; margin:22px 0 6px; }
  .feat { flex:1; min-width:200px; text-align:left; background:var(--card);
          border:1px solid var(--line); border-radius:18px; padding:22px;
          box-shadow:0 10px 30px rgba(0,0,0,.28); transition:transform .15s ease, border-color .15s ease; }
  .feat:hover { transform:translateY(-4px); border-color:rgba(128,131,255,.45); }
  .feat .ic { width:48px; height:48px; border-radius:14px; display:grid; place-items:center;
              font-size:1.4rem; margin-bottom:12px;
              background:linear-gradient(140deg, rgba(128,131,255,.30), rgba(236,72,153,.20));
              border:1px solid rgba(255,255,255,.14); }
  .feat h4 { margin:0 0 5px; font-size:1.02rem; font-weight:700; }
  .feat p { margin:0; font-size:.88rem; color:var(--ink-dim); line-height:1.55; }
  .step { display:inline-flex; align-items:center; gap:8px; background:rgba(128,131,255,.12);
          border:1px solid rgba(128,131,255,.3); color:var(--primary); font-weight:700;
          font-size:.78rem; padding:4px 12px; border-radius:999px; margin-bottom:4px; }

  /* Section heading */
  .sec { font-size:.78rem; font-weight:700; letter-spacing:.12em; text-transform:uppercase;
         color:var(--ink-dim); margin:6px 0 2px; }

  /* Scrollbar */
  ::-webkit-scrollbar { width:10px; height:10px; }
  ::-webkit-scrollbar-thumb { background:rgba(128,131,255,.35); border-radius:10px; }
</style>
""")

st.html("""
<div class="hero">
  <span class="badge">⚡ Powered by GPT-4o</span>
  <h1>📄 Doc Assistant</h1>
  <p>Chat with your documents — every answer cites its source and never makes things up.</p>
  <div class="pills">
    <span class="pill">⚡ Instant answers</span>
    <span class="pill">📎 Source citations</span>
    <span class="pill">🔒 Won't hallucinate</span>
  </div>
</div>
""")

# ----------------------------------------------------------------------------
# API key — from Streamlit secrets (deployment) or a sidebar box (local)
# ----------------------------------------------------------------------------
try:
    api_key = st.secrets["OPENAI_API_KEY"]
except Exception:
    api_key = ""

# ----------------------------------------------------------------------------
# The prompt: only use provided context, and admit when the answer isn't there.
# ----------------------------------------------------------------------------
PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a helpful assistant that answers questions about the user's uploaded "
     "documents. Use ONLY the context below to answer. The context is made of "
     "excerpts from the documents; it may have odd line breaks or placeholders in "
     "brackets (such as a bracketed organization name) — treat those placeholders "
     "as the company. Read the context carefully and give a clear, complete answer "
     "drawn from it, summarizing in your own words. "
     "Only if the context contains nothing relevant to the question, reply exactly: "
     "\"I couldn't find that in the documents.\" Do not use outside knowledge.\n\n"
     "Context:\n{context}"),
    ("human", "Question: {question}"),
])

EXAMPLE_QUESTIONS = [
    "What is the policy against workplace harassment?",
    "What leave benefits does the organization offer?",
    "What are the hours of work and attendance rules?",
    "What happens to company property when employment ends?",
]


@st.cache_resource(show_spinner="📚 Reading and indexing your documents...")
def build_index(file_bytes_list, file_names):
    """Turn uploaded PDFs into a searchable vector store. Cached so it only
    re-runs when the uploaded files change."""
    all_chunks = []
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)

    for data, name in zip(file_bytes_list, file_names):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(data)
            tmp_path = tmp.name

        pages = PyPDFLoader(tmp_path).load()
        for p in pages:
            p.metadata["source"] = name
        all_chunks.extend(splitter.split_documents(pages))
        os.unlink(tmp_path)

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = InMemoryVectorStore.from_documents(all_chunks, embeddings)
    return vectorstore, len(all_chunks)


def format_context(docs):
    return "\n\n".join(d.page_content for d in docs)


def bubble_html(role, content):
    """Render one chat message as a styled bubble (user right, AI left)."""
    safe = html_lib.escape(content).replace("\n", "<br>")
    if role == "user":
        return (f'<div class="row user"><div class="bubble ub">{safe}</div>'
                f'<div class="av uav">🧑</div></div>')
    return (f'<div class="row ai"><div class="av aav">🤖</div>'
            f'<div class="bubble ab">{safe}</div></div>')


def sources_html(docs):
    """Build the source-citation cards as one HTML string."""
    cards = ['<div class="src-wrap"><div class="src-head">📎 Sources</div>']
    for d in docs:
        src = html_lib.escape(str(d.metadata.get("source", "unknown")))
        page = d.metadata.get("page", "?")
        snippet = html_lib.escape(d.page_content.strip().replace("\n", " "))
        if len(snippet) > 320:
            snippet = snippet[:320] + "…"
        cards.append(
            f'<div class="src-card"><span class="src-tag">{src} · page {page}</span>'
            f'<br>{snippet}</div>'
        )
    cards.append("</div>")
    return "".join(cards)


# ----------------------------------------------------------------------------
# Sidebar — setup + status
# ----------------------------------------------------------------------------
with st.sidebar:
    st.html("""
    <div class="brand">
      <div class="logo">📄</div>
      <div><div class="name">Doc Assistant</div>
      <div class="tag">Premium AI Analysis</div></div>
    </div>
    <div class="navlist">
      <div class="item active">📊 Dashboard</div>
      <div class="item">📚 Library</div>
      <div class="item">🔎 Analysis</div>
      <div class="item">⚙️ Settings</div>
    </div>
    """)

    st.markdown("**SETUP**")
    if not api_key:
        api_key = st.text_input("OpenAI API key", type="password",
                                help="Starts with sk-… Used only in your session.")
        if api_key:
            st.success("Key loaded ✓")
    else:
        st.success("API key loaded ✓")

    uploaded = st.file_uploader("Upload Document(s)", type="pdf",
                                accept_multiple_files=True)

    st.divider()
    show_debug = st.toggle("🔍 Show sources / debug", value=False)
    if st.button("🗑️ Clear chat"):
        st.session_state.messages = []
        st.rerun()

os.environ["OPENAI_API_KEY"] = api_key or ""

# ----------------------------------------------------------------------------
# Empty states
# ----------------------------------------------------------------------------
if not api_key:
    st.html("""
    <div class="welcome">
      <div class="step">① Step 1</div>
      <div class="big">🔑</div>
      <h2>Add your OpenAI key to begin</h2>
      <p>Paste your API key in the sidebar. It stays in your session and is never stored.</p>
    </div>
    """)
    st.stop()

if not uploaded:
    st.html("""
    <div class="welcome">
      <div class="step">② Step 2</div>
      <h2>Upload a document to start chatting</h2>
      <p>Drop a PDF in the sidebar — a handbook, policy manual, or product doc — and ask anything.</p>
    </div>
    <div class="feat-grid">
      <div class="feat"><div class="ic">📥</div><h4>Upload</h4>
        <p>Add one or more PDFs. They're split into searchable chunks instantly.</p></div>
      <div class="feat"><div class="ic">💬</div><h4>Ask</h4>
        <p>Chat in plain English. It searches your docs for the most relevant passages.</p></div>
      <div class="feat"><div class="ic">📎</div><h4>Verify</h4>
        <p>Every answer cites the exact file and page — and refuses to guess off-topic.</p></div>
    </div>
    """)
    st.stop()

vectorstore, n_chunks = build_index(
    [f.getvalue() for f in uploaded], [f.name for f in uploaded]
)

# Hard guard: if no text was extracted, say so loudly instead of failing silently.
if n_chunks == 0:
    st.error(
        "⚠️ I indexed **0 chunks** — no readable text was extracted from the PDF. "
        "This usually means the PDF is a scanned image (pictures of text, not real "
        "text), so there's nothing to search. Try a different PDF that you can "
        "select/copy text from. If you just changed the file, also clear the cache: "
        "top-right ⋮ menu → Clear cache → Rerun."
    )
    st.stop()

retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Status row — custom icon-tile metric cards
st.html(f"""
<div class="metric-row">
  <div class="metric"><div class="tile">📄</div>
    <div><div class="lbl">Files Processed</div><div class="val">{len(uploaded)}</div></div></div>
  <div class="metric"><div class="tile">🧩</div>
    <div><div class="lbl">Chunks Indexed</div><div class="val">{n_chunks}</div></div></div>
  <div class="metric"><div class="tile">🎯</div>
    <div><div class="lbl">Sources / Answer</div><div class="val">4</div></div></div>
</div>
""")

# ----------------------------------------------------------------------------
# Chat
# ----------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# Example-question buttons (only before the first message)
if not st.session_state.messages:
    st.html('<div class="sec">💡 Try asking</div>')
    cols = st.columns(2)
    for i, q in enumerate(EXAMPLE_QUESTIONS):
        if cols[i % 2].button(q, key=f"ex_{i}"):
            st.session_state.pending_q = q
            st.rerun()

# Render chat history as custom bubbles
for m in st.session_state.messages:
    st.html(bubble_html(m["role"], m["content"]))
    if m.get("sources"):
        st.html(sources_html(m["sources"]))
    if show_debug and m.get("debug"):
        with st.expander(f"🔍 Debug — {len(m['debug'])} chunks retrieved"):
            for j, d in enumerate(m["debug"], 1):
                st.markdown(f"**Chunk {j}** (page {d.metadata.get('page','?')})")
                st.text(d.page_content[:500])

# Get question from chat box OR a clicked example button
typed = st.chat_input("Ask a question about your documents…")
pending = st.session_state.pop("pending_q", None)
question = typed or pending

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    st.html(bubble_html("user", question))

    with st.spinner("🔎 Searching your documents…"):
        docs = retriever.invoke(question)
        chain = PROMPT | llm | StrOutputParser()
        answer = chain.invoke(
            {"context": format_context(docs), "question": question}
        )
    st.html(bubble_html("assistant", answer))

    found = "couldn't find" not in answer.lower()
    if found:
        st.html(sources_html(docs))
    if show_debug:
        with st.expander(f"🔍 Debug — {len(docs)} chunks retrieved"):
            for j, d in enumerate(docs, 1):
                st.markdown(f"**Chunk {j}** (page {d.metadata.get('page','?')})")
                st.text(d.page_content[:500])

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": docs if found else None,
        "debug": docs,
    })

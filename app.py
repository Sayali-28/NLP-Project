"""ResearchGPT - Intelligent Research Paper Analysis and Question Answering.

Run with:  streamlit run app.py
"""
import streamlit as st

from src.analysis import extract_keywords, identify_topics, summarize
from src.chunker import make_chunks
from src.embedder import VectorIndex, load_embedder
from src.pdf_utils import load_paper
from src.qa import answer_question, load_generator

st.set_page_config(page_title="ResearchGPT", page_icon="📄", layout="wide")


@st.cache_resource(show_spinner="Loading embedding model (first run downloads it)...")
def get_embedder():
    return load_embedder()


@st.cache_resource(show_spinner="Loading answer model (first run downloads ~1 GB)...")
def get_generator():
    return load_generator()


st.title("📄 ResearchGPT")
st.caption("Intelligent Research Paper Analysis and Question Answering")

with st.sidebar:
    st.header("Upload paper")
    uploaded = st.file_uploader("Research paper (PDF)", type=["pdf"])
    st.header("Settings")
    chunk_size = st.slider("Chunk size (words)", 80, 300, 150, 10)
    top_k = st.slider("Passages to retrieve", 1, 6, 3)
    process = st.button("Process paper", type="primary", disabled=uploaded is None)

if process and uploaded:
    with st.spinner("Reading and indexing the paper..."):
        pages = load_paper(uploaded.getvalue())
        if not pages:
            st.error("No text could be extracted. The PDF may be a scanned image.")
            st.stop()
        embedder = get_embedder()
        chunks = make_chunks(pages, size=chunk_size, overlap=chunk_size // 5)
        index = VectorIndex(embedder, chunks)
        full_text = " ".join(t for _, t in pages)
        st.session_state.paper = {
            "name": uploaded.name,
            "pages": len(pages),
            "index": index,
            "summary": summarize(embedder, full_text),
            "keywords": extract_keywords(embedder, full_text),
            "topics": identify_topics(index),
        }
    st.success(f"Processed {uploaded.name}: {len(pages)} pages, {len(chunks)} chunks.")

paper = st.session_state.get("paper")
if not paper:
    st.info("Upload a research paper PDF in the sidebar and click **Process paper**.")
    st.stop()

tab_qa, tab_sum, tab_kw, tab_ret = st.tabs(
    ["💬 Ask a question", "📝 Summary", "🔑 Keywords & topics", "🔎 Passage retrieval"])

with tab_qa:
    q = st.text_input("Ask a question about the paper", placeholder="What method do the authors propose?")
    if st.button("Get answer") and q.strip():
        with st.spinner("Searching the paper and generating the answer..."):
            ans, hits = answer_question(get_generator(), paper["index"], q, top_k=top_k)
        st.subheader("Answer")
        st.write(ans)
        with st.expander("Supporting passages"):
            for chunk, score in hits:
                st.markdown(f"**Page {chunk.page}** · similarity {score:.2f}")
                st.write(chunk.text)

with tab_sum:
    st.subheader("Paper summary")
    st.write(paper["summary"])

with tab_kw:
    st.subheader("Keywords")
    st.write(", ".join(paper["keywords"]) or "No keywords found.")
    st.subheader("Main topics")
    for i, terms in enumerate(paper["topics"], 1):
        st.markdown(f"**Topic {i}:** " + ", ".join(terms))

with tab_ret:
    query = st.text_input("Search passages by meaning", key="ret")
    if query.strip():
        for chunk, score in paper["index"].search(query, top_k=top_k):
            st.markdown(f"**Page {chunk.page}** · similarity {score:.2f}")
            st.write(chunk.text)
            st.divider()

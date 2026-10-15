
import streamlit as st
import os
import time

from langchain_groq import ChatGroq
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader

from dotenv import load_dotenv

# ---------------- CONFIGURATION ----------------
st.set_page_config(
    page_title="Research AI | RAG Assistant",
    page_icon="📚",
    layout="wide"
)

load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")
openai_api_key = os.getenv("OPENAI_API_KEY")

if groq_api_key:
    os.environ["GROQ_API_KEY"] = groq_api_key

if openai_api_key:
    os.environ["OPENAI_API_KEY"] = openai_api_key

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #0b1020 0%, #101a30 55%, #111827 100%);
        color: #f1f5f9;
    }

    .block-container {
        max-width: 1150px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    .hero {
        padding: 30px;
        border: 1px solid rgba(148, 163, 184, 0.22);
        border-radius: 22px;
        background: linear-gradient(120deg, #172554, #1e293b, #164e63);
        margin-bottom: 25px;
    }

    .hero h1 {
        color: #ffffff;
        font-size: 36px;
        margin-bottom: 8px;
    }

    .hero p {
        color: #cbd5e1;
        font-size: 16px;
        margin-bottom: 0;
    }

    .section-card {
        background: rgba(30, 41, 59, 0.75);
        border: 1px solid rgba(148, 163, 184, 0.2);
        border-radius: 16px;
        padding: 22px;
        margin-bottom: 18px;
    }

    .section-card h3 {
        color: #67e8f9;
        margin-top: 0;
    }

    div.stButton > button {
        width: 100%;
        border: none;
        border-radius: 10px;
        padding: 12px 18px;
        font-weight: 600;
        color: white;
        background: linear-gradient(90deg, #0891b2, #2563eb);
        transition: 0.2s ease;
    }

    div.stButton > button:hover {
        background: linear-gradient(90deg, #0e7490, #1d4ed8);
        transform: translateY(-1px);
    }

    div[data-testid="stTextInput"] input {
        background: #111827;
        color: white;
        border: 1px solid #475569;
        border-radius: 10px;
    }

    div[data-testid="stExpander"] {
        border: 1px solid #334155;
        border-radius: 12px;
        background: rgba(15, 23, 42, 0.7);
    }

    .metric-card {
        padding: 18px;
        border-radius: 14px;
        background: rgba(30, 41, 59, 0.85);
        border: 1px solid #334155;
        text-align: center;
    }

    .metric-title {
        color: #94a3b8;
        font-size: 13px;
    }

    .metric-value {
        color: #67e8f9;
        font-size: 23px;
        font-weight: 700;
    }

    .footer {
        text-align: center;
        color: #94a3b8;
        font-size: 13px;
        padding-top: 25px;
    }
</style>
""", unsafe_allow_html=True)

# ---------------- LLM ----------------
if not groq_api_key:
    st.error("GROQ_API_KEY missing. Please add it to your .env file.")
    st.stop()

llm = ChatGroq(
    groq_api_key=groq_api_key,
    model_name="openai/gpt-oss-20b"
)

prompt = ChatPromptTemplate.from_template("""
Answer the questions based on the provided context only.
Please provide the most accurate response based on the question.

<context>
{context}
</context>

Question: {input}
""")

# ---------------- VECTOR DATABASE ----------------
def create_vector_embedding():
    if not openai_api_key:
        st.error("OPENAI_API_KEY missing. OpenAI embeddings need this key.")
        return False

    try:
        with st.spinner("Loading PDFs and creating embeddings..."):
            embeddings = OpenAIEmbeddings()

            loader = PyPDFDirectoryLoader("research_papers")
            docs = loader.load()

            if not docs:
                st.warning(
                    "No PDFs found. Add PDF files to the research_papers folder."
                )
                return False

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )

            final_documents = text_splitter.split_documents(docs[:50])

            if not final_documents:
                st.warning("No document text was extracted from the PDFs.")
                return False

            vectors = FAISS.from_documents(
                final_documents,
                embeddings
            )

            st.session_state.vectors = vectors
            st.session_state.document_count = len(docs)
            st.session_state.chunk_count = len(final_documents)

        return True

    except Exception as e:
        st.error(f"Could not create vector database: {e}")
        return False


# Initialize session state before querying
if "vectors" not in st.session_state:
    st.session_state.vectors = None

if "document_count" not in st.session_state:
    st.session_state.document_count = 0

if "chunk_count" not in st.session_state:
    st.session_state.chunk_count = 0

# ---------------- HEADER ----------------
st.markdown("""
<div class="hero">
    <h1>📚 Research AI Assistant</h1>
    <p>Ask questions about your research papers using
    Retrieval-Augmented Generation (RAG).</p>
</div>
""", unsafe_allow_html=True)

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    st.markdown("---")
    st.markdown("**Language Model**")
    st.caption("Groq · openai/gpt-oss-20b")
    st.markdown("**Embeddings**")
    st.caption("OpenAI Embeddings")
    st.markdown("**Vector Store**")
    st.caption("FAISS")
    st.markdown("---")

    if st.session_state.vectors is not None:
        st.success("Vector database ready")
        st.write(f"PDFs loaded: {st.session_state.document_count}")
        st.write(f"Text chunks: {st.session_state.chunk_count}")
    else:
        st.info("Create the vector database to begin.")

# ---------------- MAIN LAYOUT ----------------
left, right = st.columns([1, 2], gap="large")

with left:
    st.markdown("""
    <div class="section-card">
        <h3>📂 Knowledge Base</h3>
        <p>Load PDFs from your research_papers folder and
        prepare them for question answering.</p>
    </div>
    """, unsafe_allow_html=True)

    if st.button("🚀 Create Document Embeddings"):
        success = create_vector_embedding()

        if success:
            st.success("Vector database is ready!")
            st.rerun()

    st.markdown("""
    <div class="section-card">
        <h3>💡 How to use</h3>
        <p>1. Put PDFs in research_papers.</p>
        <p>2. Create document embeddings.</p>
        <p>3. Enter a question.</p>
        <p>4. Read the answer and source passages.</p>
    </div>
    """, unsafe_allow_html=True)

with right:
    st.markdown("""
    <div class="section-card">
        <h3>💬 Ask your research question</h3>
        <p>Get answers grounded in your research documents.</p>
    </div>
    """, unsafe_allow_html=True)

    user_prompt = st.text_input(
        "Your question",
        placeholder="e.g. What is the main objective of this research?"
    )

    ask_button = st.button("🔍 Get Answer")

    if ask_button:
        if not user_prompt.strip():
            st.warning("Please enter a question first.")

        elif st.session_state.vectors is None:
            st.warning(
                "Please click 'Create Document Embeddings' first."
            )

        else:
            try:
                with st.spinner("Searching documents and generating answer..."):
                    document_chain = create_stuff_documents_chain(
                        llm,
                        prompt
                    )

                    retriever = st.session_state.vectors.as_retriever()

                    retrieval_chain = create_retrieval_chain(
                        retriever,
                        document_chain
                    )

                    start = time.perf_counter()
                    response = retrieval_chain.invoke({
                        "input": user_prompt
                    })
                    elapsed = time.perf_counter() - start

                st.markdown("### ✨ Answer")
                st.markdown(
                    f'<div class="section-card">{response["answer"]}</div>',
                    unsafe_allow_html=True
                )

                col1, col2 = st.columns(2)

                with col1:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-title">Response Time</div>
                        <div class="metric-value">{elapsed:.2f}s</div>
                    </div>
                    """, unsafe_allow_html=True)

                with col2:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-title">Retrieved Passages</div>
                        <div class="metric-value">{len(response["context"])}</div>
                    </div>
                    """, unsafe_allow_html=True)

                with st.expander("📖 View document similarity search"):
                    for i, doc in enumerate(response["context"], start=1):
                        st.markdown(f"**Passage {i}**")
                        st.write(doc.page_content)

                        source = doc.metadata.get("source")
                        if source:
                            st.caption(f"Source: {source}")

                        st.divider()

            except Exception as e:
                st.error(f"Something went wrong: {e}")

st.markdown("""
<div class="footer">
    Research AI Assistant · Powered by Streamlit, LangChain, Groq and FAISS
</div>
""", unsafe_allow_html=True)

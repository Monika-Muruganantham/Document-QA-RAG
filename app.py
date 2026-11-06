
import os
import hashlib
import tempfile
import streamlit as st

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="DocuMind AI",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------- CUSTOM CSS ----------------

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background:
        radial-gradient(circle at 10% 0%, #18213f 0%, transparent 35%),
        #0b1020;
    color: #e8ecf8;
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #11182b, #0c1222);
    border-right: 1px solid #27304a;
}

[data-testid="stSidebar"] * {
    color: #e8ecf8 !important;
}

.block-container {
    max-width: 1250px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

.hero {
    padding: 35px;
    border-radius: 24px;
    background: linear-gradient(120deg, #171f3c, #22234b, #30245a);
    border: 1px solid #393d70;
    margin-bottom: 25px;
    box-shadow: 0 15px 45px rgba(0,0,0,0.25);
}

.hero-tag {
    color: #a5b4fc;
    font-size: 13px;
    font-weight: 700;
    letter-spacing: 2px;
    text-transform: uppercase;
}

.hero h1 {
    color: #ffffff;
    font-size: 38px;
    font-weight: 800;
    margin: 10px 0;
}

.hero p {
    color: #c4cbe2;
    font-size: 16px;
    margin-bottom: 0;
}

.section-title {
    color: #f1f5ff;
    font-size: 22px;
    font-weight: 700;
    margin: 15px 0;
}

.metric-card {
    background: #151d32;
    border: 1px solid #293450;
    border-radius: 16px;
    padding: 18px;
    text-align: center;
}

.metric-label {
    color: #9aa8c7;
    font-size: 13px;
}

.metric-value {
    color: #ffffff;
    font-size: 25px;
    font-weight: 800;
    margin-top: 5px;
}

.answer-card {
    background: linear-gradient(145deg, #151d32, #171a35);
    border: 1px solid #353c65;
    border-radius: 18px;
    padding: 24px;
    margin: 10px 0 20px 0;
}

.source-card {
    background: #121a2c;
    border: 1px solid #293650;
    border-radius: 12px;
    padding: 15px;
    margin-bottom: 10px;
}

div[data-testid="stFileUploader"] {
    background: #141c30;
    border: 1px dashed #6675bb;
    border-radius: 16px;
    padding: 12px;
}

.stButton > button {
    background: linear-gradient(90deg, #6366f1, #8b5cf6);
    color: white;
    border: none;
    border-radius: 10px;
    padding: 0.65rem 1.2rem;
    font-weight: 700;
    transition: 0.2s ease;
}

.stButton > button:hover {
    border: 1px solid #a5b4fc;
    color: white;
    transform: translateY(-2px);
}

.stTextInput input, .stTextArea textarea {
    background: #141c30 !important;
    color: white !important;
    border: 1px solid #35415f !important;
    border-radius: 12px !important;
}

[data-testid="stChatMessage"] {
    background: #141c30;
    border: 1px solid #283450;
    border-radius: 15px;
    padding: 15px;
}

hr {
    border-color: #293450;
}

footer {
    visibility: hidden;
}
</style>
""", unsafe_allow_html=True)


# ---------------- MODEL LOADERS ----------------

@st.cache_resource
def load_embedding_model():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


@st.cache_resource
def load_llm():
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0
    )


# ---------------- DATABASE CONNECTION ----------------

CHROMA_DIR = "./chroma_db"

if "vectorstore" not in st.session_state:
    if os.path.exists(CHROMA_DIR):
        try:
            st.session_state.vectorstore = Chroma(
                collection_name="document_chunks",
                embedding_function=load_embedding_model(),
                persist_directory=CHROMA_DIR
            )
        except Exception:
            pass


# ---------------- SIDEBAR ----------------

with st.sidebar:

    st.markdown("""
    <div style="padding:10px 0 20px 0;">
        <div style="font-size:32px;">✦</div>
        <h2 style="margin:0;">DocuMind <span style="color:#a78bfa;">AI</span></h2>
        <p style="color:#9aa8c7;">Intelligent document assistant</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("### 📂 Your documents")

    uploaded_files = st.file_uploader(
        "Upload PDF files",
        type=["pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed"
    )

    if uploaded_files:
        st.caption(f"{len(uploaded_files)} PDF(s) selected")

        for file in uploaded_files:
            st.write(f"📄 {file.name}")

    process_button = st.button(
        "⚡ Process documents",
        use_container_width=True
    )

    st.markdown("---")

    st.markdown("### ⚙️ System")
    st.caption("Embedding model")
    st.code("MiniLM-L6-v2", language=None)

    st.caption("Vector database")
    st.code("ChromaDB", language=None)

    st.caption("Language model")
    st.code("Gemini 2.5 Flash", language=None)

    if "vectorstore" in st.session_state:
        st.success("● Knowledge base connected")
    else:
        st.warning("○ Upload and process documents")


# ---------------- HERO SECTION ----------------

st.markdown("""
<div class="hero">
    <div class="hero-tag">✦ AI-powered knowledge workspace</div>
    <h1>Ask your documents anything.</h1>
    <p>
        Upload your PDFs, explore their knowledge, and get answers
        grounded in your documents.
    </p>
</div>
""", unsafe_allow_html=True)


# ---------------- METRICS ----------------

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-label">Document Intelligence</div>
        <div class="metric-value">PDF · RAG</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-label">Knowledge Retrieval</div>
        <div class="metric-value">Semantic Search</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    status = "Ready" if "vectorstore" in st.session_state else "Awaiting PDFs"

    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Assistant Status</div>
        <div class="metric-value">{status}</div>
    </div>
    """, unsafe_allow_html=True)


# ---------------- PROCESS DOCUMENTS ----------------

if process_button:

    if not uploaded_files:
        st.warning("Please upload at least one PDF first.")

    else:
        try:
            with st.spinner("Preparing your document knowledge base..."):

                embeddings = load_embedding_model()
                all_chunks = []

                for uploaded_file in uploaded_files:

                    with tempfile.NamedTemporaryFile(
                        delete=False,
                        suffix=".pdf"
                    ) as temp_file:

                        temp_file.write(uploaded_file.getvalue())
                        pdf_path = temp_file.name

                    try:
                        documents = PyPDFLoader(pdf_path).load()

                        for document in documents:
                            document.metadata["source"] = uploaded_file.name

                        splitter = RecursiveCharacterTextSplitter(
                            chunk_size=1000,
                            chunk_overlap=200
                        )

                        chunks = splitter.split_documents(documents)
                        all_chunks.extend(chunks)

                    finally:
                        os.remove(pdf_path)

                vectorstore = Chroma(
                    collection_name="document_chunks",
                    embedding_function=embeddings,
                    persist_directory=CHROMA_DIR
                )

                ids = []

                for chunk in all_chunks:
                    source = chunk.metadata.get("source", "unknown")
                    page = chunk.metadata.get("page", 0)
                    content = chunk.page_content

                    chunk_id = hashlib.sha256(
                        f"{source}:{page}:{content}".encode()
                    ).hexdigest()

                    ids.append(chunk_id)

                existing = vectorstore.get(ids=ids)
                existing_ids = set(existing["ids"])

                new_docs = []
                new_ids = []

                for doc, doc_id in zip(all_chunks, ids):
                    if doc_id not in existing_ids:
                        new_docs.append(doc)
                        new_ids.append(doc_id)

                if new_docs:
                    vectorstore.add_documents(
                        documents=new_docs,
                        ids=new_ids
                    )

                st.session_state.vectorstore = vectorstore

            st.success(
                f"Successfully processed {len(uploaded_files)} PDF(s). "
                f"Created {len(all_chunks)} chunks."
            )

        except Exception as e:
            st.error(f"Processing failed: {e}")


# ---------------- CHAT SECTION ----------------

st.markdown('<div class="section-title">💬 Document conversation</div>',
            unsafe_allow_html=True)

if "vectorstore" not in st.session_state:

    st.info(
        "Upload your PDF documents from the sidebar and click "
        "**Process documents** to begin."
    )

else:

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:

        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    question = st.chat_input(
        "Ask a question about your documents..."
    )

    if question:

        st.session_state.messages.append({
            "role": "user",
            "content": question
        })

        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):

            with st.spinner("Searching your documents..."):

                try:
                    retriever = st.session_state.vectorstore.as_retriever(
                        search_type="similarity_score_threshold",
                        search_kwargs={
                            "k": 4,
                            "score_threshold": 0.35
                        }
                    )

                    relevant_docs = retriever.invoke(question)

                    if not relevant_docs:
                        answer = (
                            "I couldn't find relevant information "
                            "in your uploaded documents."
                        )

                        st.warning(answer)

                    else:
                        context = "\n\n".join(
                            doc.page_content
                            for doc in relevant_docs
                        )

                        prompt = ChatPromptTemplate.from_template("""
You are DocuMind AI, a helpful document assistant.

Answer the user's question using ONLY the provided context.

If the context does not contain the answer, say:
"I couldn't find this information in your documents."

Be clear, structured, and concise.

Context:
{context}

Question:
{question}

Answer:
""")

                        messages = prompt.invoke({
                            "context": context,
                            "question": question
                        })

                        response = load_llm().invoke(messages)
                        answer = response.content

                        st.markdown(
                            f'<div class="answer-card">{answer}</div>',
                            unsafe_allow_html=True
                        )

                        st.markdown("#### 📚 Sources")

                        for i, doc in enumerate(relevant_docs, start=1):

                            source = doc.metadata.get(
                                "source", "Unknown document"
                            )

                            page = doc.metadata.get("page", 0) + 1

                            with st.expander(
                                f"📄 {source}  ·  Page {page}"
                            ):
                                st.write(doc.page_content)

                except Exception as e:
                    answer = f"Something went wrong: {e}"
                    st.error(answer)

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer
        })


# ---------------- FOOTER ----------------

st.markdown("""
<div style="text-align:center; color:#66728f; padding:35px 0 10px 0;">
    Built with Python · LangChain · ChromaDB · Gemini
    <br><br>
    <span style="font-size:12px;">DocuMind AI — Your knowledge, intelligently connected.</span>
</div>
""", unsafe_allow_html=True)

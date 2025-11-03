import os
# Disable tokenizers parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import streamlit as st
import pandas as pd
from RAGHelper import RAGHelper
from io import BytesIO

st.set_page_config(page_title="End-to-end RAG Architecture", layout="wide", page_icon="📚")

# ----------------------------------------------------------
# STYLES
# ----------------------------------------------------------
st.markdown("""
<style>
    /* Allow full-width horizontal scrolling */
    .scrollable-table {
        overflow-x: auto;
    }
    /* Custom header styling */
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        color: white;
    }
    .feature-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #667eea;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------------
# HEADER
# ----------------------------------------------------------
st.markdown("""
<div class="main-header">
    <h1 style="margin:0; font-size: 2.5rem;">📚 End-to-end RAG Architecture</h1>
    <p style="margin-top: 0.5rem; font-size: 1.1rem; opacity: 0.9;">
        Retrieval-Augmented Generation powered by ChromaDB & OpenAI
    </p>
</div>
""", unsafe_allow_html=True)

# ----------------------------------------------------------
# WHAT IS THIS APP
# ----------------------------------------------------------
with st.expander("ℹ️ What is this app?", expanded=False):
    st.markdown("""
    ### 🤔 What is RAG?
    **Retrieval-Augmented Generation (RAG)** is an AI technique that combines:
    - **Document Retrieval**: Finding relevant information from your documents
    - **LLM Generation**: Using AI to answer questions based on that information

    This approach ensures answers are **grounded in your actual documents** rather than relying on the AI's pre-trained knowledge.

    ### 🎯 What Can You Do?
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class="feature-box">
            <h4>📤 Upload & Process PDFs</h4>
            <ul>
                <li>Upload any PDF document</li>
                <li>Automatically chunks text into manageable pieces</li>
                <li>Creates semantic embeddings using SentenceTransformers</li>
                <li>Stores in ChromaDB vector database</li>
                <li>Prevents duplicate uploads (content-based detection)</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="feature-box">
            <h4>🔍 Semantic Search</h4>
            <ul>
                <li>Finds most relevant chunks for your question</li>
                <li>Uses cosine similarity on embeddings</li>
                <li>Returns top-K most relevant passages</li>
                <li>Shows similarity scores for transparency</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="feature-box">
            <h4>💬 Ask Questions</h4>
            <ul>
                <li>Ask natural language questions about your PDFs</li>
                <li>Get answers powered by OpenAI GPT models</li>
                <li>See the exact context sent to the AI</li>
                <li>View which document chunks were used</li>
                <li>Transparent and explainable AI responses</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="feature-box">
            <h4>🛠️ Manage Your Data</h4>
            <ul>
                <li>View collection statistics and summaries</li>
                <li>Choose different OpenAI models (GPT-4, GPT-4o, etc.)</li>
                <li>Adjust top-K retrieval parameter</li>
                <li>Reset ChromaDB to start fresh</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    ### 🔧 How It Works
    1. **Upload**: You upload a PDF document
    2. **Process**: Text is extracted and split into semantic chunks (10 sentences each)
    3. **Embed**: Each chunk is converted to a 768-dimensional vector using `all-mpnet-base-v2`
    4. **Store**: Vectors are stored in ChromaDB for fast similarity search
    5. **Query**: You ask a question in natural language
    6. **Retrieve**: System finds the most relevant chunks using vector similarity
    7. **Generate**: OpenAI GPT model generates an answer based on retrieved context
    8. **Display**: You see the answer, context, and source chunks

    ### 🚀 Tech Stack
    - **Frontend**: Streamlit
    - **Vector DB**: ChromaDB (persistent storage)
    - **Embeddings**: SentenceTransformers (`all-mpnet-base-v2`)
    - **LLM**: OpenAI GPT (gpt-4o-mini, gpt-4o, gpt-4-turbo)
    - **PDF Processing**: PyMuPDF
    - **Text Processing**: NLTK
    """)

st.markdown("---")

# ----------------------------------------------------------
# SIDEBAR CONFIG
# ----------------------------------------------------------
st.sidebar.header("⚙️ Configuration")

data_dir = "data"
os.makedirs(data_dir, exist_ok=True)

# Model selection
model_id = st.sidebar.selectbox("Model", ["gpt-4o-mini", "gpt-4o", "gpt-4-turbo"])
top_k = st.sidebar.slider("Top K Chunks", 1, 10, 5)



# ----------------------------------------------------------
# INIT RAG HELPER
# ----------------------------------------------------------
rag = RAGHelper(data_dir=data_dir, collection_name="rag_collection", model_id=model_id)

# ----------------------------------------------------------
# PDF UPLOAD AND PROCESSING
# ----------------------------------------------------------
st.subheader("📥 Step 1: Upload or Process PDF")

# Create two columns for upload and reset
col1, col2 = st.columns([3, 1])

with col1:
    uploaded_file = st.file_uploader("Upload your PDF file", type=["pdf"])

with col2:
    st.write("")  # Add spacing
    st.write("")  # Add spacing
    if st.button("🗑️ Reset ChromaDB", type="secondary", use_container_width=True):
        if "confirm_reset" not in st.session_state:
            st.session_state.confirm_reset = True
        else:
            st.session_state.confirm_reset = not st.session_state.confirm_reset

# Show confirmation dialog
if st.session_state.get("confirm_reset", False):
    st.warning("⚠️ Are you sure you want to reset ChromaDB? This will delete ALL stored data!")
    confirm_col1, confirm_col2 = st.columns(2)
    with confirm_col1:
        if st.button("✅ Yes, Reset", type="primary"):
            with st.spinner("Resetting ChromaDB..."):
                if rag.reset_chromadb():
                    st.success("✅ ChromaDB has been reset successfully!")
                    st.session_state.confirm_reset = False
                    st.rerun()
                else:
                    st.error("❌ Failed to reset ChromaDB. Check logs.")
    with confirm_col2:
        if st.button("❌ Cancel"):
            st.session_state.confirm_reset = False
            st.rerun()

process_button = st.button("Process PDF")

if process_button and uploaded_file:
    pdf_path = os.path.join(data_dir, uploaded_file.name)

    # Create a progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()

    # Step 1: Save file
    status_text.text("📁 Saving file...")
    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    progress_bar.progress(20)

    # Step 2: Calculate hash
    status_text.text("🔐 Calculating file hash...")
    file_hash = rag.calculate_file_hash(pdf_path)
    progress_bar.progress(40)

    # Step 3: Read PDF (always read to check for new content)
    status_text.text("📄 Reading PDF pages...")
    pages = rag.readPDF(uploaded_file.name)
    progress_bar.progress(50)

    # Step 4: Process chunks
    status_text.text("✂️ Processing text into chunks...")
    chunks = rag.processPages_to_sentences(pages)
    progress_bar.progress(70)

    # Step 5: Check for duplicates and store in ChromaDB
    status_text.text("💾 Checking for duplicates and storing new chunks...")

    # Capture the print output to show statistics
    import io
    import sys
    old_stdout = sys.stdout
    sys.stdout = buffer = io.StringIO()

    rag.storeInChromaDB(chunks, file_hash=file_hash, deduplicate_chunks=True)

    output = buffer.getvalue()
    sys.stdout = old_stdout
    progress_bar.progress(100)

    status_text.empty()

    # Parse the output to show appropriate message
    if "No new chunks to store" in output:
        st.info(f"📋 File '{uploaded_file.name}' processed: All {len(chunks)} chunks already exist in ChromaDB")
        st.success("✅ No duplicates added - your database is up to date!")
    else:
        st.success(f"✅ PDF '{uploaded_file.name}' processed and stored in ChromaDB!")
        st.info(f"📊 Processed {len(pages)} pages into {len(chunks)} chunks")
        # Show deduplication stats if any
        if "Skipped" in output:
            for line in output.split('\n'):
                if "Skipped" in line or "Stored" in line:
                    st.info(f"ℹ️ {line}")

# Show current ChromaDB status
if rag.collection.count() > 0:
    st.info(f"✅ ChromaDB has **{rag.collection.count()} chunks** ready for queries.")


# ----------------------------------------------------------
# SHOW SUMMARY
# ----------------------------------------------------------
st.subheader("📊 Summary of ChromaDB Collection")

summary = rag.get_summary_info()
if summary:
    df_summary = pd.DataFrame([summary])
    st.table(df_summary)
else:
    st.warning("No summary available yet. Please process a PDF first.")

# Display the model being used
st.info(f"🤖 **Question Answering Model:** {model_id}")

# ----------------------------------------------------------
# ASK A QUESTION
# ----------------------------------------------------------
st.subheader("💬 Step 2: Ask a Question")
query = st.text_input("Enter your question:", value="What was the name of the special broomstick Harry received before this first quidditch match?")
# What was the name of the special broomstick Harry received before this first quidditch match?
# Who gives Harry his first broomstick?

if query:
    with st.spinner("Retrieving context and querying model..."):
        # Get retrieved chunks
        results = rag.search(query=query, top_k=top_k)
    
        if not results:
            st.error("No relevant chunks found. Try rephrasing your question.")
        else:
            # Safely build dataframe
            df = pd.DataFrame(results)

            # Expanders for detailed text
            st.markdown("#### 📜 Chunk Texts")
            for idx, row in df.iterrows():
                with st.expander(f"Page {row['page_number']} (Score: {row['score']:.3f})"):
                    st.write(row["text"])

            # Build messages (context prompt)
            messages = rag.build_context_prompt(query, top_k=top_k)
            with st.expander("🧠 What the LLM Sees (System + User Prompt)", expanded=False):
                for msg in messages:
                    st.markdown(f"**{msg['role'].upper()}:**")
                    st.code(msg["content"])

            # Get model answer
            answer = rag.ask(query, top_k=top_k)
            st.success("### 🧩 Model Answer")
            st.write(answer)

# ----------------------------------------------------------
# FOOTER
# ----------------------------------------------------------
st.markdown("---")
st.caption("Built with ❤️ using Streamlit, OpenAI, and SentenceTransformers.")

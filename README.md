# 📚 End-to-end RAG Architecture

> **AI-powered PDF chat assistant** using Retrieval-Augmented Generation (RAG), ChromaDB, and your choice of LLMs (Ollama open-source models or OpenAI GPT).
> Upload PDFs, ask natural-language questions, and get **accurate, source-grounded answers** with full context visibility.
> **100% free and private** with local open-source Ollama models, or use OpenAI for cloud-based inference.

### Why this matters

- Documents like white-papers, manuals, legal briefs and research reports still hide knowledge. This tool turns them from static blobs into interactive knowledge bases.
- Many systems claim to “chat with your documents” but hide the retrieval step or don’t show provenance. This one shows you the chunks, how they were selected and how the answer was formed.

---

## 📑 Table of Contents

- [Overview](#-overview)
- [Core Features](#-core-features)
- [Architecture](#-architecture)
- [Quick Start](#-quick-start)
- [Next Steps](#-next-steps)
  - [Agentic RAG](#agentic-rag)
  - [Evaluation Framework](#evaluation-framework)
  - [Scaling Strategies](#scaling-strategies)
- [Screenshots](#-screenshots--user-interface-overview)
- [Contributing](#-contributing)
- [License](#-license)
- [Author](#-author)

---

## 🌟 Overview

RAG PDF Chat Assistant combines **semantic search** and **large language models** to help you intelligently query and understand PDF documents.
It extracts, chunks, embeds, and stores document text for fast, context-aware retrieval and question answering.

### Key Highlights

- **PDF Processing** – Text extraction and chunking via PyMuPDF + NLTK
- **Semantic Search** – 768-dim embeddings with SentenceTransformers
- **ChromaDB Storage** – Persistent, deduplicated vector database
- **Dual LLM Support** – Choose between Ollama open-source models (free, local) or OpenAI GPT (cloud-based)
- **AI-Powered Q&A** – Contextual responses from your chosen LLM
- **Transparency** – Shows retrieved chunks and similarity scores
- **Fast** – Retrieves context in milliseconds

---

## ✨ Core Features

<table>
  <tr>
    <td width="50%" valign="top">
      <h3>📤 PDF Upload & Processing</h3>
      <ul>
        <li>Streamlit-based drag-and-drop interface</li>
        <li>Automatic extraction, chunking, and embedding</li>
        <li>Real-time progress and summary statistics</li>
      </ul>
    </td>
    <td width="50%" valign="top">
      <h3>🔐 Smart Deduplication</h3>
      <ul>
        <li>File- and chunk-level SHA-256 hashing</li>
        <li>Avoids duplicate storage across revisions</li>
        <li>Saves 40–60% storage on repeated uploads</li>
      </ul>
    </td>
  </tr>
  <tr>
    <td width="50%" valign="top">
      <h3>🔍 Semantic Search</h3>
      <ul>
        <li>Uses <code>all-mpnet-base-v2</code> embeddings</li>
        <li>Cosine-similarity retrieval (Top-K configurable)</li>
      </ul>
    </td>
    <td width="50%" valign="top">
      <h3>🤖 AI-Powered Q&A</h3>
      <ul>
        <li><strong>Ollama (Default):</strong> <code>llama3.2</code>, <code>llama3.1</code>, <code>mistral</code>, <code>qwen2.5</code> - Free & Local</li>
        <li><strong>OpenAI (Optional):</strong> <code>gpt-4o-mini</code>, <code>gpt-4o</code>, <code>gpt-4-turbo</code> - Cloud-based</li>
        <li>Automatic context-building and prompt expansion</li>
      </ul>
    </td>
  </tr>
</table>

---

## 🏗️ Architecture

![RAG PDF Chat Architecture](images/Architecture.png)

- **Document Ingestion**: PDF → each page → sentences/chunks → SHA-256 hash for deduplication
- **Embedding & Storage**: Sentence-transformer embeddings → store in ChromaDB vector store
- **Query Handling**: Accept user query → retrieve top-k relevant chunks → pass them + query into LLM → return answer + provenance
- **LLM Choice**: open-source models (e.g., llama3.2, mistral, qwen2.5) or OpenAI models (gpt-4o, gpt-4-turbo)
- **UI**: Streamlit interface for upload, chat & inspection of retrieval results.

---

## 🚀 Quick Start

### 1. Install

```bash
git clone https://github.com/mmariappan/end-to-end-rag-architecture
cd end-to-end-rag-architecture
uv sync
```

### 2. Configure LLM Provider

**Option A: Ollama (Recommended - Free & Local)**

```bash
# 1. Download ollama
# Visit https://ollama.com/download and install
brew install ollama

# 2. Pull model and start chatting
ollama pull llama3.2:3b
curl http://localhost:11434/api/chat -d '{"model":"llama3.2:3b","messages":[{"role":"user","content":"Hello!"}],"stream":false}'
```

**Recommended Models:**

- `llama3.2:3b` - Fast, works on 8GB+ RAM, great for development
- `llama3.1:8b` - Better reasoning, requires 16GB+ RAM
- `mistral:7b` - Strong performance, 16GB+ RAM
- `qwen2.5:7b` - Excellent instruction following, 16GB+ RAM

**Option B: OpenAI (Cloud-based)**

Create a `.env` file:

```bash
OPENAI_API_KEY=your_openai_api_key
TOKENIZERS_PARALLELISM=false
```

**Configuration Options:**

- **LLM Provider:** Ollama (default) or OpenAI (selected in UI)
- **Ollama Models:** `llama3.2:3b`, `llama3.1:8b`, `mistral:7b`, `qwen2.5:7b` - Free, runs locally
- **OpenAI Models:** `gpt-4o-mini`, `gpt-4o`, `gpt-4-turbo` - Paid, cloud-based
- **Top-K:** number of chunks to retrieve (default = 5)
- **Chunk Size:** adjustable in `chatpdf_base.py → max_sentences`

### 3. Load Database with PDFs (Python)

Place your PDF files in the `data/` folder and run:

```python
from RAGHelper import RAGHelper

# Initialize RAG helper
rag = RAGHelper(data_dir="data", collection_name="rag_collection")

# Process a PDF
filename = "data/harrypotter.pdf"  # REPLACE with your own data
file_hash = rag.calculate_file_hash(filename)  # avoid duplicates
pages = rag.readPDF(filename)
chunks = rag.processPages_to_sentences(pages)
rag.storeInChromaDB(chunks, file_hash=file_hash, deduplicate_chunks=True)

print(f"Loaded {len(chunks)} chunks into ChromaDB")
```

### 4. Run the App

```bash
uv run streamlit run app.py
```

Visit [http://localhost:8501](http://localhost:8501)

### 5. Using the App

1. **Select LLM Provider** – In the sidebar, choose:

   - **Ollama (Open Source)** - Recommended, free, runs locally (requires Ollama installed and model pulled)
   - **OpenAI** - Cloud-based alternative (requires API key in `.env`)

2. **Upload a PDF** – The app automatically chunks and indexes your document into ChromaDB

3. **Ask Questions** – Type queries like:

   - "Who is the main character introduced in chapter 1?"
   - "Summarize the key points of chapter 3."

4. **Get Contextual Answers** – The system retrieves relevant chunks and uses your chosen LLM to generate accurate, context-aware answers

---

## 📸 Screenshots & User Interface Overview

Below are the screenshots for a visual walkthrough of the Streamlit app.

<table>
  <tr>
    <td width="50%">
      <b>1. App Home Page</b><br/>
      <img src="images/home_page.png" alt="App Home Page" style="width:100%; height:300px; object-fit:cover;"/>
    </td>
    <td width="50%">
      <b>2. PDF Upload Screen</b><br/>
      <img src="images/upload_pdf.png" alt="Upload PDF" style="width:100%; height:300px; object-fit:cover;"/>
    </td>
  </tr>
  <tr>
    <td width="50%">
      <b>3. Question & Answer Screen</b><br/>
      <img src="images/query_response.png" alt="Question & Answer" style="width:100%; height:300px; object-fit:cover;"/>
    </td>
    <td width="50%">
      <b>4. What the LLM Sees</b><br/>
      <img src="images/what_the_LLM_Sees.png" alt="What the LLM Sees" style="width:100%; height:300px; object-fit:cover;"/>
    </td>
  </tr>
</table>

---

## 🚀 Next Steps

The next focus will be on building an **Agentic RAG** system that adds reasoning, reflection, and self-correction to improve answer quality. I’ll also develop an **Evaluation Framework** using metrics like Precision@k, RAGAS, and LLM-as-a-Judge to measure retrieval and response accuracy. Finally, I’ll work on **Scaling and Optimization**, focusing on distributed batch inference for chunking, embedding, and LLM evaluation, as well as parallel online inference using tools like Ray to improve speed and scalability.

---

## 🤝 Contributing

1. Fork → create a branch → commit changes
2. Run tests (`pytest`) and format with `black *.py`
3. Submit a Pull Request

---

## 📄 License

**MIT License © 2025 Mohandas Mariappan**

---

## 👤 Author

Built by **Mohandas Mariappan**

- [LinkedIn](https://www.linkedin.com/in/sunmohandas/)
- [GitHub @mmariappan](https://github.com/mmariappan)

---

**Star this repo if you find it helpful!**

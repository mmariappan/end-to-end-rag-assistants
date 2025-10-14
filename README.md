# 📚 RAG PDF Chat Assistant

> **AI-powered PDF chat assistant** using Retrieval-Augmented Generation (RAG), ChromaDB, and OpenAI GPT models.  
> Upload PDFs, ask natural-language questions, and get **accurate, source-grounded answers** — all with full context visibility.

---

## 🌟 Overview

RAG PDF Chat Assistant combines **semantic search** and **large language models** to help you intelligently query and understand PDF documents.  
It extracts, chunks, embeds, and stores document text for fast, context-aware retrieval and question answering.

### Key Highlights

- 📄 **PDF Processing** – Text extraction and chunking via PyMuPDF + NLTK
- 🔍 **Semantic Search** – 768-dim embeddings with SentenceTransformers
- 💾 **ChromaDB Storage** – Persistent, deduplicated vector database
- 🤖 **AI-Powered Q&A** – Contextual responses from OpenAI GPT models
- 🔎 **Transparency** – Shows retrieved chunks and similarity scores
- ⚡ **Fast** – Retrieves context in milliseconds

**Perfect for:** researchers, students, lawyers, analysts, and anyone managing large document collections.

---

## ✨ Core Features

### 📤 PDF Upload & Processing

- Streamlit-based drag-and-drop interface
- Automatic extraction, chunking, and embedding
- Real-time progress and summary statistics

### 🔐 Smart Deduplication

- File- and chunk-level SHA-256 hashing
- Avoids duplicate storage across revisions
- Saves 40–60 % storage on repeated uploads

### 🔍 Semantic Search

- Uses `all-mpnet-base-v2` embeddings
- Cosine-similarity retrieval (Top-K configurable)

### 🤖 AI-Powered Q&A

- Models: `gpt-4o-mini`, `gpt-4o`, `gpt-4-turbo`
- Temperature = 0.2 for factual answers
- Automatic context-building and prompt expansion

---

## 🏗️ Architecture

![RAG PDF Chat Architecture](RAG_PDF_Chat_Architecture.png)

| Component                | Description                          |
| ------------------------ | ------------------------------------ |
| **Streamlit UI**         | Web interface for upload & chat      |
| **ChatPDF Base**         | PDF parsing, chunking, deduplication |
| **RAGHelper**            | Query handling, GPT API calls        |
| **ChromaDB**             | Vector store for embeddings          |
| **SentenceTransformers** | Generates semantic embeddings        |

---

## ⚙️ Tech Stack

| Layer        | Technology                                 |
| ------------ | ------------------------------------------ |
| UI           | Streamlit                                  |
| Vector DB    | ChromaDB                                   |
| Embeddings   | SentenceTransformers (`all-mpnet-base-v2`) |
| LLM          | OpenAI GPT                                 |
| PDF          | PyMuPDF                                    |
| Tokenization | NLTK                                       |
| Hashing      | hashlib                                    |
| Config       | python-dotenv                              |
| Data         | pandas                                     |

---

## 🚀 Quick Start

### 1️⃣ Install

```bash
git clone https://github.com/mmariappan/rag-pdf-chat-assistant.git
cd rag-pdf-chat-assistant
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

### 2️⃣ Configure Environment

Create a `.env` file:

```bash
OPENAI_API_KEY=your_openai_api_key
TOKENIZERS_PARALLELISM=false
```

### 3️⃣ Run the App

```bash
streamlit run app.py
```

Visit [http://localhost:8501](http://localhost:8501)

---

## 💡 Usage

1. **Upload a PDF** → it’s automatically chunked and stored in ChromaDB
2. **Ask a Question** → GPT model retrieves relevant context and answers
3. **Inspect Sources** → see the exact text chunks used
4. **Re-upload Docs** → deduplication skips previously processed content

---

## 🧠 Configuration

- **Model:** choose `gpt-4o-mini`, `gpt-4o`, or `gpt-4-turbo`
- **Top-K:** number of chunks to retrieve (default = 5)
- **Chunk Size:** adjustable in `chatpdf_base.py → max_sentences`

---

## 🌐 Deployment

### Streamlit Cloud

1. Push repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io) → “New app”
3. Choose `app.py` as entry point
4. Add secret:
   ```toml
   OPENAI_API_KEY = "your_openai_api_key"
   ```

### Docker (optional)

```dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt
EXPOSE 8501
CMD ["streamlit", "run", "app.py"]
```

Run:

```bash
docker build -t rag-pdf-chat .
docker run -p 8501:8501 -e OPENAI_API_KEY=your_key rag-pdf-chat
```

---

## 🧩 Troubleshooting

| Issue                      | Solution                                              |
| -------------------------- | ----------------------------------------------------- |
| `OPENAI_API_KEY not found` | Add `.env` file                                       |
| NLTK error                 | `python -c "import nltk; nltk.download('punkt_tab')"` |
| ChromaDB lock              | Delete `chroma_db/` folder and restart                |
| Slow performance           | Lower `top_k` or use `gpt-4o-mini`                    |

---

## 🤝 Contributing

1. Fork → create a branch → commit changes
2. Run tests (`pytest`) and format with `black *.py`
3. Submit a Pull Request 🚀

---

## 📄 License

**MIT License © 2024 Mohandas Mariappan**

---

## 👤 Author

Built with ❤️ by **Mohandas Mariappan**

- 💼 [LinkedIn](https://www.linkedin.com/in/sunmohandas/)
- 🌐 [GitHub @mmariappan](https://github.com/mmariappan)

---

**⭐ Star this repo if you find it helpful!**

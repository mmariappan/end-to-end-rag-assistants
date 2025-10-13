
# Download PDF File
import os
import requests
import hashlib

# Open and read PDF file
import pymupdf as fitz

import nltk
from nltk.tokenize import sent_tokenize

# For vector database
import chromadb
# ChromaDB client and embedding functions
from chromadb.utils import embedding_functions

# For chapter summarization (optional)
from openai import OpenAI
from dotenv import load_dotenv

# Download NLTK data files (only need to run once)
nltk.download('punkt_tab')

# Main class for handling PDF and embeddings
class ChatPDF:
    def __init__(self, data_dir="data", collection_name="rag_collection"):
        """
        ChromaDB-only implementation for vector storage and retrieval.
        """
        #Save directory passed when instantiating
        self.data_dir = data_dir
        #exist_ok=True → means "don't raise an error if the folder already exists."
        os.makedirs(self.data_dir, exist_ok=True)

        # Initialize ChromaDB
        # Create or connect to a persistent ChromaDB database
        self.chroma_client = chromadb.PersistentClient(path="./chroma_db")
        # Create or get a collection in ChromaDB
        self.collection = self.chroma_client.get_or_create_collection(
            name=collection_name,
            embedding_function=embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name="all-mpnet-base-v2"
                )
        )

    # Calculate file hash (SHA-256)
    def calculate_file_hash(self, file_path):
        """
        Calculate SHA-256 hash of a file to uniquely identify it.
        """
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            # Read file in chunks to handle large files
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()

    # Check if file already processed in ChromaDB
    def is_file_already_processed(self, file_hash):
        """
        Check if a file with the given hash already exists in ChromaDB.
        """
        try:
            # Query ChromaDB for any chunk with this file_hash
            results = self.collection.get(
                where={"file_hash": file_hash},
                limit=1
            )
            return len(results.get("ids", [])) > 0
        except Exception as e:
            print(f"Error checking for duplicate: {e}")
            return False

    # Reset ChromaDB collection
    def reset_chromadb(self):
        """
        Delete all data from the ChromaDB collection.
        """
        try:
            # Delete the collection
            self.chroma_client.delete_collection(name=self.collection.name)
            # Recreate the collection
            self.collection = self.chroma_client.get_or_create_collection(
                name=self.collection.name,
                embedding_function=embedding_functions.SentenceTransformerEmbeddingFunction(
                    model_name="all-mpnet-base-v2"
                )
            )
            print(f"ChromaDB collection '{self.collection.name}' has been reset")
            return True
        except Exception as e:
            print(f"Error resetting ChromaDB: {e}")
            return False

    # Download PDF File
    def downloadPDF(self, filename, url):
        # Build full path inside chosen data directory
        pdf_path = os.path.join(self.data_dir, filename)

        # Download PDF if it doesn't exist
        if not os.path.exists(pdf_path):
            print("File not found. Downloading...")

            # Send a GET request to the URL
            response = requests.get(url)

            # Check if the request was successful
            if response.status_code == 200:
                # Write the content to a file
                with open(pdf_path, 'wb') as file:
                    file.write(response.content)
                print(f"File downloaded and saved to {pdf_path}")
            elif response.status_code == 404:
                print(f"Failed to download file. The URL was not found (404). Please check the URL.")
            else:
                print(f"Failed to download file. Server returned Status code: {response.status_code}")
        else:
            print(f"File found at {pdf_path} exists. Skipping download.")
    
    # Open and read PDF file
    def readPDF(self, filename: str) -> list[dict]:
         # Build full path inside chosen data directory
        pdf_path = os.path.join(self.data_dir, filename)

        # Open the PDF file
        doc = fitz.open(pdf_path)

        pagesfromPDF = []
        for pageNum, pageText in enumerate(doc, start=1):
            text = pageText.get_text()
            pagesfromPDF.append({"page_number": pageNum,
                                 "page_charcount": len(text),
                                 "page_wordcount": len(text.split(" ")),
                                 "page_sentencecount": len(text.split(". ")),
                                 "page_tokencount": len(text) / 4,
                                "text": text})
        return pagesfromPDF
    
    # Function to chunk text into sentences and group them
    def chunk_sentences(self, text, max_sentences=10):
        """
        Split text into sentences and group them into chunks of max_sentences each.
        """
        # Split text into sentences
        sentences = sent_tokenize(text)
        chunks = []
        # Group sentences into chunks
        for i in range(0, len(sentences), max_sentences):
            # Join sentences to form a chunk, in this ex 10 sentences forms a chunk
            # ' '.join(...) → join sentences with a space
            chunk = ' '.join(sentences[i:i + max_sentences])
            # Append chunk to list
            chunks.append(chunk)
        return chunks
    
    # Process PDF pages into sentences and chunks
    def processPages_to_sentences(self, pagesfromPDF, max_sentences=10):
        """
        Process PDF pages:
        - Split into sentences
        - Group sentences into chunks of max_sentences each
        """

        chunked_pages = []
        # Iterate over each page and chunk its text
        for row in pagesfromPDF:
            # chunk_sentences returns a list of strings
            chunks = self.chunk_sentences(row["text"], max_sentences=max_sentences)
            # For each chunk, create a new entry with metadata
            for i, chunk in enumerate(chunks):
                chunked_pages.append({
                    "page_number": row["page_number"],
                    "chunk_id": f"{row['page_number']}_{i}",
                    "text": chunk,
                    "sentence_count": len(sent_tokenize(chunk))
                })
        return chunked_pages
    

    # Calculate chunk hash for content-based deduplication
    def calculate_chunk_hash(self, text):
        """
        Calculate SHA-256 hash of chunk text for deduplication.
        """
        return hashlib.sha256(text.encode('utf-8')).hexdigest()

    # Check if chunk already exists in ChromaDB
    def is_chunk_already_stored(self, chunk_hash):
        """
        Check if a chunk with the given content hash already exists in ChromaDB.
        """
        try:
            results = self.collection.get(
                where={"chunk_hash": chunk_hash},
                limit=1
            )
            return len(results.get("ids", [])) > 0
        except Exception as e:
            # If chunk_hash field doesn't exist in metadata, return False
            return False

    # Store chunks in ChromaDB with deduplication
    def storeInChromaDB(self, chunks, file_hash=None, deduplicate_chunks=True):
        """
        Store chunks in ChromaDB with optional chunk-level deduplication.

        Args:
            chunks: List of chunk dictionaries
            file_hash: Optional file hash for tracking source file
            deduplicate_chunks: If True, skip chunks that already exist (based on content hash)
        """
        new_chunks = []
        duplicate_count = 0

        # Filter out duplicate chunks if deduplication is enabled
        if deduplicate_chunks:
            for chunk in chunks:
                chunk_hash = self.calculate_chunk_hash(chunk["text"])
                if not self.is_chunk_already_stored(chunk_hash):
                    # Add chunk_hash to the chunk data
                    chunk["chunk_hash"] = chunk_hash
                    new_chunks.append(chunk)
                else:
                    duplicate_count += 1
        else:
            # No deduplication, store all chunks
            for chunk in chunks:
                chunk_hash = self.calculate_chunk_hash(chunk["text"])
                chunk["chunk_hash"] = chunk_hash
                new_chunks.append(chunk)

        # Only proceed if there are new chunks to store
        if new_chunks:
            # Prepare data for ChromaDB
            # Use chunk_hash as ID to ensure uniqueness across files
            ids = [c["chunk_hash"][:16] for c in new_chunks]  # Use first 16 chars of hash as ID
            texts = [c["text"] for c in new_chunks]
            # Include file_hash and chunk_hash in metadata
            if file_hash:
                metas = [{"page_number": c["page_number"],
                         "file_hash": file_hash,
                         "chunk_hash": c["chunk_hash"],
                         "original_chunk_id": c["chunk_id"]} for c in new_chunks]
            else:
                metas = [{"page_number": c["page_number"],
                         "chunk_hash": c["chunk_hash"],
                         "original_chunk_id": c["chunk_id"]} for c in new_chunks]

            # Add data to ChromaDB collection
            self.collection.add(ids=ids,
                                documents=texts,
                                metadatas=metas)
            print(f"Stored {len(new_chunks)} new chunks in ChromaDB collection '{self.collection.name}'")
        else:
            print(f"No new chunks to store. All {len(chunks)} chunks already exist.")

        # Report deduplication statistics
        if deduplicate_chunks and duplicate_count > 0:
            print(f"Skipped {duplicate_count} duplicate chunks (already in ChromaDB)")

    # Search in ChromaDB
    def searchInChromaDB(self, query, top_k=5):
        # Query ChromaDB collection
        results = self.collection.query(
            query_texts=[query],
            n_results=top_k
        )
        # Format results
        matches = []
        for doc, meta, dist in zip(results["documents"][0], results["metadatas"][0], results["distances"][0]):
            matches.append({
                "score": float(1 - dist), # Convert distance to similarity score
                "page_number": meta["page_number"],
                "text": doc
            })
        return matches

    # Search for similar text chunks based on query
    def search(self, query, top_k=5):
        """
        Search for similar text chunks using ChromaDB.
        """
        return self.searchInChromaDB(query, top_k=top_k)

    # Chapter summarization using OpenAI
    def summarize_chapter(self, chapter_title, chapter_text, model="gpt-4o-mini"):
        """
        Summarize a chapter in 10 concise sentences using OpenAI.

        Args:
            chapter_title (str): The title of the chapter
            chapter_text (str): The full text of the chapter
            model (str): OpenAI model to use (default: gpt-4o-mini)

        Returns:
            str: Summary of the chapter
        """
        # Load environment variables
        load_dotenv()
        api_key = os.getenv("OPENAI_API_KEY")

        if not api_key:
            raise ValueError("OPENAI_API_KEY not found. Please set it in your .env file.")

        # Initialize OpenAI client
        client = OpenAI(api_key=api_key)

        # Truncate text if too long (keep first 12000 characters)
        truncated_text = chapter_text[:12000]

        # Create prompt
        prompt = f"""
Summarize this chapter in 20 concise sentences, focusing on key events, characters, and emotions.
Chapter title: {chapter_title}

Text:
{truncated_text}
"""

        # Call OpenAI API
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=500
        )

        return response.choices[0].message.content.strip() if response.choices[0].message.content else ""



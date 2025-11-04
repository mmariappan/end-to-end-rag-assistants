
from chatpdf_base import ChatPDF
from openai import OpenAI
from dotenv import load_dotenv
import os

# RAGHelperOllama class extending ChatPDF for Ollama-hosted open-source LLMs
class RAGHelperOllama(ChatPDF):
    # Initialize with data directory, collection name, and model ID
    def __init__(self, data_dir="data", collection_name="rag_collection", model_id="llama3.2:3b", ollama_base_url="http://localhost:11434/v1"):
        """
        RAG Helper using Ollama-hosted open-source models with ChromaDB.
        Uses OpenAI-compatible API provided by Ollama.

        Args:
            data_dir: Directory for PDF storage
            collection_name: ChromaDB collection name
            model_id: Model identifier (e.g., "llama3.2:3b", "mistral:7b")
            ollama_base_url: Ollama server endpoint (default: http://localhost:11434/v1)
        """
        # Load environment variables from .env
        load_dotenv()
        # Initialize the base ChatPDF class (ChromaDB only)
        super().__init__(data_dir=data_dir, collection_name=collection_name)

        # Initialize OpenAI client pointing to Ollama server
        # Ollama provides OpenAI-compatible API
        self.ollama_base_url = ollama_base_url
        self.client = OpenAI(
            base_url=ollama_base_url,
            api_key="ollama"  # Ollama doesn't require API key but OpenAI client needs something
        )

        # Model ID
        self.model_id = model_id
        print(f"Using Ollama model: {model_id} at {ollama_base_url}")

    # ------------------------------------------------------------- #
    # Search with query expansion
    # ------------------------------------------------------------- #
    def _expand_query(self, query: str) -> str:
        """
        Expand queries with related keywords to improve semantic recall.
        """
        # Simple hardcoded expansions
        expansions = {
            "friends": ["companions", "classmates", "peers", "fellow students"],
            "school": ["Hogwarts", "lessons", "classes"],
            "family": ["parents", "relatives", "guardians"],
        }

        # Expand the query
        for key, words in expansions.items():
            if key in query.lower():
                query += " OR " + " OR ".join(words)
        return query


    # ------------------------------------------------------------- #
    # Build structured prompt
    # ------------------------------------------------------------- #
    def build_context_prompt(self, query, top_k=5, system_instructions_dict=None):
        """
        Build a context-rich prompt.
        The model is allowed to state clearly when context is missing.
        """
        query_expanded = self._expand_query(query)
        results = self.search(query=query_expanded, top_k=top_k)

        # If nothing retrieved at all
        if not results:
            return [
                {"role": "system", "content": "You are a factual assistant."},
                {"role": "user",
                 "content": f"No relevant context was found in the documents. "
                            f"Please reply exactly:\n"
                            f"The context provided does not contain the answer to the question:\n{query}"}
            ]
        # Build context text
        context_blocks = [
            f"📄 Source p.{r.get('page_number', '?')}:\n{r.get('text', '').strip()}"
            for r in results
        ]
        context_text = "\n\n".join(context_blocks)

        # Base system instructions
        base_system_instructions = """
        You are an expert reasoning assistant specialized in analyzing excerpts from text.
        Your goal is to answer questions using ONLY the information in the provided context.

        Guidelines:
        - Use only facts stated or strongly implied in the context.
        - If the answer cannot be found in the context, respond exactly with:
        "The provided context does not contain the answer."
        - Do NOT add or infer information from outside knowledge.
        - Be concise and direct (1–2 sentences maximum).
        - Never mention "context missing" unless truly absent or unrelated.
        """.strip()

        # Add any additional system instructions
        if system_instructions_dict:
            base_system_instructions += "\n\nAdditional instructions:\n" + str(system_instructions_dict)
        # Final messages
        messages = [
            {"role": "system", "content": base_system_instructions},
            {"role": "user",
             "content": f"Context:\n{context_text}\n\nQuestion:\n{query}\n\nAnswer:"}
        ]
        return messages

    def print_messages(self, messages):
        # Print messages for debugging
        for i, m in enumerate(messages, 1):
            pass
            # print(f"\n🔹 {i}. {m['role'].upper()} MESSAGE")
            # print("-" * 60)
            # print(m["content"])

    # ------------------------------------------------------------- #
    # Main query interface
    # ------------------------------------------------------------- #
    def ask(self, query, top_k=5, system_instructions_dict=None):
        """
        Ask a question and get an answer from Ollama model.

        Args:
            query: User question
            top_k: Number of chunks to retrieve
            system_instructions_dict: Additional system instructions

        Returns:
            Model's answer as string
        """
        # Build messages
        messages = self.build_context_prompt(query=query,
                                             top_k=top_k,
                                             system_instructions_dict=system_instructions_dict)

        try:
            # Call Ollama server via OpenAI-compatible API
            completion = self.client.chat.completions.create(
                model=self.model_id,
                messages=messages,
                temperature=0.2,
                max_tokens=300,
            )
            return completion.choices[0].message.content.strip() if completion.choices[0].message.content else ""
        except Exception as e:
            return f"Error calling Ollama server: {str(e)}\n\nPlease ensure Ollama is running and the model '{self.model_id}' is installed.\n\nInstall with: ollama pull {self.model_id}"


    # Get summary info from the ChromaDB collection
    def get_summary_info(self):
        """
        Get summary information from ChromaDB collection.
        """
        try:
            # Get collection count
            count = self.collection.count()

            if count == 0:
                return {"Chunks": 0, "Collection": "Empty", "Embedding Model": "all-mpnet-base-v2", "Dimension": "N/A"}

            # Get a sample to determine stats (including embeddings)
            sample = self.collection.get(limit=min(count, 100), include=["documents", "metadatas", "embeddings"])

            # Calculate average chunk length from sample
            texts = sample.get("documents", [])
            avg_length = round(sum(len(t) for t in texts) / len(texts), 1) if texts else "N/A"

            # Extract unique page numbers from metadata
            metadatas = sample.get("metadatas", [])
            pages = set(m.get("page_number") for m in metadatas if m.get("page_number"))
            num_pages = len(pages) if pages else "N/A"

            # Get embedding dimension from the first embedding
            embeddings = sample.get("embeddings", [])
            if embeddings is not None and len(embeddings) > 0:
                embedding_dim = len(embeddings[0])
            else:
                embedding_dim = "N/A"

            return {
                "Pages": num_pages,
                "Chunks": count,
                "Avg Chunk Length": avg_length,
                "Embedding Model": "all-mpnet-base-v2",
                "Dimension": embedding_dim
            }

        except Exception as e:
            import traceback
            print(f"Error in get_summary_info: {e}")
            print(traceback.format_exc())
            # Return a default dict instead of empty so UI shows something
            return {"Error": str(e)}



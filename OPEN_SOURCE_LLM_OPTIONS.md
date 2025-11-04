# Open Source LLM Options for Your RAG App

Your RAG application now supports **two LLM providers**: OpenAI and Ollama.

## Quick Comparison

| Provider | Best For | macOS Support | Setup Difficulty | Performance |
|----------|----------|---------------|------------------|-------------|
| **OpenAI** | Production, highest quality | ✅ Yes | ⭐ Easy (API key) | ⚡⚡⚡ Very Fast |
| **Ollama** | Local development, privacy | ✅ Yes | ⭐⭐ Easy (install app) | ⚡⚡ Fast |

## Why Ollama for Open Source?

**Ollama is the recommended choice** for running open-source LLMs locally:

1. **Easy installation** - Just download and install
2. **Works on macOS** - Optimized for both Intel and Apple Silicon
3. **Good performance** - Uses Metal acceleration on Mac
4. **Free and private** - Everything runs locally
5. **Active development** - Regular updates and new models

## Getting Started with Ollama

### 1. Install Ollama

```bash
# Download from https://ollama.com/download
# Or use Homebrew:
brew install ollama
```

### 2. Pull a Model

```bash
# Lightweight model (recommended for testing)
ollama pull llama3.2:3b

# More powerful model (if you have 16GB+ RAM)
ollama pull llama3.1:8b
```

### 3. Run Your App

```bash
uv run streamlit run app.py
```

### 4. Configure in UI

1. Select **"Ollama (Open Source)"** in the sidebar
2. Choose your model (e.g., `llama3.2:3b`)
3. Start asking questions!

## Complete Setup Guide

- **[OLLAMA_SETUP.md](OLLAMA_SETUP.md)** - Complete Ollama setup guide with detailed instructions

## Model Recommendations

### For Quick Testing (8GB RAM)
```bash
ollama pull llama3.2:3b  # ~2GB download
```

### For Better Quality (16GB+ RAM)
```bash
ollama pull llama3.1:8b  # ~4.7GB download
```

### Alternative Models
```bash
ollama pull mistral:7b   # ~4.1GB download
ollama pull qwen2.5:7b   # ~4.4GB download
```

## Cost Comparison

### OpenAI (gpt-4o-mini)
- **Input**: $0.150 per 1M tokens
- **Output**: $0.600 per 1M tokens
- **Typical RAG query**: ~$0.001-0.003 per query

### Ollama (llama3.2:3b)
- **Cost**: $0 (free)
- **Hardware**: Uses your existing Mac
- **Privacy**: All data stays local

## When to Use What?

### Use OpenAI if:
- ✅ You need the highest quality answers
- ✅ You're building a production application
- ✅ You don't mind cloud-based processing
- ✅ You want the fastest development experience

### Use Ollama if:
- ✅ You're on macOS (works on both Intel and Apple Silicon)
- ✅ You want privacy/local processing
- ✅ You're prototyping or developing
- ✅ You want to experiment with different models
- ✅ You don't want to pay per query
- ✅ You have a stable internet connection to download models (only needed once)

## Quality Comparison

Based on typical RAG tasks:

1. **OpenAI GPT-4o** - Best quality, most accurate
2. **OpenAI GPT-4o-mini** - Very good, cost-effective
3. **Llama 3.1 (8B)** - Good quality, free
4. **Mistral (7B)** - Good quality, free
5. **Llama 3.2 (3B)** - Decent quality, very fast, free

## Privacy Comparison

| Provider | Data Location | Internet Required | Data Retention |
|----------|---------------|-------------------|----------------|
| OpenAI | OpenAI servers | Yes | Per OpenAI policy |
| Ollama | Your Mac | No (after model download) | Never leaves your device |

## Troubleshooting

### Ollama won't install on macOS
- Check macOS version (requires macOS 11 Big Sur or later)
- Download directly from [ollama.com](https://ollama.com)
- For older macOS versions, check Ollama documentation for compatibility

### Connection Error in Streamlit
- Make sure Ollama is running (it starts automatically after install)
- Check that the server URL is correct: `http://localhost:11434/v1`
- Restart Ollama if needed

### Out of memory with Ollama
```bash
# Use a smaller model
ollama pull llama3.2:3b

# Close other applications to free up RAM
```

### Slow responses
- Switch to a smaller model (3B instead of 8B)
- Close background applications
- On Intel Macs, performance will be slower than Apple Silicon

## Next Steps

1. ✅ Install Ollama: [OLLAMA_SETUP.md](OLLAMA_SETUP.md)
2. ✅ Pull a model: `ollama pull llama3.2:3b`
3. ✅ Run your app: `uv run streamlit run app.py`
4. ✅ Select "Ollama (Open Source)" in the UI
5. ✅ Start querying your documents!

---

**Have questions?** Check the setup guides or open an issue on GitHub.

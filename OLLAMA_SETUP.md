# Ollama Setup Guide (Recommended for macOS)

This guide explains how to use open-source LLMs with your RAG application via Ollama - the easiest way to run LLMs locally on macOS.

## What is Ollama?

Ollama is a simple, lightweight tool for running large language models locally. It provides:
- **Easy installation** - Single command install
- **macOS optimized** - Works on both Intel and Apple Silicon Macs
- **OpenAI-compatible API** - Drop-in replacement for OpenAI
- **Model management** - Simple pull/list/remove commands
- **No GPU required** - Runs on CPU (faster on Apple Silicon with Metal)

## Why Ollama for macOS?

✅ **Works on macOS** - Unlike vLLM which requires NVIDIA CUDA
✅ **Simple setup** - No complex dependencies
✅ **Good performance** - Optimized for Apple Silicon (M1/M2/M3)
✅ **Free & Private** - All data stays local
✅ **Active community** - Well-maintained and documented

## Installation

### Step 1: Install Ollama

**Option A: Download from website** (Recommended)
1. Visit [https://ollama.com/download](https://ollama.com/download)
2. Download the macOS installer
3. Open the .dmg file and drag Ollama to Applications

**Option B: Using Homebrew**
```bash
brew install ollama
```

### Step 2: Verify Installation

```bash
ollama --version
```

### Step 3: Start Ollama Service

Ollama runs as a background service. It should start automatically after installation.

To manually start:
```bash
ollama serve
```

## Pulling Models

Before using a model, you need to download it:

### Recommended Models for RAG

```bash
# Lightweight model (3B parameters, ~2GB)
ollama pull llama3.2:3b

# Balanced model (8B parameters, ~4.7GB)
ollama pull llama3.1:8b

# Alternative models
ollama pull mistral:7b        # 7B parameters, ~4.1GB
ollama pull qwen2.5:7b        # 7B parameters, ~4.4GB
```

### Model Selection Guide

| Model | Size | Download | RAM Needed | Best For |
|-------|------|----------|------------|----------|
| `llama3.2:3b` | 3B | ~2GB | 8GB+ | Fast responses, prototyping |
| `llama3.1:8b` | 8B | ~4.7GB | 16GB+ | Better reasoning, more accurate |
| `mistral:7b` | 7B | ~4.1GB | 16GB+ | Strong general performance |
| `qwen2.5:7b` | 7B | ~4.4GB | 16GB+ | Excellent instruction following |

### List Downloaded Models

```bash
ollama list
```

### Remove Models

```bash
ollama rm llama3.2:3b
```

## Using Ollama with Your RAG App

### Step 1: Pull a Model

```bash
ollama pull llama3.2:3b
```

### Step 2: Run Your Streamlit App

```bash
uv run streamlit run app.py
```

### Step 3: Configure in UI

1. In the sidebar, select **"Ollama (Open Source)"** as LLM Provider
2. Choose your model (e.g., `llama3.2:3b`)
3. Verify server URL is `http://localhost:11434/v1`
4. Upload PDFs and ask questions!

## Testing Ollama

### Test via Command Line

```bash
# Simple test
ollama run llama3.2:3b "What is RAG?"

# Exit with /bye
```

### Test via API

```bash
# Check if server is running
curl http://localhost:11434/api/tags

# Test chat completion
curl http://localhost:11434/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama3.2:3b",
    "messages": [
      {"role": "user", "content": "What is retrieval-augmented generation?"}
    ],
    "temperature": 0.2,
    "max_tokens": 100
  }'
```

## Performance Tips

### For Apple Silicon (M1/M2/M3)
- Ollama automatically uses Metal for GPU acceleration
- 8B models run smoothly on 16GB+ RAM
- 3B models run well on 8GB RAM

### For Intel Macs
- Performance is slower (CPU-only)
- Stick to smaller models (3B recommended)
- Close other applications to free up RAM

### Optimize Performance

```bash
# Set number of CPU threads (default: auto)
OLLAMA_NUM_THREADS=8 ollama serve

# Set context window size (default: 2048)
OLLAMA_CONTEXT_SIZE=4096 ollama serve
```

## Troubleshooting

### Ollama Not Starting

```bash
# Check if Ollama is running
ps aux | grep ollama

# Start manually
ollama serve
```

### Connection Refused Error

```bash
# Make sure Ollama service is running
ollama serve

# Check if port 11434 is in use
lsof -i :11434
```

### Model Not Found

```bash
# List available models
ollama list

# Pull the model first
ollama pull llama3.2:3b
```

### Slow Responses

- **Use smaller models**: Switch from 8B to 3B
- **Close other apps**: Free up RAM
- **Reduce context**: Lower the `top_k` parameter in Streamlit
- **Apple Silicon only**: Make sure Metal is enabled (automatic)

### Out of Memory

```bash
# Use a smaller model
ollama pull llama3.2:3b

# Or reduce context window
OLLAMA_CONTEXT_SIZE=2048 ollama serve
```

## Ollama vs OpenAI

| Feature | OpenAI | Ollama |
|---------|--------|--------|
| **Setup** | API key only | Simple install |
| **macOS Support** | ✅ Yes | ✅ Yes |
| **Cost** | Pay per token | Free |
| **Privacy** | Cloud-based | Local |
| **Speed** | Very fast | Fast (especially on M-series) |
| **Quality** | Excellent | Good |
| **Best For** | Production, highest quality | Local development, privacy |

## Advanced Usage

### Running Multiple Models

```bash
# Ollama can run one model at a time
# To switch models, just select a different one in the UI
```

### Custom Models

```bash
# Create a Modelfile
cat > Modelfile << EOF
FROM llama3.2:3b
SYSTEM You are an expert at analyzing documents.
EOF

# Create custom model
ollama create my-custom-model -f Modelfile

# Use in the app by manually typing "my-custom-model" in the UI
```

### API Configuration

By default, Ollama uses:
- **Host**: `localhost`
- **Port**: `11434`
- **API Base**: `http://localhost:11434/v1`

To change:
```bash
OLLAMA_HOST=0.0.0.0:8080 ollama serve
```

## Common Commands

```bash
# List all commands
ollama --help

# Pull a model
ollama pull <model-name>

# List downloaded models
ollama list

# Run a model interactively
ollama run <model-name>

# Remove a model
ollama rm <model-name>

# Show model information
ollama show <model-name>

# Start server
ollama serve

# Check version
ollama --version
```

## Model Library

Browse all available models at: [https://ollama.com/library](https://ollama.com/library)

Popular models for RAG:
- **Llama 3.2** - Latest Meta model, excellent quality
- **Llama 3.1** - Strong reasoning, good for complex questions
- **Mistral** - Fast and efficient
- **Qwen 2.5** - Great at following instructions
- **Phi-3** - Small but capable (3.8B)
- **Gemma** - Google's open model

## Best Practices

1. **Start small** - Begin with `llama3.2:3b` to test
2. **Monitor RAM** - Use Activity Monitor to check memory usage
3. **One model at a time** - Ollama loads one model into memory
4. **Keep models updated** - Regularly check for new versions
5. **Clean up unused models** - Remove models you don't use to save space

## Resources

- [Ollama Website](https://ollama.com)
- [Ollama GitHub](https://github.com/ollama/ollama)
- [Ollama Model Library](https://ollama.com/library)
- [Ollama Discord Community](https://discord.gg/ollama)

## Next Steps

1. ✅ Install Ollama
2. ✅ Pull a model (`ollama pull llama3.2:3b`)
3. ✅ Start your Streamlit app
4. ✅ Select "Ollama (Open Source)" in the UI
5. ✅ Upload PDFs and start querying!

---

**Questions?** Check the [Ollama GitHub Discussions](https://github.com/ollama/ollama/discussions) or open an issue.

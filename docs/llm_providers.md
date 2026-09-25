# LLM Provider Architecture

The application now uses a provider abstraction layer so the rest of the system can stay provider-agnostic.

## Supported providers

- Ollama (default for local development)
- Gemini
- Groq
- OpenAI
- Claude
- DeepSeek

## Switching providers

Change only the environment variable:

```env
LLM_PROVIDER=ollama
```

For Gemini:

```env
LLM_PROVIDER=gemini
GEMINI_API_KEY=your-key
```

## Local development with Ollama

1. Install and start Ollama.
2. Pull the default model:

```bash
ollama pull qwen3:8b
```

3. Set the environment variables:

```env
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=qwen3:8b
LLM_TIMEOUT=120
LLM_MAX_TOKENS=4096
TEMPERATURE=0.2
TOP_P=0.9
TOP_K=40
```

## Notes

- The same prompt registry and conversation memory flow are used regardless of provider.
- Streaming, retries, timeouts, and friendly fallback errors are handled inside the provider layer.
- The health endpoint will report provider readiness based on the selected backend.

# Model Configuration Examples

This document shows how to configure different LLM providers and models using environment variables.

## Configuration Options

### Environment Variables

| Variable | Description | Default | Options |
|----------|-------------|---------|---------|
| `LLM_PROVIDER` | Which LLM provider to use | `gemini` | `gemini`, `claude`, `grok` |
| `LLM_TEMPERATURE` | Temperature for model responses | `0.1` | `0.0` to `1.0` |
| `GOOGLE_API_KEY` | Google Gemini API key | - | Your API key |
| `GEMINI_MODEL` | Gemini model to use | `gemini-2.0-flash` | `gemini-2.0-flash`, `gemini-1.5-pro`, etc. |
| `ANTHROPIC_API_KEY` | Anthropic Claude API key | - | Your API key |
| `CLAUDE_MODEL` | Claude model to use | `claude-3-5-sonnet-20241022` | `claude-3-5-sonnet-20241022`, `claude-3-haiku-20240307`, etc. |
| `GROK_API_KEY` | xAI Grok API key | - | Your API key |
| `GROK_MODEL` | Grok model to use | `grok-beta` | `grok-beta`, `grok-2`, etc. |

## Configuration Examples

### 1. Using Gemini (Default)

```bash
# .env file
LLM_PROVIDER=gemini
LLM_TEMPERATURE=0.1
GOOGLE_API_KEY=your_gemini_api_key_here
GEMINI_MODEL=gemini-2.0-flash
```

### 2. Using Claude

```bash
# .env file
LLM_PROVIDER=claude
LLM_TEMPERATURE=0.1
ANTHROPIC_API_KEY=your_claude_api_key_here
CLAUDE_MODEL=claude-3-5-sonnet-20241022
```

### 3. Using Claude Haiku (Faster, Cheaper)

```bash
# .env file
LLM_PROVIDER=claude
LLM_TEMPERATURE=0.1
ANTHROPIC_API_KEY=your_claude_api_key_here
CLAUDE_MODEL=claude-3-haiku-20240307
```

### 4. Using Gemini Pro (More Capable)

```bash
# .env file
LLM_PROVIDER=gemini
LLM_TEMPERATURE=0.1
GOOGLE_API_KEY=your_gemini_api_key_here
GEMINI_MODEL=gemini-1.5-pro
```

### 5. Using Grok

```bash
# .env file
LLM_PROVIDER=grok
LLM_TEMPERATURE=0.1
GROK_API_KEY=your_grok_api_key_here
GROK_MODEL=grok-beta
```

## Model Comparison

### Gemini Models
- **gemini-2.0-flash**: Fast, efficient, good for most tasks
- **gemini-1.5-pro**: More capable, better for complex reasoning
- **gemini-1.5-flash**: Fast, good balance of speed and capability

### Claude Models
- **claude-3-5-sonnet-20241022**: Most capable, best for complex tasks
- **claude-3-haiku-20240307**: Fast, cost-effective, good for simple tasks
- **claude-3-opus-20240229**: Most powerful, best for advanced reasoning

### Grok Models
- **grok-beta**: Current beta version, good for most tasks
- **grok-2**: Latest version when available

## Switching Models

To switch between models, simply update your `.env` file:

```bash
# Switch from Gemini to Claude
LLM_PROVIDER=claude
ANTHROPIC_API_KEY=your_new_api_key

# Switch to Grok
LLM_PROVIDER=grok
GROK_API_KEY=your_grok_api_key

# Switch back to Gemini
LLM_PROVIDER=gemini
GOOGLE_API_KEY=your_gemini_key
```

## Temperature Settings

- **0.0**: Most deterministic, consistent responses
- **0.1**: Good balance (default)
- **0.5**: More creative, varied responses
- **1.0**: Most creative, unpredictable responses

## API Key Setup

### Getting Gemini API Key
1. Go to [Google AI Studio](https://aistudio.google.com/)
2. Create a new API key
3. Add to your `.env` file: `GOOGLE_API_KEY=your_key_here`

### Getting Claude API Key
1. Go to [Anthropic Console](https://console.anthropic.com/)
2. Create a new API key
3. Add to your `.env` file: `ANTHROPIC_API_KEY=your_key_here`

### Getting Grok API Key
1. Go to [xAI Console](https://console.x.ai/)
2. Create a new API key
3. Add to your `.env` file: `GROK_API_KEY=your_key_here`

## Usage in Code

The model factory automatically handles the configuration:

```python
from adapters.model_factory import ModelFactory

# Create LLM instances based on environment configuration
llm = ModelFactory.create_llm()
langchain_llm = ModelFactory.create_langchain_llm()

# Use the LLM
response = llm.invoke("Your prompt here")
```

## Troubleshooting

### Common Issues

1. **Missing API Key**: Ensure you have the correct API key for your chosen provider
2. **Invalid Model Name**: Check the model name is correct for your provider
3. **Rate Limits**: Some models have different rate limits and costs

### Error Messages

- `ANTHROPIC_API_KEY environment variable is not set`: Add your Claude API key to `.env`
- `GROK_API_KEY environment variable is not set`: Add your Grok API key to `.env`
- `GOOGLE_API_KEY environment variable is not set`: Add your Gemini API key to `.env`
- `Invalid model name`: Check the model name in your `.env` file 
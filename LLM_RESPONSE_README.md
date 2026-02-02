# LLM Response Generator

A script to process transcribed text through OpenAI's API to generate short, conversational responses.

## Features

- **Loads API key from .env file** - Automatically loads `OPENAI_API_KEY` from your `.env` file
- **Streaming support** - Stream responses in real-time or get complete response at once
- **Conversational responses** - Configured to return short, friendly, natural responses (1-3 sentences)
- **File processing** - Process transcription files from the `transcriptions/` directory
- **Direct text input** - Send text directly from command line
- **Configurable temperature** - Control randomness of responses (0.0-2.0)

## Installation

The required dependencies are already added to `pyproject.toml`:
- `openai>=1.0.0` - OpenAI API client
- `python-dotenv>=1.0.0` - Load environment variables from .env

Install them with:
```bash
pip install -e .
```

## Usage

### Process the latest transcription file (non-streaming)
```bash
python llm_response.py
```

### Process specific transcription file
```bash
python llm_response.py --file transcriptions/transcription_20260111_142157.txt
```

### Send direct text input
```bash
python llm_response.py "What is the weather like today?"
```

### Stream the response
```bash
python llm_response.py "Hello, how are you?" --stream
```

### Adjust response randomness
```bash
python llm_response.py "Tell me a joke" --temperature 0.9
```

### Use a different OpenAI model
```bash
python llm_response.py "Your text here" --model gpt-4
```

### Help
```bash
python llm_response.py --help
```

## Command-line Options

- `text` - Text to send to LLM (optional, reads from latest transcription if not provided)
- `-f, --file` - Path to transcription file to process
- `-s, --stream` - Stream the response instead of waiting for complete response
- `-t, --temperature` - Temperature for response generation (0.0-2.0, default: 0.7)
- `-m, --model` - OpenAI model to use (default: gpt-4o-mini)

## Python API

```python
from llm_response import LLMResponseGenerator

# Initialize
llm = LLMResponseGenerator()

# Get non-streaming response
response = llm.get_response("Your text here", stream=False)
print(response)

# Get streaming response
for chunk in llm.get_response("Your text here", stream=True):
    print(chunk, end="", flush=True)

# Process a transcription file
llm.process_transcription_file("path/to/file.txt", stream=True)
```

## Integration with speak_bot

You can easily integrate this with the main voice-to-text pipeline:

```python
from main import run_voice_to_text
from llm_response import LLMResponseGenerator

# Record and transcribe
transcribed_text, file_path = run_voice_to_text()

# Get LLM response
llm = LLMResponseGenerator()
response = llm.process_transcription_file(file_path, stream=True)
```

## Notes

- The script is configured to return short, conversational responses by default
- You can customize the system prompt in the `LLMResponseGenerator.__init__()` method if needed
- The default model is `gpt-4o-mini` which offers a good balance of speed and cost
- Streaming is useful for better UX with longer responses

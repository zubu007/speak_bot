This is a project speaking bot that can interact with users through voice commands and responses. It utilizes advanced speech recognition and synthesis technologies to provide a seamless conversational experience. The bot can be integrated into various applications, including customer service, virtual assistants, and interactive voice response systems.

## Steps (concept)
1. Using the mic to capture user voice input.
2. Converting the voice input to text using speech recognition.
3. Processing the text input to generate a relevant response using a LLM (Large Language Model).
4. Converting the text response back to speech using text-to-speech synthesis.
5. Playing the synthesized speech back to the user through speakers.

## Quick Start

This project uses `uv` for environment and package management.

1. **Create and activate the virtual environment:**
   ```bash
   uv venv
   source .venv/bin/activate
   ```

2. **Install dependencies:**
   ```bash
   uv pip install .
   ```

3. **Run the integrated pipeline (record + transcribe):**
   ```bash
   python -m speak_bot.main --loop --use-llm
   ```
   
   *Alternatively, after setting up the global CLI command (see below), you can use:*
   ```bash
   talk --loop --use-llm
   ```

This will:
1. Record audio from your microphone until 2 seconds of silence
2. Transcribe the audio using faster-whisper (optimized Whisper implementation)
3. Save the transcription to `transcriptions/transcription_<timestamp>.txt`
4. (Optional) Generate an LLM response and show/stream it
5. Repeat the loop until you press `q` then Enter during recording

## Global CLI Command Setup

For convenience, you can install the speak-bot as a global command that can be used from anywhere on your system. This allows you to run `talk` from any directory instead of having to navigate to the project folder and use `python -m speak_bot.main`.

### Installation Steps

1. **Ensure you're in the project directory:**
   ```bash
   cd /path/to/speak_bot
   ```

2. **Install the package globally using UV:**
   ```bash
   uv pip install -e .
   ```
   
   The `-e` flag installs in "editable" mode, meaning changes to your code will be immediately reflected when running the command.

3. **Verify installation:**
   ```bash
   talk --help
   ```

### Usage

Once installed, you can use the `talk` command from anywhere on your system:

```bash
# Basic usage (from any directory)
talk

# With LLM integration
talk --use-llm --llm-stream

# Full conversation mode with speech synthesis
talk --use-llm --use-tts --loop

# Custom settings
talk --model large --silence-seconds 3.0 --language en

# Output to specific directory
talk --output-dir ~/my-transcriptions --use-llm
```

### Benefits

- **Global Access**: Works from any directory on your system
- **Same Features**: All command-line options from the original script are available
- **Editable Install**: Code changes automatically reflect in the global command
- **Clean Integration**: Uses standard Python packaging conventions
- **Smart Path Resolution**: TTS models and other resources are automatically located regardless of your current directory

### Important Notes

- **TTS Models**: The default TTS model path (`tts_models/jarvis-medium.onnx`) is automatically resolved to the correct location when using the global command, so TTS features work from any directory.
- **Output Files**: Transcriptions and audio files are saved relative to your current working directory when you run the command.

### Uninstalling

To remove the global command:

```bash
uv pip uninstall speak-bot
```

### Options

The `main.py` script provides several command-line arguments to customize the behavior of the speaking bot:

- `-o, --output-dir`: Directory to save transcription files. Default: `transcriptions`.
- `-m, --model`: faster-whisper model size. Choices: `tiny`, `base`, `small`, `medium`, `large`. Default: `tiny`.
- `-r, --samplerate`: Sample rate in Hz. Default: `16000`.
- `--silence-seconds`: Stop after this many seconds of silence. Default: `2.0`.
- `--silence-threshold`: Amplitude threshold for silence. Default: `1500`.
- `-l, --language`: Language code (e.g., 'en', 'es'). Auto-detect if omitted.
- `--use-llm`: Process transcription with LLM to generate a response.
- `--llm-model`: OpenAI model to use. Default: `gpt-4o-mini`.
- `--llm-stream`: Stream the LLM response.
- `--use-tts`: Convert LLM response to speech and play it.
- `--tts-model`: Path to TTS ONNX model. Default: `tts_models/jarvis-medium.onnx`.
- `--loop`: Keep recording in a conversation loop until 'q' + Enter is pressed.

**Examples:**

Choose a different faster-whisper model (tiny, base, small, medium, large):
```bash
# Using python module
python -m speak_bot.main --model small

# Using global command (after installation)
talk --model small
```

Customize silence detection:
```bash
# Using python module
python -m speak_bot.main --silence-seconds 3 --silence-threshold 1200

# Using global command (after installation)
talk --silence-seconds 3 --silence-threshold 1200
```

Specify language (for faster transcription):
```bash
# Using python module
python -m speak_bot.main --language en

# Using global command (after installation)
talk --language en
```

Save to a custom directory:
```bash
# Using python module
python -m speak_bot.main --output-dir my_transcriptions

# Using global command (after installation)
talk --output-dir my_transcriptions
```

## Modular Components

### Record Audio (Step 1)

The `record_audio.py` module captures mic input and stops after continuous silence.

**Standalone usage:**

```bash
python record_audio.py --output recording.wav
```

**As a module:**

```python
from record_audio import record_audio_until_silence

audio_data, samplerate = record_audio_until_silence(
    samplerate=16000,
    silence_seconds=2.0,
    silence_threshold=1500
)
```

### Speech to Text (Step 2)

The `speech_to_text.py` module transcribes audio using faster-whisper, an optimized implementation of OpenAI Whisper that provides 4x faster performance with lower memory usage.

**Standalone usage:**

```bash
python speech_to_text.py input.wav --output transcription.txt --model tiny
```

**As a module:**

```python
from speech_to_text import transcribe_audio

text = transcribe_audio(
    audio_data,
    samplerate,
    model_name="tiny",
    language="en"
)
```

## macOS Microphone Permission

On macOS, grant microphone access to your terminal/VS Code:
System Settings → Privacy & Security → Microphone → enable for your app.

## Tool Development

The speak_bot system uses the Model Context Protocol (MCP) for tool integration, allowing the assistant to call functions like stopping conversations, managing calendar events, and reading emails.

**For detailed information on adding new tools, see [TOOL_DEVELOPMENT_GUIDE.md](TOOL_DEVELOPMENT_GUIDE.md).**

Key features:
- Tools are defined via MCP with comprehensive descriptions
- The LLM automatically learns tool usage from descriptions (no system prompt modifications needed)
- Scalable architecture supporting unlimited tools
- Current tools: `stop_conversation` (say "goodbye" to end conversation)
- Planned tools: Calendar management, email reading/sending

## Useful links
- https://huggingface.co/rhasspy/piper-voices/tree/main/en/en_US
- https://github.com/OHF-Voice/piper1-gpl/blob/main/docs/API_PYTHON.md
- https://modelcontextprotocol.io/ (MCP documentation)

## Plans for future improvements:
- ✅ JARVIS-like wake word detection (implemented in wake_word project)
- ⏳ Persistant Conversation history for context-aware responses (implemented)
- ⏳ Add Google Calendar integration for scheduling via voice commands
- ⏳ Add email reading and management tools
- ⏳ Create a C++ version for performance optimization
- ⏳ Add german language support

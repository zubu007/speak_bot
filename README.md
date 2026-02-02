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

This will:
1. Record audio from your microphone until 2 seconds of silence
2. Transcribe the audio using Whisper
3. Save the transcription to `transcriptions/transcription_<timestamp>.txt`
4. (Optional) Generate an LLM response and show/stream it
5. Repeat the loop until you press `q` then Enter during recording

### Options

Choose a different Whisper model (tiny, base, small, medium, large):

```bash
python main.py --model small
```

Customize silence detection:

```bash
python main.py --silence-seconds 3 --silence-threshold 1200
```

Specify language (for faster transcription):

```bash
python main.py --language en
```

Save to a custom directory:

```bash
python main.py --output-dir my_transcriptions
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

The `speech_to_text.py` module transcribes audio using OpenAI Whisper.

**Standalone usage:**

```bash
python speech_to_text.py input.wav --output transcription.txt --model base
```

**As a module:**

```python
from speech_to_text import transcribe_audio

text = transcribe_audio(
    audio_data,
    samplerate,
    model_name="base",
    language="en"
)
```

## macOS Microphone Permission

On macOS, grant microphone access to your terminal/VS Code:
System Settings → Privacy & Security → Microphone → enable for your app.

## Useful links
- https://huggingface.co/rhasspy/piper-voices/tree/main/en/en_US
- https://github.com/OHF-Voice/piper1-gpl/blob/main/docs/API_PYTHON.md

## Plans for future improvements:
- Add JARVIS-like wake word detection.
- Implement conversation history for context-aware responses.
- Create a C++ version for performance optimization.
- Add german language support.
- Add Google Calendar integration for scheduling via voice commands.

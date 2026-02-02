# Text-to-Speech with Piper TTS

This module provides text-to-speech functionality using Piper TTS with ONNX models.

## Installation

Install the required Piper TTS library:

```bash
pip install piper-tts
```

## Usage

### Command Line

Convert text to speech from the command line:

```bash
# Basic usage
python text_to_speech.py "Hello, this is a test"

# Specify output file
python text_to_speech.py "Hello world" -o my_audio.wav

# Use a different model
python text_to_speech.py "Hello" -m tts_models/en_US-amy-low.onnx
```

### Python API

Use the module in your Python code:

```python
from text_to_speech import text_to_speech, PiperTTS

# Simple function call
output_file = text_to_speech("Hello, world!", output_path="output/hello.wav")

# Using the class for more control
tts = PiperTTS(model_path="tts_models/en_US-amy-low.onnx")
audio_array, sample_rate = tts.synthesize(
    text="This is a longer text to convert to speech",
    output_path="output/speech.wav"
)
```

## Features

- **ONNX Model Support**: Uses efficient ONNX models for fast inference
- **WAV Output**: Generates standard WAV audio files
- **Streaming Synthesis**: Processes audio in chunks for memory efficiency
- **Simple API**: Easy-to-use function and class interfaces

## Model

The script uses the `en_US-amy-low.onnx` model located in the `tts_models/` directory. You can download additional Piper TTS models from the [Piper repository](https://github.com/rhasspy/piper).

## Parameters

- `text`: The text to convert to speech
- `output_path`: Path where the WAV file will be saved (default: "output/speech.wav")
- `model_path`: Path to the ONNX model file (default: "tts_models/en_US-amy-low.onnx")

## Output

The script generates a WAV file with the following specifications:
- Format: WAV
- Channels: 1 (Mono)
- Sample Width: 16-bit
- Sample Rate: Depends on the model (typically 22050 Hz)

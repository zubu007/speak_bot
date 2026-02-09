import argparse
import io
import wave

import numpy as np
from faster_whisper import WhisperModel


def transcribe_audio(
    audio_data: np.ndarray,
    samplerate: int,
    model_name: str = "tiny",
    language: str | None = None,
    verbose: bool = True,
) -> str:
    """Transcribe audio data using faster-whisper.
    
    Args:
        audio_data: int16 numpy array
        samplerate: sample rate of the audio
        model_name: Whisper model size (tiny, base, small, medium, large)
        language: Optional language code (e.g., 'en', 'es')
        verbose: Print progress messages
    
    Returns:
        Transcribed text
    """
    if verbose:
        print(f"Loading faster-whisper model: {model_name}")
    model = WhisperModel(model_name)
    
    # Convert int16 to float32 normalized to [-1, 1]
    audio_float = audio_data.astype(np.float32) / 32768.0
    
    # Whisper expects mono audio at 16kHz
    # If audio is stereo, convert to mono
    if audio_float.ndim > 1 and audio_float.shape[1] > 1:
        audio_float = audio_float.mean(axis=1)
    else:
        audio_float = audio_float.flatten()
    
    # Resample to 16kHz if needed
    if samplerate != 16000:
        if verbose:
            print(f"Resampling from {samplerate}Hz to 16000Hz")
        import scipy.signal
        num_samples = int(len(audio_float) * 16000 / samplerate)
        audio_float = scipy.signal.resample(audio_float, num_samples)
    
    if verbose:
        print("Transcribing audio...")
    
    # faster-whisper returns (segments, info) where segments is an iterator
    segments, info = model.transcribe(
        audio_float,
        language=language,
    )
    
    # Convert segments iterator to text
    text = "".join([segment.text for segment in segments]).strip()
    
    if verbose:
        print(f"Transcription complete: {len(text)} characters")
    
    return text


def transcribe_wav_file(
    wav_path: str,
    model_name: str = "tiny",
    language: str | None = None,
    verbose: bool = True,
) -> str:
    """Transcribe a WAV file using faster-whisper.
    
    Args:
        wav_path: Path to WAV file
        model_name: Whisper model size
        language: Optional language code
        verbose: Print progress messages
    
    Returns:
        Transcribed text
    """
    with wave.open(wav_path, "rb") as wf:
        samplerate = wf.getframerate()
        channels = wf.getnchannels()
        frames = wf.readframes(wf.getnframes())
        audio_data = np.frombuffer(frames, dtype=np.int16)
        
        if channels > 1:
            audio_data = audio_data.reshape(-1, channels)
    
    return transcribe_audio(audio_data, samplerate, model_name, language, verbose)


def parse_args():
    parser = argparse.ArgumentParser(description="Transcribe WAV file to text using faster-whisper.")
    parser.add_argument(
        "input",
        help="Input WAV file path",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Output text file path (default: print to stdout)",
    )
    parser.add_argument(
        "-m",
        "--model",
        default="tiny",
        choices=["tiny", "base", "small", "medium", "large"],
        help="faster-whisper model size (default: tiny)",
    )
    parser.add_argument(
        "-l",
        "--language",
        default=None,
        help="Language code (e.g., 'en', 'es'). Auto-detect if omitted.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    text = transcribe_wav_file(
        args.input,
        model_name=args.model,
        language=args.language,
    )
    
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"Saved transcription to: {args.output}")
    else:
        print("\n--- Transcription ---")
        print(text)


if __name__ == "__main__":
    main()

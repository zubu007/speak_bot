import argparse
import io
import wave

import numpy as np
from faster_whisper import WhisperModel


class WhisperModelCache:
    """Singleton cache for Whisper models to avoid reloading.
    
    This significantly improves performance by keeping models in memory
    across multiple transcription requests. First load takes normal time,
    subsequent loads are instant.
    """
    _models: dict[str, WhisperModel] = {}
    
    @classmethod
    def get_model(
        cls, 
        model_name: str = "tiny.en",
        device: str = "cpu",
        compute_type: str = "int8"
    ) -> WhisperModel:
        """Get a cached Whisper model or create a new one.
        
        Args:
            model_name: Whisper model size (tiny, tiny.en, base, small, medium, large)
            device: Device to run on (cpu or cuda). Apple Silicon uses cpu with Core ML.
            compute_type: Compute precision (int8, float16, float32)
            
        Returns:
            Cached or newly created WhisperModel instance
        """
        cache_key = f"{model_name}_{device}_{compute_type}"
        if cache_key not in cls._models:
            print(f"Loading Whisper model: {model_name} (device={device}, compute_type={compute_type})")
            cls._models[cache_key] = WhisperModel(
                model_name, 
                device=device, 
                compute_type=compute_type
            )
            print("✓ Whisper model loaded and cached")
        return cls._models[cache_key]
    
    @classmethod
    def clear_cache(cls):
        """Clear all cached models to free memory."""
        cls._models.clear()


def transcribe_audio(
    audio_data: np.ndarray,
    samplerate: int,
    model_name: str = "tiny.en",
    language: str | None = "en",
    device: str = "cpu",
    compute_type: str = "int8",
    use_cache: bool = True,
    verbose: bool = True,
) -> str:
    """Transcribe audio data using faster-whisper.
    
    Args:
        audio_data: int16 numpy array
        samplerate: sample rate of the audio
        model_name: Whisper model size (tiny, tiny.en, base, small, medium, large)
                    Default: tiny.en for faster English-only transcription
        language: Optional language code (e.g., 'en', 'es'). Default: 'en'
        device: Device to run on (cpu or cuda). Apple Silicon uses cpu with Core ML.
        compute_type: Compute precision (int8, float16, float32). Default: int8
        use_cache: Use cached model if available (recommended for better performance)
        verbose: Print progress messages
    
    Returns:
        Transcribed text
    """
    if verbose:
        print(f"Transcribing with model: {model_name}")
    
    # Use cached model if enabled (recommended for performance)
    if use_cache:
        model = WhisperModelCache.get_model(model_name, device, compute_type)
    else:
        if verbose:
            print(f"Loading faster-whisper model: {model_name}")
        model = WhisperModel(model_name, device=device, compute_type=compute_type)
    
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
    model_name: str = "tiny.en",
    language: str | None = "en",
    device: str = "cpu",
    compute_type: str = "int8",
    use_cache: bool = True,
    verbose: bool = True,
) -> str:
    """Transcribe a WAV file using faster-whisper.
    
    Args:
        wav_path: Path to WAV file
        model_name: Whisper model size (default: tiny.en for faster English-only)
        language: Optional language code (default: en)
        device: Device to run on (cpu or cuda)
        compute_type: Compute precision (int8, float16, float32)
        use_cache: Use cached model if available
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
    
    return transcribe_audio(
        audio_data, samplerate, model_name, language, 
        device, compute_type, use_cache, verbose
    )


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
        default="tiny.en",
        choices=["tiny", "tiny.en", "base", "base.en", "small", "small.en", "medium", "medium.en", "large"],
        help="faster-whisper model size (default: tiny.en for faster English-only)",
    )
    parser.add_argument(
        "-l",
        "--language",
        default="en",
        help="Language code (e.g., 'en', 'es'). Default: en",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device to run on (default: cpu, uses Core ML on Apple Silicon)",
    )
    parser.add_argument(
        "--compute-type",
        default="int8",
        choices=["int8", "float16", "float32"],
        help="Compute precision (default: int8, good for Apple Silicon)",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable model caching (default: caching enabled)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    text = transcribe_wav_file(
        args.input,
        model_name=args.model,
        language=args.language,
        device=args.device,
        compute_type=args.compute_type,
        use_cache=not args.no_cache,
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

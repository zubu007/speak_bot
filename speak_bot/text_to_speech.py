import wave
from pathlib import Path

from piper import PiperVoice


def text_to_speech(
    text: str,
    output_file: str,
    model_path: str = "tts_models/en_US-amy-low.onnx",
    use_cuda: bool = True,
    verbose: bool = True,
) -> None:
    """Convert text to speech and save as a WAV file using Piper TTS.
    
    Args:
        text: The text to synthesize
        output_file: Path where the WAV file will be saved
        model_path: Path to the Piper TTS model file
        use_cuda: Whether to use GPU acceleration
        verbose: If True, print status messages
    """
    if verbose:
        print(f"Loading Piper TTS model from: {model_path}")
    
    voice = PiperVoice.load(model_path, use_cuda=use_cuda)
    
    # Create output directory if it doesn't exist
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if verbose:
        print(f"Synthesizing speech: '{text[:50]}{'...' if len(text) > 50 else ''}'")
    
    with wave.open(output_file, "wb") as wav_file:
        voice.synthesize_wav(text, wav_file)
    
    if verbose:
        print(f"Speech saved to: {output_file}")

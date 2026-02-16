#!/usr/bin/env python3
"""Main script for the speaking bot - Record audio and transcribe to text."""

import argparse
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import sounddevice as sd
import scipy.io.wavfile as wavfile

from speak_bot.record_audio import record_audio_until_silence
from speak_bot.speech_to_text import transcribe_audio
from speak_bot.mcp_client import LLMResponseGenerator
from speak_bot.text_to_speech import text_to_speech, text_to_speech_direct, text_to_speech_streaming



def get_package_root() -> Path:
    """Get the root directory of the speak_bot package."""
    return Path(__file__).parent.parent


def resolve_tts_model_path(model_path: str) -> str:
    """Resolve TTS model path to absolute path if it's a relative path.
    
    Args:
        model_path: Path to TTS model (relative or absolute)
        
    Returns:
        Absolute path to TTS model
    """
    path = Path(model_path)
    if path.is_absolute():
        return str(path)
    
    # If it's a relative path starting with "tts_models/", 
    # resolve it relative to the package root
    if model_path.startswith("tts_models/"):
        package_root = get_package_root()
        absolute_path = package_root / model_path
        if absolute_path.exists():
            return str(absolute_path)
    
    # If the file exists in the current directory, use it
    if path.exists():
        return str(path.absolute())
    
    # Otherwise, return the original path (let it fail with a clear error)
    return model_path


def play_audio(wav_file: str, verbose: bool = True):
    """Play audio from a WAV file.
    
    Args:
        wav_file: Path to the WAV file to play
        verbose: If True, print status messages
    """
    if verbose:
        print(f"Playing audio: {wav_file}")
    
    # Read the WAV file
    sample_rate, audio_data = wavfile.read(wav_file)
    
    # Play the audio and wait until it finishes
    sd.play(audio_data, sample_rate)
    sd.wait()
    
    if verbose:
        print("Audio playback complete.")


def play_audio_streaming(
    text: str, 
    model_path: str, 
    use_cuda: bool = False,
    use_cache: bool = True,
    verbose: bool = True
):
    """Convert text to speech and play with streaming synthesis and playback.
    
    This function starts playing audio immediately as synthesis progresses,
    providing much lower latency than batch processing.
    
    Args:
        text: The text to synthesize and play
        model_path: Path to the Piper TTS model file
        use_cuda: Whether to use GPU acceleration (not used on Apple Silicon)
        use_cache: Use cached voice model if available (recommended for performance)
        verbose: If True, print status messages
    """
    if verbose:
        print(f"Streaming TTS: '{text[:50]}{'...' if len(text) > 50 else ''}'")
    
    try:
        # Use streaming synthesis and playback
        text_to_speech_streaming(
            text=text,
            model_path=model_path,
            use_cuda=use_cuda,
            use_cache=use_cache,
            verbose=verbose
        )
            
    except Exception as e:
        if verbose:
            print(f"Streaming TTS failed, falling back to direct playback: {e}")
        
        # Fallback to direct (non-streaming) playback
        play_audio_direct(text, model_path, use_cuda, use_cache, verbose)


def play_audio_direct(
    text: str, 
    model_path: str, 
    use_cuda: bool = False,
    use_cache: bool = True,
    verbose: bool = True
):
    """Convert text to speech and play directly without saving to file.
    
    Args:
        text: The text to synthesize and play
        model_path: Path to the Piper TTS model file
        use_cuda: Whether to use GPU acceleration (not used on Apple Silicon)
        use_cache: Use cached voice model if available (recommended for performance)
        verbose: If True, print status messages
    """
    if verbose:
        print(f"Converting text to speech and playing: '{text[:50]}{'...' if len(text) > 50 else ''}'")
    
    try:
        # Generate audio data directly in memory
        audio_data, sample_rate = text_to_speech_direct(
            text=text,
            model_path=model_path,
            use_cuda=use_cuda,
            use_cache=use_cache,
            verbose=verbose
        )
        
        if len(audio_data) == 0:
            if verbose:
                print("No audio generated - skipping playback")
            return
        
        # Play the audio directly from memory
        if verbose:
            print("Playing audio...")
        sd.play(audio_data, sample_rate)
        sd.wait()
        
        if verbose:
            print("Audio playback complete.")
            
    except Exception as e:
        print(f"Error in direct TTS playback: {e}")
        raise


def run_voice_to_text(
    llm: LLMResponseGenerator | None = None,
    output_dir: str = "transcriptions",
    model_name: str = "tiny.en",
    samplerate: int = 16000,
    silence_seconds: float = 2.0,
    silence_threshold: int = 1500,
    timeout_seconds: float | None = None,
    language: str | None = "en",
    device: str = "cpu",
    compute_type: str = "int8",
    use_model_cache: bool = True,
    use_llm: bool = False,
    llm_model: str = "gpt-4o-mini",
    llm_stream: bool = False,
    history: list[dict[str, str]] | None = None,
    use_tts: bool = False,
    tts_model: str = "tts_models/jarvis-medium.onnx",
) -> tuple[str, str, bool, str, bool, str | None]:
    """Record audio from mic and transcribe to text, optionally process with LLM.
    
    Args:
        output_dir: Directory to save transcription files
        model_name: faster-whisper model size (tiny, tiny.en, base, small, medium, large)
        samplerate: Sample rate in Hz
        silence_seconds: Stop after this many seconds of silence
        silence_threshold: Amplitude threshold for silence
        timeout_seconds: Auto-terminate after this many seconds of no voice activity
        language: Language code for transcription (default: en)
        device: Device for Whisper model (cpu or cuda). Apple Silicon uses cpu with Core ML.
        compute_type: Compute precision for Whisper (int8, float16, float32)
        use_model_cache: Use cached models for better performance (recommended)
        use_llm: If True, process transcription with LLM
        llm_model: OpenAI model to use for LLM
        llm_stream: If True, stream the LLM response
        history: Previous conversation history for context
        use_tts: If True, convert LLM response to speech and play it
        tts_model: Path to TTS model file
    
    Returns:
        tuple: (transcribed_text, output_file_path, user_exit, assistant_response, timeout_exit, tool_signal)
    """
    # Step 1: Record audio from microphone
    print("=== Step 1: Recording Audio ===")
    audio_data, actual_samplerate, user_exit, timeout_exit = record_audio_until_silence(
        samplerate=samplerate,
        channels=1,
        silence_seconds=silence_seconds,
        silence_threshold=silence_threshold,
        timeout_seconds=timeout_seconds,
        verbose=True,
    )

    if user_exit:
        print("User requested exit during recording. Stopping loop.")
        return "", "", True, "", False, None
    if timeout_exit:
        print("Auto-termination: No voice activity detected.")
        return "", "", True, "", False, None
    
    if timeout_exit:
        print("Auto-termination: No voice activity detected.")
        return "", "", False, "", True, None
    
    if not audio_data.size:
        return "", "", False, "", False, None
    
    # Step 2: Transcribe audio to text
    print("\n=== Step 2: Transcribing Audio ===")
    text = transcribe_audio(
        audio_data,
        actual_samplerate,
        model_name=model_name,
        language=language,
        device=device,
        compute_type=compute_type,
        use_cache=use_model_cache,
        verbose=True,
    )
    
    # Step 3: Save transcription to file
    print("\n=== Step 3: Saving Transcription ===")
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    text_file = output_path / f"transcription_{timestamp}.txt"
    
    with open(text_file, "w", encoding="utf-8") as f:
        f.write(text)
    
    print(f"Saved transcription to: {text_file}")
    print(f"\nTranscribed text:\n{text}")
    
    # Step 4 (Optional): Process with LLM
    assistant_response = ""
    tool_signal = None
    
    if use_llm and llm:
        print("\n=== Step 4: Processing with LLM ===")
        try:
            if llm_stream:
                print("LLM Response (streaming):")
                assistant_response = ""
                # Note: Streaming doesn't support control signals yet
                response_content = llm.get_response(text, stream=True, history=history)
                if isinstance(response_content, str):
                    assistant_response = response_content
                    print(assistant_response)
                else:
                    for chunk in response_content:
                        if chunk:  # Check if chunk is not None
                            print(chunk, end="", flush=True)
                            assistant_response += chunk
                print()
            else:
                # Get response with control signal
                response_data = llm.get_response(text, stream=False, history=history, return_control_signal=True)
                if isinstance(response_data, tuple):
                    assistant_response, tool_signal = response_data
                else:
                    assistant_response = str(response_data)
                    tool_signal = None
                    
                print("LLM Response:")
                print(assistant_response)
        except Exception as e:
            print(f"Error processing with LLM: {e}")
    elif use_llm and not llm:
        print("Warning: LLM processing requested but no LLM instance provided")

    # Step 5 (Optional): Convert LLM response to speech and play it
    # Skip TTS if we're stopping (tool already played goodbye message)
    if use_tts and assistant_response and isinstance(assistant_response, str) and tool_signal != "stop_conversation":
        print("\n=== Step 5: Converting Response to Speech ===")
        try:
            # Generate and play speech with streaming synthesis and playback
            resolved_tts_model = resolve_tts_model_path(tts_model)
            play_audio_streaming(
                text=assistant_response,
                model_path=resolved_tts_model,
                use_cuda=False,  # Apple Silicon doesn't use CUDA
                use_cache=use_model_cache,
                verbose=True
            )
        except Exception as e:
            print(f"Error with TTS: {e}")

    return text, str(text_file), False, str(assistant_response), False, tool_signal


def parse_args():
    parser = argparse.ArgumentParser(
        description="Speaking Bot: Record audio from mic and transcribe to text"
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        default="transcriptions",
        help="Directory to save transcription files (default: transcriptions)",
    )
    parser.add_argument(
        "-m",
        "--model",
        default="tiny.en",
        choices=["tiny", "tiny.en", "base", "base.en", "small", "small.en", "medium", "medium.en", "large"],
        help="faster-whisper model size (default: tiny.en for faster English-only)",
    )
    parser.add_argument(
        "-r",
        "--samplerate",
        type=int,
        default=16000,
        help="Sample rate in Hz (default: 16000)",
    )
    parser.add_argument(
        "--silence-seconds",
        type=float,
        default=2.0,
        help="Stop after this many seconds of silence (default: 2.0)",
    )
    parser.add_argument(
        "--silence-threshold",
        type=int,
        default=1500,
        help="Amplitude threshold for silence (default: 1500)",
    )
    parser.add_argument(
        "-l",
        "--language",
        default="en",
        help="Language code (e.g., 'en', 'es'). Default: en",
    )
    parser.add_argument(
        "--use-llm",
        action="store_true",
        help="Process transcription with LLM to generate response",
    )
    parser.add_argument(
        "--llm-model",
        default="gpt-4o-mini",
        help="OpenAI model to use (default: gpt-4o-mini)",
    )
    parser.add_argument(
        "--llm-stream",
        action="store_true",
        help="Stream the LLM response",
    )
    parser.add_argument(
        "--use-tts",
        action="store_true",
        help="Convert LLM response to speech and play it",
    )
    parser.add_argument(
        "--tts-model",
        default="tts_models/jarvis-medium.onnx",
        help="Path to TTS ONNX model (default: tts_models/jarvis-medium.onnx)",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=None,
        help="Auto-terminate after this many seconds of no voice activity (default: disabled)",
    )
    parser.add_argument(
        "--loop",
        action="store_true",
        help="Keep recording in a conversation loop until 'q' + Enter is pressed",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device for Whisper model (default: cpu, uses Core ML on Apple Silicon)",
    )
    parser.add_argument(
        "--compute-type",
        default="int8",
        choices=["int8", "float16", "float32"],
        help="Compute type for Whisper (default: int8, good for Apple Silicon)",
    )
    parser.add_argument(
        "--no-model-cache",
        action="store_true",
        help="Disable model caching (default: caching enabled for better performance)",
    )
    return parser.parse_args()


def run_conversation(args):
    print("Conversation loop started. Press 'q' then Enter during recording to exit.")
    if args.timeout_seconds:
        print(f"Auto-termination enabled: Will exit after {args.timeout_seconds}s of no voice activity.")
    
    # Pre-load models for better performance (only if caching is enabled)
    if not args.no_model_cache:
        print("\n=== Pre-loading Models for Better Performance ===")
        print("Loading Whisper model (this may take a few seconds)...")
        from speak_bot.speech_to_text import WhisperModelCache
        WhisperModelCache.get_model(
            model_name=args.model,
            device=args.device,
            compute_type=args.compute_type
        )
        print("✓ Whisper model loaded and cached")
        
        if args.use_tts:
            print("Loading TTS voice model...")
            from speak_bot.text_to_speech import PiperVoiceCache
            resolved_tts = resolve_tts_model_path(args.tts_model)
            PiperVoiceCache.get_voice(model_path=resolved_tts, use_cuda=False)
            print("✓ TTS voice loaded and cached")
        
        print("=== Models Ready - Starting Conversation ===\n")
    
    history: list[dict[str, str]] = []

    # Initialize LLM with MCP server connection
    llm = LLMResponseGenerator(model=args.llm_model)
    
    # Auto-start MCP server connection
    mcp_connected = False
    try:
        from pathlib import Path
        import logging
        
        # Set up logging to see MCP connection details
        logging.basicConfig(level=logging.INFO)
        
        mcp_server_path = Path(__file__).parent / "mcp" / "mcp_server.py"
        
        print(f"🔌 Connecting to MCP server at: {mcp_server_path}")
        llm.connect_to_server(str(mcp_server_path))
        mcp_connected = True
        print("✅ MCP server connected for conversation control.")
        print("   You can now say 'goodbye' or 'stop' to end the conversation.")
    except Exception as e:
        print(f"⚠️  MCP server connection failed: {e}")
        print("   Conversation will continue without voice stop functionality.")
        print("   You can still press 'q' + Enter during recording to exit.")
        # Don't exit, continue with fallback mode
    
    while True:
        text, _, user_exit, assistant_response, timeout_exit, tool_signal = run_voice_to_text(
            llm=llm,
            output_dir=args.output_dir,
            model_name=args.model,
            samplerate=args.samplerate,
            silence_seconds=args.silence_seconds,
            silence_threshold=args.silence_threshold,
            timeout_seconds=args.timeout_seconds,
            language=args.language,
            device=args.device,
            compute_type=args.compute_type,
            use_model_cache=not args.no_model_cache,
            use_llm=args.use_llm,
            llm_model=args.llm_model,
            llm_stream=args.llm_stream,
            history=history,
            use_tts=args.use_tts,
            tts_model=args.tts_model,
        )

        # Check all exit conditions
        if user_exit:
            print("Exiting conversation loop.")
            sys.exit(0)
        
        if timeout_exit:
            print("Auto-termination activated. Exiting conversation loop.")
            sys.exit(0)
            
        # NEW: Check for tool-initiated stop
        if tool_signal == "stop_conversation":
            print("Conversation ended by assistant.")
            sys.exit(0)

        # Update conversation history
        if args.use_llm and text:
            history.append({"role": "user", "content": text})
            if assistant_response:
                history.append({"role": "assistant", "content": assistant_response})


def main():
    args = parse_args()
    if args.loop:
        run_conversation(args)
    else:
        run_voice_to_text(
            output_dir=args.output_dir,
            model_name=args.model,
            samplerate=args.samplerate,
            silence_seconds=args.silence_seconds,
            silence_threshold=args.silence_threshold,
            timeout_seconds=args.timeout_seconds,
            language=args.language,
            device=args.device,
            compute_type=args.compute_type,
            use_model_cache=not args.no_model_cache,
            use_llm=args.use_llm,
            llm_model=args.llm_model,
            llm_stream=args.llm_stream,
            use_tts=args.use_tts,
            tts_model=args.tts_model,
        )


if __name__ == "__main__":
    main()

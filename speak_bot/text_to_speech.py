import wave
from pathlib import Path
import queue
import threading
import time

import numpy as np
import sounddevice as sd
from piper import PiperVoice


class PiperVoiceCache:
    """Singleton cache for Piper TTS voices to avoid reloading.
    
    This significantly improves performance by keeping voice models in memory
    across multiple synthesis requests. First load takes normal time,
    subsequent loads are instant.
    """
    _voices: dict[str, PiperVoice] = {}
    
    @classmethod
    def get_voice(
        cls,
        model_path: str = "tts_models/jarvis-medium.onnx",
        use_cuda: bool = False  # Apple Silicon doesn't use CUDA
    ) -> PiperVoice:
        """Get a cached Piper voice or create a new one.
        
        Args:
            model_path: Path to the Piper TTS model file
            use_cuda: Whether to use CUDA acceleration (not used on Apple Silicon)
            
        Returns:
            Cached or newly created PiperVoice instance
        """
        cache_key = f"{model_path}_{use_cuda}"
        if cache_key not in cls._voices:
            print(f"Loading Piper voice: {model_path}")
            cls._voices[cache_key] = PiperVoice.load(model_path, use_cuda=use_cuda)
            print("✓ Piper voice loaded and cached")
        return cls._voices[cache_key]
    
    @classmethod
    def clear_cache(cls):
        """Clear all cached voices to free memory."""
        cls._voices.clear()


def text_to_speech(
    text: str,
    output_file: str,
    model_path: str = "tts_models/jarvis-medium.onnx",
    use_cuda: bool = False,
    use_cache: bool = True,
    verbose: bool = True,
) -> None:
    """Convert text to speech and save as a WAV file using Piper TTS.
    
    Args:
        text: The text to synthesize
        output_file: Path where the WAV file will be saved
        model_path: Path to the Piper TTS model file
        use_cuda: Whether to use GPU acceleration (not used on Apple Silicon)
        use_cache: Use cached voice model if available (recommended for performance)
        verbose: If True, print status messages
    """
    if verbose:
        print(f"Loading Piper TTS model from: {model_path}")
    
    # Use cached voice if enabled (recommended for performance)
    if use_cache:
        voice = PiperVoiceCache.get_voice(model_path, use_cuda)
    else:
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


def text_to_speech_streaming(
    text: str,
    model_path: str = "tts_models/jarvis-medium.onnx",
    use_cuda: bool = False,
    use_cache: bool = True,
    verbose: bool = True,
) -> None:
    """Convert text to speech with streaming synthesis and playback using Piper TTS.
    
    This function starts playing audio immediately as synthesis progresses, rather than
    waiting for complete synthesis before playback begins. This significantly reduces
    perceived latency for longer texts.
    
    Args:
        text: The text to synthesize
        model_path: Path to the Piper TTS model file
        use_cuda: Whether to use GPU acceleration (not used on Apple Silicon)
        use_cache: Use cached voice model if available (recommended for performance)
        verbose: If True, print status messages
    """
    if verbose:
        print(f"Loading Piper TTS model from: {model_path}")
    
    # Use cached voice if enabled (recommended for performance)
    if use_cache:
        voice = PiperVoiceCache.get_voice(model_path, use_cuda)
    else:
        voice = PiperVoice.load(model_path, use_cuda=use_cuda)
    
    if verbose:
        print(f"Streaming synthesis: '{text[:50]}{'...' if len(text) > 50 else ''}'")
    
    # Audio streaming setup
    audio_queue = queue.Queue(maxsize=3)  # Small buffer to prevent memory buildup
    synthesis_complete = threading.Event()
    synthesis_error = threading.Event()
    error_message = None
    
    # State for audio callback
    current_chunk = None
    current_position = 0
    sample_rate = 22050  # Default, will be updated from first chunk
    
    def synthesis_worker():
        """Producer thread: generates audio chunks and puts them in the queue."""
        nonlocal error_message, sample_rate
        try:
            chunk_count = 0
            for chunk in voice.synthesize(text):
                if chunk_count == 0:
                    sample_rate = chunk.sample_rate
                    if verbose:
                        print(f"Starting streaming playback at {sample_rate} Hz...")
                
                audio_queue.put(chunk)
                chunk_count += 1
                
                if verbose:
                    chunk_duration = len(chunk.audio_float_array) / chunk.sample_rate
                    print(f"Generated chunk {chunk_count} ({chunk_duration:.2f}s)")
            
            if verbose:
                print(f"Synthesis complete. Generated {chunk_count} chunks.")
                
        except Exception as e:
            error_message = str(e)
            synthesis_error.set()
            if verbose:
                print(f"Synthesis error: {e}")
        finally:
            synthesis_complete.set()
    
    def audio_callback(outdata, frames, time_info, status):
        """Consumer callback: reads chunks from queue and outputs to audio device."""
        nonlocal current_chunk, current_position
        
        if status:
            if verbose:
                print(f"Audio callback status: {status}")
        
        # Initialize output with silence
        outdata.fill(0)
        
        # Check for synthesis errors
        if synthesis_error.is_set():
            return
        
        frames_written = 0
        
        while frames_written < frames:
            # Get new chunk if current one is exhausted
            if current_chunk is None:
                try:
                    current_chunk = audio_queue.get_nowait()
                    current_position = 0
                except queue.Empty:
                    # No more chunks available
                    if synthesis_complete.is_set():
                        # Synthesis is done and queue is empty - we're finished
                        break
                    else:
                        # Still synthesizing, just wait for more chunks
                        break
            
            # Write audio data from current chunk
            if current_chunk is not None:
                chunk_data = current_chunk.audio_float_array
                remaining_in_chunk = len(chunk_data) - current_position
                remaining_in_buffer = frames - frames_written
                
                if remaining_in_chunk > 0:
                    samples_to_copy = min(remaining_in_buffer, remaining_in_chunk)
                    
                    # Copy audio data to output buffer
                    start_pos = frames_written
                    end_pos = frames_written + samples_to_copy
                    chunk_start = current_position
                    chunk_end = current_position + samples_to_copy
                    
                    outdata[start_pos:end_pos, 0] = chunk_data[chunk_start:chunk_end]
                    
                    frames_written += samples_to_copy
                    current_position += samples_to_copy
                    
                    # Check if we've finished this chunk
                    if current_position >= len(chunk_data):
                        current_chunk = None
                        current_position = 0
                else:
                    # Chunk is exhausted
                    current_chunk = None
                    current_position = 0
    
    # Start synthesis in background thread
    synth_thread = threading.Thread(target=synthesis_worker, daemon=True)
    synth_thread.start()
    
    # Start streaming audio playback
    try:
        with sd.OutputStream(
            callback=audio_callback,
            samplerate=sample_rate,
            channels=1,
            blocksize=1024,  # Small block size for low latency
            dtype=np.float32
        ):
            if verbose:
                print("Streaming playback started...")
            
            # Wait for synthesis thread to complete
            synth_thread.join()
            
            # Check for synthesis errors
            if synthesis_error.is_set():
                raise Exception(f"Synthesis failed: {error_message}")
            
            # Continue playback until all chunks are consumed
            while current_chunk is not None or not audio_queue.empty():
                time.sleep(0.1)
            
            # Small delay to ensure last chunk finishes playing
            time.sleep(0.5)
            
        if verbose:
            print("Streaming playback complete.")
            
    except Exception as e:
        if verbose:
            print(f"Error during streaming playback: {e}")
        raise


def text_to_speech_direct(
    text: str,
    model_path: str = "tts_models/jarvis-medium.onnx",
    use_cuda: bool = False,
    use_cache: bool = True,
    verbose: bool = True,
) -> tuple[np.ndarray, int]:
    """Convert text to speech and return audio data in memory using Piper TTS.
    
    Args:
        text: The text to synthesize
        model_path: Path to the Piper TTS model file
        use_cuda: Whether to use GPU acceleration (not used on Apple Silicon)
        use_cache: Use cached voice model if available (recommended for performance)
        verbose: If True, print status messages
    
    Returns:
        tuple: (audio_data as float32 numpy array, sample_rate)
    """
    if verbose:
        print(f"Loading Piper TTS model from: {model_path}")
    
    # Use cached voice if enabled (recommended for performance)
    if use_cache:
        voice = PiperVoiceCache.get_voice(model_path, use_cuda)
    else:
        voice = PiperVoice.load(model_path, use_cuda=use_cuda)
    
    if verbose:
        print(f"Synthesizing speech: '{text[:50]}{'...' if len(text) > 50 else ''}'")
    
    # Synthesize text and get audio chunks
    chunks = list(voice.synthesize(text))
    
    if not chunks:
        if verbose:
            print("No audio chunks generated")
        return np.array([], dtype=np.float32), 22050
    
    # Combine all chunks into a single audio array
    audio_data = np.concatenate([chunk.audio_float_array for chunk in chunks])
    sample_rate = chunks[0].sample_rate
    
    if verbose:
        print(f"Generated {len(audio_data)} audio samples at {sample_rate} Hz")
    
    return audio_data, sample_rate

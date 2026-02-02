import argparse
import queue
import sys
import threading
import wave

import numpy as np
import sounddevice as sd


def record_audio_until_silence(
    samplerate: int = 16000,
    channels: int = 1,
    blocksize: int = 2048,
    silence_seconds: float = 2.0,
    silence_threshold: int = 1500,
    verbose: bool = True,
) -> tuple[np.ndarray, int, bool]:
    """Record audio from microphone until silence is detected.
    
    Returns:
        tuple: (audio_data as int16 numpy array, samplerate, user_exit)
    """
    q: queue.Queue[np.ndarray] = queue.Queue()
    chunks = []

    stop_event = threading.Event()
    user_exit = False

    def callback(indata, frames, time_info, status):
        if status and verbose:
            print(status, file=sys.stderr)
        q.put(indata.copy())

    silent_target_frames = int(silence_seconds * samplerate)
    silent_frames = 0
    activity_started = False

    def exit_listener():
        nonlocal user_exit
        for line in sys.stdin:
            if stop_event.is_set():
                break
            if line.strip().lower() == "q":
                user_exit = True
                stop_event.set()
                break

    listener_thread = threading.Thread(target=exit_listener, daemon=True)
    listener_thread.start()

    if verbose:
        print(
            f"Recording... stop after {silence_seconds}s of silence (threshold={silence_threshold})."
        )
        print("Press 'q' then Enter to stop recording and exit.")

    try:
        with sd.InputStream(
            samplerate=samplerate,
            channels=channels,
            dtype="int16",
            blocksize=blocksize,
            callback=callback,
        ):
            while True:
                if stop_event.is_set():
                    break

                try:
                    chunk = q.get(timeout=0.1)
                except queue.Empty:
                    continue
                chunks.append(chunk)

                amp = int(np.max(np.abs(chunk)))
                if amp > silence_threshold:
                    activity_started = True
                    silent_frames = 0
                else:
                    if activity_started:
                        silent_frames += chunk.shape[0]
                        if silent_frames >= silent_target_frames:
                            break
    except KeyboardInterrupt:
        if verbose:
            print("\nStopped by user.")

    stop_event.set()
    listener_thread.join(timeout=0.1)

    if chunks:
        audio_data = np.concatenate(chunks, axis=0)
    else:
        audio_data = np.zeros((0, channels), dtype=np.int16)
    if verbose:
        print(f"Recording complete: {len(audio_data)} frames")
    return audio_data, samplerate, user_exit


def save_audio_to_wav(audio_data: np.ndarray, samplerate: int, output_path: str, channels: int = 1):
    """Save audio data to a WAV file."""
    with wave.open(output_path, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(samplerate)
        wf.writeframes(audio_data.tobytes())
    print(f"Saved recording to: {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Record mic audio to WAV until silence.")
    parser.add_argument(
        "-o",
        "--output",
        default="recording.wav",
        help="Output WAV file path (default: recording.wav)",
    )
    parser.add_argument(
        "-r",
        "--samplerate",
        type=int,
        default=16000,
        help="Sample rate in Hz (default: 16000)",
    )
    parser.add_argument(
        "-c",
        "--channels",
        type=int,
        default=1,
        help="Number of channels (1=mono, 2=stereo). Default: 1",
    )
    parser.add_argument(
        "--silence-seconds",
        type=float,
        default=2.0,
        help="Stop after this many seconds of continuous silence (default: 2.0)",
    )
    parser.add_argument(
        "--silence-threshold",
        type=int,
        default=1500,
        help="Amplitude threshold for silence, int16 scale (default: 1500)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    audio_data, samplerate, user_exit = record_audio_until_silence(
        samplerate=args.samplerate,
        channels=args.channels,
        silence_seconds=args.silence_seconds,
        silence_threshold=args.silence_threshold,
    )
    if user_exit:
        print("Recording stopped by user.")
        return
    save_audio_to_wav(audio_data, samplerate, args.output, args.channels)


if __name__ == "__main__":
    main()

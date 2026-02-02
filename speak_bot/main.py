#!/usr/bin/env python3
"""Main script for the speaking bot - Record audio and transcribe to text."""

import argparse
import sys
from datetime import datetime
from pathlib import Path

import sounddevice as sd
import scipy.io.wavfile as wavfile

from speak_bot.record_audio import record_audio_until_silence
from speak_bot.speech_to_text import transcribe_audio
from speak_bot.llm_response import LLMResponseGenerator
from speak_bot.text_to_speech import text_to_speech


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


def run_voice_to_text(
    output_dir: str = "transcriptions",
    model_name: str = "base",
    samplerate: int = 16000,
    silence_seconds: float = 2.0,
    silence_threshold: int = 1500,
    language: str | None = None,
    use_llm: bool = False,
    llm_model: str = "gpt-4o-mini",
    llm_stream: bool = False,
    history: list[dict[str, str]] | None = None,
    use_tts: bool = False,
    tts_model: str = "tts_models/en_US-amy-low.onnx",
) -> tuple[str, str, bool, str]:
    """Record audio from mic and transcribe to text, optionally process with LLM.
    
    Args:
        output_dir: Directory to save transcription files
        model_name: Whisper model size (tiny, base, small, medium, large)
        samplerate: Sample rate in Hz
        silence_seconds: Stop after this many seconds of silence
        silence_threshold: Amplitude threshold for silence
        language: Language code for transcription
        use_llm: If True, process transcription with LLM
        llm_model: OpenAI model to use for LLM
        llm_stream: If True, stream the LLM response
        use_tts: If True, convert LLM response to speech and play it
        tts_model: Path to the TTS ONNX model
    
    Returns:
        tuple: (transcribed_text, output_file_path, user_exit, assistant_response)
    """
    # Step 1: Record audio from microphone
    print("=== Step 1: Recording Audio ===")
    audio_data, actual_samplerate, user_exit = record_audio_until_silence(
        samplerate=samplerate,
        channels=1,
        silence_seconds=silence_seconds,
        silence_threshold=silence_threshold,
        verbose=True,
    )

    if user_exit:
        print("User requested exit during recording. Stopping loop.")
        return "", "", True, ""
    if audio_data.size == 0:
        print("No audio captured. Will retry.")
        return "", "", False, ""
    
    # Step 2: Transcribe audio to text
    print("\n=== Step 2: Transcribing Audio ===")
    text = transcribe_audio(
        audio_data,
        actual_samplerate,
        model_name=model_name,
        language=language,
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
    if use_llm:
        print("\n=== Step 4: Processing with LLM ===")
        try:
            llm = LLMResponseGenerator(model=llm_model)
            if llm_stream:
                print("LLM Response (streaming):")
                for chunk in llm.get_response(text, stream=True, history=history):
                    print(chunk, end="", flush=True)
                    assistant_response += chunk
                print()
            else:
                assistant_response = llm.get_response(
                    text, stream=False, history=history
                )
                print("LLM Response:")
                print(assistant_response)
        except Exception as e:
            print(f"Error processing with LLM: {e}")
        
        # Step 5 (Optional): Convert LLM response to speech and play it
        if use_tts and assistant_response:
            print("\n=== Step 5: Converting Response to Speech ===")
            try:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                tts_output = f"output/tts_response_{timestamp}.wav"
                
                # Generate speech from LLM response
                text_to_speech(
                    text=assistant_response,
                    output_file=tts_output,
                    model_path=tts_model,
                    verbose=True
                )
                
                # Play the audio
                play_audio(tts_output, verbose=True)
            except Exception as e:
                print(f"Error with TTS: {e}")

    return text, str(text_file), False, assistant_response


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
        default="base",
        choices=["tiny", "base", "small", "medium", "large"],
        help="Whisper model size (default: base)",
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
        default=None,
        help="Language code (e.g., 'en', 'es'). Auto-detect if omitted.",
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
        default="tts_models/en_US-amy-low.onnx",
        help="Path to TTS ONNX model (default: tts_models/en_US-amy-low.onnx)",
    )
    parser.add_argument(
        "--loop",
        action="store_true",
        help="Keep recording in a conversation loop until 'q' + Enter is pressed",
    )
    return parser.parse_args()


def run_conversation(args):
    print("Conversation loop started. Press 'q' then Enter during recording to exit.")
    history: list[dict[str, str]] = []
    while True:
        text, _, user_exit, assistant_response = run_voice_to_text(
            output_dir=args.output_dir,
            model_name=args.model,
            samplerate=args.samplerate,
            silence_seconds=args.silence_seconds,
            silence_threshold=args.silence_threshold,
            language=args.language,
            use_llm=args.use_llm,
            llm_model=args.llm_model,
            llm_stream=args.llm_stream,
            history=history,
            use_tts=args.use_tts,
            tts_model=args.tts_model,
        )

        if user_exit:
            print("Exiting conversation loop.")
            sys.exit(0)

        if args.use_llm and text:
            # Update conversation history
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
            language=args.language,
            use_llm=args.use_llm,
            llm_model=args.llm_model,
            llm_stream=args.llm_stream,
            use_tts=args.use_tts,
            tts_model=args.tts_model,
        )


if __name__ == "__main__":
    main()

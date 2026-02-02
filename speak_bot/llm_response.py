#!/usr/bin/env python3
"""LLM response generator - Process transcribed text through OpenAI API."""

import os
from pathlib import Path
from typing import Generator, List, Dict

from dotenv import load_dotenv
from openai import OpenAI, APIError


# Load environment variables from .env file
load_dotenv()


class LLMResponseGenerator:
    """Generate conversational responses using OpenAI API."""

    def __init__(self, api_key: str | None = None, model: str = "gpt-4o-mini"):
        """Initialize the LLM response generator.
        
        Args:
            api_key: OpenAI API key. If None, will load from OPENAI_API_KEY env var.
            model: Model to use (default: gpt-4o-mini for faster, cheaper responses)
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenAI API key not found. Please set OPENAI_API_KEY in .env file "
                "or pass it directly."
            )
        self.client = OpenAI(api_key=self.api_key)
        self.model = model
        self.system_prompt = (
            "You are a helpful, friendly assistant. Keep your responses short, "
            "concise, and conversational (1-3 sentences max). Avoid long explanations "
            "or verbose text. Be natural and friendly."
        )

    def get_response(
        self,
        text: str,
        stream: bool = False,
        temperature: float = 0.7,
        history: List[Dict[str, str]] | None = None,
    ) -> str | Generator[str, None, None]:
        """Get a response from the LLM for the given text.
        
        Args:
            text: The transcribed text to send to the LLM
            stream: If True, returns a generator for streaming response.
                    If False, returns the complete response as a string.
            temperature: Controls randomness (0.0-2.0). Lower = more deterministic.
        
        Returns:
            str if stream=False: Complete response text
            Generator[str] if stream=True: Generator that yields response chunks
            
        Raises:
            APIError: If there's an issue with the OpenAI API call
        """
        if not text or not text.strip():
            raise ValueError("Input text cannot be empty")

        try:
            messages: List[Dict[str, str]] = [{"role": "system", "content": self.system_prompt}]
            if history:
                messages.extend(history)
            messages.append({"role": "user", "content": text})

            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                stream=stream,
            )

            if stream:
                return self._stream_response(response)
            else:
                return response.choices[0].message.content

        except APIError as e:
            raise APIError(f"Error communicating with OpenAI API: {e}")

    @staticmethod
    def _stream_response(response) -> Generator[str, None, None]:
        """Process streaming response and yield chunks.
        
        Args:
            response: Streaming response from OpenAI API
            
        Yields:
            Text chunks from the response
        """
        for chunk in response:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    def process_transcription_file(
        self,
        file_path: str,
        stream: bool = False,
        temperature: float = 0.7,
        history: List[Dict[str, str]] | None = None,
    ) -> str:
        """Process a transcription file and return LLM response.
        
        Args:
            file_path: Path to the transcription text file
            stream: If True, streams and prints the response.
                    If False, returns complete response.
            temperature: Controls randomness of response
            
        Returns:
            The LLM response (for non-streaming) or full accumulated text (for streaming)
        """
        # Read transcription file
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Transcription file not found: {file_path}")

        with open(file_path, "r", encoding="utf-8") as f:
            transcribed_text = f.read()

        if not transcribed_text.strip():
            raise ValueError("Transcription file is empty")

        # Get response from LLM
        if stream:
            print("LLM Response (streaming):")
            accumulated_text = ""
            for chunk in self.get_response(
                transcribed_text, stream=True, temperature=temperature, history=history
            ):
                print(chunk, end="", flush=True)
                accumulated_text += chunk
            print()  # Newline after streaming
            return accumulated_text
        else:
            response = self.get_response(
                transcribed_text, stream=False, temperature=temperature, history=history
            )
            print("LLM Response:")
            print(response)
            return response


def main():
    """Example usage of LLM response generator."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Process transcribed text with LLM"
    )
    parser.add_argument(
        "text",
        nargs="?",
        help="Text to send to LLM (if not provided, reads from latest transcription file)",
    )
    parser.add_argument(
        "-f",
        "--file",
        help="Path to transcription file to process",
    )
    parser.add_argument(
        "-s",
        "--stream",
        action="store_true",
        help="Stream the response instead of waiting for complete response",
    )
    parser.add_argument(
        "-t",
        "--temperature",
        type=float,
        default=0.7,
        help="Temperature for response generation (0.0-2.0, default: 0.7)",
    )
    parser.add_argument(
        "-m",
        "--model",
        default="gpt-4o-mini",
        help="OpenAI model to use (default: gpt-4o-mini)",
    )

    args = parser.parse_args()

    try:
        llm = LLMResponseGenerator(model=args.model)

        if args.text:
            # Direct text input
            if args.stream:
                print("LLM Response (streaming):")
                for chunk in llm.get_response(args.text, stream=True, temperature=args.temperature):
                    print(chunk, end="", flush=True)
                print()
            else:
                response = llm.get_response(
                    args.text, stream=False, temperature=args.temperature
                )
                print("LLM Response:")
                print(response)

        elif args.file:
            # Process file
            llm.process_transcription_file(
                args.file, stream=args.stream, temperature=args.temperature
            )

        else:
            # Find latest transcription file
            transcription_dir = Path("transcriptions")
            if not transcription_dir.exists():
                print("Error: No transcriptions directory found")
                return

            files = list(transcription_dir.glob("transcription_*.txt"))
            if not files:
                print("Error: No transcription files found")
                return

            latest_file = max(files, key=lambda p: p.stat().st_mtime)
            print(f"Processing latest transcription: {latest_file.name}")
            llm.process_transcription_file(
                latest_file, stream=args.stream, temperature=args.temperature
            )

    except (ValueError, FileNotFoundError) as e:
        print(f"Error: {e}")
    except APIError as e:
        print(f"API Error: {e}")


if __name__ == "__main__":
    main()

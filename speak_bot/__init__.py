"""speak_bot - A voice-first assistant pipeline."""

__version__ = "0.1.0"
__author__ = "Your Name"
__description__ = "A voice-first assistant pipeline."

# Make main functionality available at package level
from .main import main, run_voice_to_text

__all__ = ["main", "run_voice_to_text"]
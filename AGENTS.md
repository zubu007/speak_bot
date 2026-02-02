# Agent Instructions

This document collects the hard and soft rules that every agent working in this repository should know before making changes. Use it as both a quickstart and a reminder of the existing conventions.

## Purpose

- The repo builds a voice-first assistant pipeline combining mic capture, Whisper transcription, optional LLM enrichment, and Piper-powered speech synthesis.
- Existing scripts live as lightweight CLI entry points; agents should treat them as sequential stages with clearly defined inputs/outputs.
- Because the user base is CLI, every change should favor reliability, meaningful console feedback, and audio artifacts stored under `transcriptions/` or `output/`.

## Repository Layout

- `main.py` orchestrates the end-to-end loop, exposing arguments for every recording/transcription/LLM/TTS knob.
- `record_audio.py` handles silence-based recording and exposes reusable helpers such as `record_audio_until_silence` + `save_audio_to_wav`.
- `speech_to_text.py` and `text_to_speech.py` wrap Whisper and Piper respectively for batch/CLI usage.
- Model weights and sample outputs live in `tts_models/` and `output/`; transcriptions are written to `transcriptions/` with timestamped filenames.

## Cursor & Copilot Rules

- Cursor rules (`.cursor/rules/` or `.cursorrules`) were not present when this file was written.
- Copilot instructions at `.github/copilot-instructions.md` were not found either.

## Build / Install / Lint / Test Commands

### Environment Bootstrapping

- `pip install .` – installs from the current `pyproject.toml`. This is the canonical setup for any agent in this repo.
- If virtual environments are preferred, wrap the command with `python -m venv .venv` + `source .venv/bin/activate` before installation.

### Common Commands (full pipeline)

- `python main.py --loop --use-llm` – keeps recording, transcribing, calling the LLM, and optionally speaking the assistant output until `q` is typed during recording. Use this to verify the interaction flow when working on end-to-end features.
- `python main.py --model small --use-llm --use-tts --llm-stream` – helpful when testing streaming response handling and TTS playback.

### Standalone Module Commands

- `python record_audio.py --output recording.wav` – records voice until silence and writes a WAV, useful for validating recording options independently.
- `python speech_to_text.py test.wav --model base` – transcribes a WAV file and either prints the text or saves it via `-o`. This is the reference command for working on transcription logic and serves as the “single test” run.
- `python text_to_speech.py "Hello from Piper"` – exercises the Piper TTS wrapper.

### Testing Guidance

- There is no automated test suite yet. The manual smoke command above (`speech_to_text.py test.wav`) counts as the single-test run every agent should know when verifying changes.
- When you add automated tests, place them under a new `tests/` directory and document new commands in this section.

## Coding Style Guidelines

### Imports

- Order imports as: (1) standard library, (2) third-party modules, (3) local project modules. Separate sections with a blank line.
- Prefer `from module import name` only for names used directly; always keep imports deterministic and sorted alphabetically within each block.

### Formatting

- Use 4 spaces for indentation; tabs are not allowed.
- Keep line lengths under ~100 characters when reasonable; wrap arguments or string concatenations using implicit parentheses or multi-line f-strings.
- Always leave a trailing newline at the end of files.
- Prefer multiline docstrings (triple double quotes) for functions that do non-trivial work; include Args/Returns sections when helpful.

### Typing

- Use type hints everywhere the existing codebase does. Favor Python 3.10 union syntax (`str | None`) and concrete collections (e.g., `list[str]`).
- When returning tuples, document tuple contents clearly in docstrings (see `run_voice_to_text`).

### Naming Conventions

- Functions and variables are snake_case; classes (e.g., `LLMResponseGenerator` in `llm_response.py`) should be PascalCase.
- CLI option names mirror parameter names, so keep function signatures aligned with argparse argument names and default values.
- Constants (e.g., `DEFAULT_MODEL = "base"`) should be uppercase with underscores.

### CLI / Argument Handling

- Use `argparse.ArgumentParser` with explicit `help` text for every flag.
- Keep default values in one place (argument definitions or top-level constants) so CLI behavior is predictable.
- Do not mutate parser defaults at runtime; instead pass configuration into helper functions.

### Logging, Prints, and Verbosity

- The repo currently uses `print` for status updates; keep this style unless you introduce a logging subsystem.
- Verbose flags should gate extra output (e.g., `verbose=True` in `transcribe_audio`). When adding new modules, mirror this behaviour with optional prints.

### Docstrings & Comments

- Every exported function (those imported elsewhere or used as CLI entry points) should include a docstring describing intent, parameters, and return values.
- Keep inline comments minimal and only when the logic is non-obvious (e.g., silence detection thresholds, resampling steps).

### Error Handling

- Wrap calls to external resources (LLM, TTS engine, microphone) with `try/except` and emit clear user-facing messages.
- Avoid swallowing exceptions silently — always log the exception message even if you continue execution.
- Prefer returning sentinel values (`""` for empty text) over raising exceptions during routine operations, especially in loops (see `run_voice_to_text`).

### Concurrency & Threads

- Threads (e.g., the `exit_listener` in `record_audio`) should be daemonized when they monitor user input; ensure events are set before joining.
- Always guard shared flags (like `user_exit`) with appropriate synchronization or thread-safe patterns (events or queue). Current pattern passes via `nonlocal` in the listener.

### Module Composition

- Keep CLI helpers thin: gather args, call pure functions, and exit. Business logic belongs in helper functions that can be imported without side effects.
- Utility helpers (recording, transcription, TTS) should avoid global state. Pass everything via parameters.

### File Structure and Outputs

- New audio or transcription outputs go under `output/` or `transcriptions/` with timestamped filenames to prevent collisions.
- If you edit model-related files, mention where training weights belong but keep huge binaries out of Git; use `tts_models/` for ONNX files and avoid tracking new ones unless explicitly requested.

### Dependencies

- `pyproject.toml` is authoritative. Add new packages there with version constraints. After adding, rerun `pip install .` to regenerate the lock file (if using one).
- Keep the dependency list minimal; prefer standard libs plus Whisper, Piper, SoundDevice, NumPy/SciPy, OpenAI, and python-dotenv.

## Workflow Suggestions

- Always run `python main.py --loop --use-llm` after touching the control loop to make sure streaming/exit handling behaves.
- When working on transcription or TTS, keep a sample `test.wav` handy in the repo root so `speech_to_text.py test.wav` remains fast.
- Document new CLI flags or configuration changes in `README.md` and mention them here for future agents.

## Output & Artifacts

- Store generated transcriptions under `transcriptions/` and name them using `transcription_<timestamp>.txt` (see `run_voice_to_text`).
- Keep synthesized speech files under `output/` with descriptive timestamps, and never commit large recordings or generated audio unless explicitly requested.
- If you need to introduce curated sample outputs for regression checks, keep them small (a few seconds) and describe their provenance in the accompanying doc blurb.
- Clean up temporary WAVs created during manual verification to avoid cluttering `output/` between runs.

## Manual Verification Checklist

- Record a short clip using `python record_audio.py --output recording.wav` to verify microphone, silence thresholds, and file writing behavior.
- Transcribe the clip with `python speech_to_text.py recording.wav` to ensure Whisper integration and language defaults behave as expected.
- Optionally run `python text_to_speech.py "Testing TTS" -o output/test_tts.wav` to ensure Piper still saves good audio and playback works without crashes.
- When changing the loop logic, run `python main.py --loop --use-llm` and simulate pressing `q` during recording to confirm exit handling is still reliable.

## PR & Documentation Notes

- Document any new dependencies or CLI arguments in `README.md` or `TTS_README.md` so future agents can find the same instructions.
- List manual verification commands you ran in the PR description (e.g., `python speech_to_text.py test.wav --model base`).
- Avoid committing `.env`, `uv.lock`, or other environment-specific artifacts unless the change strictly requires it; mention why in the PR if you must.

## References

- `README.md` – primary documentation for running the full pipeline.
- `TTS_README.md` – details Piper TTS usage.
- Module sources (especially `main.py`, `record_audio.py`, `speech_to_text.py`, `text_to_speech.py`) – follow their patterns when adding new features.

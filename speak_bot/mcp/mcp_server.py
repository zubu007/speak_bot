# MCP Server for speak_bot
"""
Model Context Protocol (MCP) server implementation for the speak_bot voice assistant.
This module provides MCP server functionality for the voice-first assistant pipeline.
"""

import sys
import asyncio
import json
import re
import subprocess
import os
from pathlib import Path
from datetime import datetime
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("speak_bot_controller")


def get_package_root() -> Path:
    """Get the root directory of the speak_bot package."""
    # mcp_server.py is in speak_bot/mcp/, so go up two levels to get package root
    return Path(__file__).parent.parent.parent


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


# ============================================================================
# Calendar Manager Helper Functions
# ============================================================================

def load_gog_prompt() -> str:
    """Load gog CLI examples from prompt file.
    
    Returns:
        Content of gog_prompt.txt as a string
        
    Raises:
        FileNotFoundError: If gog_prompt.txt is not found
    """
    prompt_path = Path(__file__).parent / "gog_prompt.txt"
    
    if not prompt_path.exists():
        raise FileNotFoundError(f"gog_prompt.txt not found at {prompt_path}")
    
    with open(prompt_path, 'r', encoding='utf-8') as f:
        return f.read()


def get_calendar_id(prompt_text: str) -> str:
    """Extract calendar_id from gog_prompt.txt content.
    
    Expects the first line to be in format: calendar_id="email@example.com"
    
    Args:
        prompt_text: Content of gog_prompt.txt
        
    Returns:
        Calendar ID (email address) or "primary" as fallback
    """
    # Parse first line: calendar_id="..."
    lines = prompt_text.strip().split('\n')
    if lines:
        first_line = lines[0].strip()
        match = re.match(r'calendar_id="([^"]+)"', first_line)
        if match:
            return match.group(1)
    
    # Fallback to "primary" if not found
    print("⚠️ Could not extract calendar_id from gog_prompt.txt, using 'primary' as fallback")
    return "primary"


def get_current_timezone_info() -> dict:
    """Get current date, time, and timezone information.
    
    Returns:
        dict with keys: today_date, current_time, timezone_offset, timezone_name
    """
    now = datetime.now()
    
    # Get timezone offset in format like "-08:00" or "+05:30"
    # Using astimezone() to get local timezone
    local_tz = now.astimezone()
    tz_offset = local_tz.strftime('%z')  # Format: +0530 or -0800
    
    # Format offset with colon: +05:30 or -08:00
    if len(tz_offset) == 5:
        tz_offset_formatted = f"{tz_offset[:3]}:{tz_offset[3:]}"
    else:
        tz_offset_formatted = tz_offset
    
    # Get timezone name (e.g., "PST", "EST")
    tz_name = local_tz.strftime('%Z')
    
    return {
        "today_date": now.strftime('%Y-%m-%d'),
        "current_time": now.strftime('%H:%M:%S'),
        "timezone_offset": tz_offset_formatted,
        "timezone_name": tz_name,
        "datetime_now": now
    }


def execute_gog_command(command: str, timeout: int = 30) -> tuple[bool, str]:
    """Execute gog CLI command and return success status + output.
    
    Args:
        command: The gog command to execute (full command as string)
        timeout: Command timeout in seconds (default: 30)
        
    Returns:
        tuple: (success: bool, output: str)
               success is True if returncode == 0
               output is stdout on success, stderr on failure
    """
    try:
        print(f"🔧 Executing: {command}")
        
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout
        )
        
        success = result.returncode == 0
        output = result.stdout if success else result.stderr
        
        if success:
            print(f"✅ Command succeeded")
        else:
            print(f"❌ Command failed with return code {result.returncode}")
            
        return success, output.strip()
        
    except subprocess.TimeoutExpired:
        error_msg = f"Command timed out after {timeout} seconds"
        print(f"⏱️ {error_msg}")
        return False, error_msg
        
    except Exception as e:
        error_msg = f"Error executing command: {str(e)}"
        print(f"⚠️ {error_msg}")
        return False, error_msg


# System prompt template for the internal LLM that generates gog commands
CALENDAR_LLM_SYSTEM_PROMPT = """You are a specialized assistant that converts natural language calendar requests into gog CLI commands.

Your task:
1. Parse the user's natural language request carefully
2. Determine the operation type: view/list, create, update, delete, or search
3. Extract relevant details: time, date, title, attendees, location, etc.
4. Generate the correct gog CLI command using the examples provided below
5. Return ONLY the complete command, no explanation or extra text

IMPORTANT RULES:
- Always use --json flag for structured output (NOT --plain)
- Times must be in RFC3339 format: YYYY-MM-DDTHH:MM:SS±HH:MM
- Use the timezone offset provided below for all timestamps
- For relative times (tomorrow, next Monday), calculate the actual date/time
- For delete operations, include --force flag to skip confirmation
- For update/delete operations that need an eventId, use search first
- Use the calendar_id provided below (not "primary" or "<calendarId>")
- Return ONLY the command, nothing else - not even quotes around it

CURRENT CONTEXT:
- Today's date: {TODAY_DATE}
- Current time: {CURRENT_TIME}
- Timezone: {TIMEZONE_NAME} ({TIMEZONE_OFFSET})
- Calendar ID: {CALENDAR_ID}

GOG CLI EXAMPLES AND REFERENCE:
{GOG_EXAMPLES}

EXAMPLE CONVERSIONS:

User: "show me today's events"
Command: gog calendar events {CALENDAR_ID} --today --json

User: "what's on my calendar tomorrow"
Command: gog calendar events {CALENDAR_ID} --tomorrow --json

User: "create a coffee meeting at 3pm tomorrow"
Command: gog calendar create {CALENDAR_ID} --summary "Coffee meeting" --from {TOMORROW_DATE}T15:00:00{TIMEZONE_OFFSET} --to {TOMORROW_DATE}T16:00:00{TIMEZONE_OFFSET} --json

User: "schedule team sync next monday at 10am for 1 hour"
Command: gog calendar create {CALENDAR_ID} --summary "Team sync" --from {NEXT_MONDAY_DATE}T10:00:00{TIMEZONE_OFFSET} --to {NEXT_MONDAY_DATE}T11:00:00{TIMEZONE_OFFSET} --json

User: "search for meetings with Anna this week"
Command: gog calendar search "Anna" --week --json

User: "find my project meetings"
Command: gog calendar search "project" --days 30 --json

IMPORTANT: Return ONLY the gog command, nothing else. No markdown, no quotes, no explanation.
"""


def generate_calendar_llm_prompt(gog_examples: str, calendar_id: str, tz_info: dict) -> str:
    """Generate the system prompt for the internal LLM with current context.
    
    Args:
        gog_examples: Content from gog_prompt.txt
        calendar_id: User's calendar ID (email)
        tz_info: Dictionary with timezone information from get_current_timezone_info()
        
    Returns:
        Formatted system prompt with all context filled in
    """
    from datetime import timedelta
    
    now = tz_info["datetime_now"]
    
    # Calculate tomorrow's date
    tomorrow = now + timedelta(days=1)
    tomorrow_date = tomorrow.strftime('%Y-%m-%d')
    
    # Calculate next Monday
    days_until_monday = (7 - now.weekday()) % 7
    if days_until_monday == 0:  # Today is Monday
        days_until_monday = 7  # Get next Monday
    next_monday = now + timedelta(days=days_until_monday)
    next_monday_date = next_monday.strftime('%Y-%m-%d')
    
    # Fill in the template
    prompt = CALENDAR_LLM_SYSTEM_PROMPT.format(
        TODAY_DATE=tz_info["today_date"],
        CURRENT_TIME=tz_info["current_time"],
        TIMEZONE_NAME=tz_info["timezone_name"],
        TIMEZONE_OFFSET=tz_info["timezone_offset"],
        CALENDAR_ID=calendar_id,
        GOG_EXAMPLES=gog_examples,
        TOMORROW_DATE=tomorrow_date,
        NEXT_MONDAY_DATE=next_monday_date
    )
    
    return prompt


def call_calendar_llm(natural_language_request: str, system_prompt: str) -> tuple[bool, str]:
    """Call the internal LLM to convert natural language to gog command.
    
    Args:
        natural_language_request: User's natural language calendar request
        system_prompt: The system prompt with gog examples and context
        
    Returns:
        tuple: (success: bool, command_or_error: str)
               success is True if LLM call succeeded
               command_or_error is the gog command on success, error message on failure
    """
    try:
        from openai import OpenAI
        
        # Get API key from environment
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            return False, "OPENAI_API_KEY not found in environment"
        
        client = OpenAI(api_key=api_key)
        
        print(f"🤖 Calling internal LLM to parse: '{natural_language_request}'")
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": natural_language_request}
            ],
            temperature=0.3,  # Lower temperature for more deterministic output
            max_tokens=300    # Enough for a gog command
        )
        
        gog_command = response.choices[0].message.content.strip()
        
        # Remove any markdown code blocks if present
        if gog_command.startswith("```"):
            lines = gog_command.split('\n')
            gog_command = '\n'.join(lines[1:-1]) if len(lines) > 2 else lines[1]
            gog_command = gog_command.strip()
        
        # Remove quotes if present
        if gog_command.startswith('"') and gog_command.endswith('"'):
            gog_command = gog_command[1:-1]
        if gog_command.startswith("'") and gog_command.endswith("'"):
            gog_command = gog_command[1:-1]
        
        print(f"✅ Generated command: {gog_command}")
        
        return True, gog_command
        
    except Exception as e:
        error_msg = f"Error calling internal LLM: {str(e)}"
        print(f"❌ {error_msg}")
        return False, error_msg


@mcp.tool()
async def stop_conversation():
    """Stop the conversation when the user wants to end the chat.
    
    Use this tool when the user expresses any intent to end the conversation,
    such as saying goodbye, indicating they want to stop, or showing the 
    conversation is finished. Be liberal in interpreting farewell intent.
    
    Examples of when to use:
    - "okay, goodbye"
    - "stop now" 
    - "that's all for now"
    - "see you later"
    - "I'm done"
    - "thanks, bye"
    - "goodbye"
    - "see ya"
    - "talk to you later"
    - "that's enough"
    
    Returns:
        dict: Contains farewell message and control action signal
    """
    goodbye_message = "Goodbye, sir."
    
    # Play TTS for goodbye message before returning control signal
    try:
        from speak_bot.text_to_speech import text_to_speech_streaming
        
        # Resolve model path (using default model)
        model_path = "tts_models/jarvis-medium.onnx"
        resolved_path = resolve_tts_model_path(model_path)
        
        # Play goodbye message via TTS
        print(f"🔊 Playing goodbye message: '{goodbye_message}'")
        text_to_speech_streaming(
            text=goodbye_message,
            model_path=resolved_path,
            use_cuda=True,
            verbose=True
        )
        print("✅ Goodbye message played successfully")
        
    except Exception as e:
        # Graceful degradation: log error but continue with stop
        print(f"⚠️ TTS playback failed: {e}")
        print("   Continuing with conversation stop...")
    
    # Return control signal (same as before)
    return {
        "message": goodbye_message,
        "control_action": "stop_conversation"
    }


@mcp.tool()
async def calendar_manager(request: str) -> dict:
    """Manage Google Calendar events using natural language.
    
    This tool accepts natural language requests to view, create, update, search,
    or delete calendar events. It uses the gog CLI to interact with Google Calendar.
    
    The tool processes your request by:
    1. Understanding your natural language input
    2. Converting it to the appropriate calendar command
    3. Executing the command
    4. Returning the results in a structured format
    
    Args:
        request: Natural language calendar request. Examples:
                 - "show me today's events"
                 - "what's on my calendar tomorrow"
                 - "create a coffee meeting at 3pm tomorrow"
                 - "schedule team sync next monday at 10am"
                 - "search for meetings with Anna"
                 - "find my project meetings this week"
                 - "create a meeting with bob@example.com at 2pm in conference room A"
    
    Returns:
        dict: {
            "success": bool,      # Whether the operation succeeded
            "command": str,       # The gog command that was executed
            "output": str,        # JSON output from gog CLI (or error message)
            "message": str        # Human-readable summary for the user
        }
    
    Note: This tool requires gog CLI to be installed and authenticated.
          Set up authentication with: gog auth manage
    """
    
    print(f"\n{'='*60}")
    print(f"📅 CALENDAR MANAGER - Processing Request")
    print(f"{'='*60}")
    print(f"Request: {request}")
    print()
    
    try:
        # Step 1: Load gog prompt examples
        print("📋 Step 1: Loading gog CLI examples...")
        try:
            gog_examples = load_gog_prompt()
            print(f"   ✅ Loaded {len(gog_examples)} chars from gog_prompt.txt")
        except FileNotFoundError as e:
            error_msg = f"gog_prompt.txt not found: {e}"
            print(f"   ❌ {error_msg}")
            return {
                "success": False,
                "command": "",
                "output": "",
                "message": f"Configuration error: {error_msg}"
            }
        
        # Step 2: Extract calendar ID and get timezone info
        print("\n🔧 Step 2: Getting calendar ID and timezone info...")
        calendar_id = get_calendar_id(gog_examples)
        tz_info = get_current_timezone_info()
        print(f"   ✅ Calendar ID: {calendar_id}")
        print(f"   ✅ Timezone: {tz_info['timezone_name']} ({tz_info['timezone_offset']})")
        print(f"   ✅ Date: {tz_info['today_date']} {tz_info['current_time']}")
        
        # Step 3: Generate system prompt for internal LLM
        print("\n🤖 Step 3: Generating system prompt for internal LLM...")
        system_prompt = generate_calendar_llm_prompt(gog_examples, calendar_id, tz_info)
        print(f"   ✅ System prompt ready ({len(system_prompt)} chars)")
        
        # Step 4: Call internal LLM to generate gog command
        print("\n💬 Step 4: Converting natural language to gog command...")
        llm_success, gog_command = call_calendar_llm(request, system_prompt)
        
        if not llm_success:
            print(f"   ❌ LLM call failed: {gog_command}")
            return {
                "success": False,
                "command": "",
                "output": "",
                "message": f"Could not process request: {gog_command}"
            }
        
        print(f"   ✅ Generated: {gog_command}")
        
        # Step 5: Validate the command (basic security check)
        print("\n🔒 Step 5: Validating command...")
        if not gog_command.startswith("gog calendar"):
            error_msg = "Generated command does not start with 'gog calendar'"
            print(f"   ❌ {error_msg}")
            return {
                "success": False,
                "command": gog_command,
                "output": "",
                "message": f"Security validation failed: {error_msg}"
            }
        print(f"   ✅ Command validated")
        
        # Step 6: Execute the gog command
        print("\n⚙️  Step 6: Executing gog command...")
        cmd_success, cmd_output = execute_gog_command(gog_command, timeout=30)
        
        # Step 7: Format the response
        print("\n📊 Step 7: Formatting response...")
        if cmd_success:
            # Try to parse as JSON for better formatting
            try:
                import json
                json_data = json.loads(cmd_output)
                
                # Create human-friendly message based on operation type
                if "events" in gog_command or "search" in gog_command:
                    # Listing or searching events
                    if isinstance(json_data, dict) and "items" in json_data:
                        event_count = len(json_data.get("items", []))
                        if event_count == 0:
                            message = "No events found."
                        elif event_count == 1:
                            message = "Found 1 event."
                        else:
                            message = f"Found {event_count} events."
                    else:
                        message = "Retrieved calendar information."
                        
                elif "create" in gog_command:
                    # Creating event
                    if isinstance(json_data, dict) and "summary" in json_data:
                        summary = json_data.get("summary", "Event")
                        message = f"Successfully created: {summary}"
                    else:
                        message = "Event created successfully."
                        
                elif "update" in gog_command:
                    # Updating event
                    message = "Event updated successfully."
                    
                elif "delete" in gog_command:
                    # Deleting event
                    message = "Event deleted successfully."
                else:
                    message = "Calendar operation completed successfully."
                    
            except json.JSONDecodeError:
                # Not JSON, use the output as-is
                message = "Calendar operation completed successfully."
            
            print(f"   ✅ {message}")
            
            result = {
                "success": True,
                "command": gog_command,
                "output": cmd_output,
                "message": message
            }
            
        else:
            # Command failed
            error_summary = cmd_output[:200] if cmd_output else "Unknown error"
            message = f"Calendar operation failed: {error_summary}"
            print(f"   ❌ {message}")
            
            result = {
                "success": False,
                "command": gog_command,
                "output": cmd_output,
                "message": message
            }
        
        print(f"\n{'='*60}")
        print(f"📅 CALENDAR MANAGER - Request Complete")
        print(f"{'='*60}\n")
        
        return result
        
    except Exception as e:
        error_msg = f"Unexpected error in calendar_manager: {str(e)}"
        print(f"\n❌ {error_msg}")
        print(f"{'='*60}\n")
        
        return {
            "success": False,
            "command": "",
            "output": "",
            "message": error_msg
        }


def main():
    """Main entry point for the MCP server."""
    try:
        # Run the FastMCP server
        mcp.run()
    except KeyboardInterrupt:
        print("\nMCP server shutdown.")
        sys.exit(0)
    except Exception as e:
        print(f"MCP server error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

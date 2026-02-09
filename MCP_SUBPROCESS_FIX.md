# MCP Subprocess Fix

## Problem
The `stop_conversation` tool (MCP) was not working when the `talk` command was spawned as a subprocess by the wake_word project's agent_controller.py. The tool worked fine when running `talk` directly from the command line.

## Root Cause
The issue was caused by how the MCP client managed asyncio event loops in subprocess environments:

1. **Event Loop Initialization**: The background event loop wasn't properly isolated and could fail in subprocess contexts
2. **Server Connection Method**: Used `uv` command which may not be available or properly configured in subprocess environments
3. **Error Handling**: Limited error reporting made debugging difficult
4. **Subprocess Detection**: No awareness of whether running as a subprocess vs direct invocation

## Changes Made

### 1. Improved Event Loop Management (`mcp_client.py:_start_event_loop`)
- Added proper exception handling in the event loop thread
- Added logging to track event loop creation and status
- Added verification that the loop was created successfully
- Signal `_loop_ready` even on error to prevent deadlocks

### 2. Subprocess-Safe MCP Server Connection (`mcp_client.py:_async_connect_to_server`)
- **Changed from `uv` to direct Python interpreter**: Instead of relying on `uv --directory ... run`, now uses `sys.executable` to directly invoke the MCP server script
- Added comprehensive logging at each connection step
- Added proper environment variable passing
- Better exception handling with full traceback logging

### 3. Enhanced Connection Wrapper (`mcp_client.py:connect_to_server`)
- Added verification that event loop is running before attempting connection
- Improved error messages with proper exception chaining
- Added timeout handling with clear error messages

### 4. Robust Query Execution (`mcp_client.py:_run_async_query`)
- Added fallback logic if event loop is not running
- Enhanced error logging with full context
- Added debug messages to track execution path

### 5. Subprocess Detection (`mcp_client.py:_detect_subprocess`)
- Added detection of parent process to identify subprocess context
- Provides helpful logging when running as a subprocess
- Uses `psutil` if available, gracefully degrades if not

### 6. Better User Feedback (`main.py:run_conversation`)
- Added detailed connection status messages
- Shows MCP server path being used
- Provides clear instructions when MCP is available vs fallback mode
- Added basic logging configuration for MCP connection details

## Testing
The fix was tested with:
```python
from speak_bot.mcp_client import LLMResponseGenerator
llm = LLMResponseGenerator()
llm.connect_to_server('speak_bot/mcp/mcp_server.py')
```

Successful output shows:
- Subprocess detection working
- Event loop created successfully
- MCP server connection established
- Tool `stop_conversation` available

## Usage
No changes required to existing code. The fix is transparent to users:

### Direct Usage (already working)
```bash
talk --loop --use-llm --use-tts
```

### Subprocess Usage (now fixed)
```python
# From agent_controller.py or similar
import subprocess
subprocess.Popen(["/path/to/talk", "--loop", "--use-llm", "--use-tts"])
```

## Key Improvements
1. **Subprocess-safe**: Event loop management works reliably when spawned as subprocess
2. **Better diagnostics**: Comprehensive logging helps debug connection issues
3. **Direct Python invocation**: More reliable than relying on `uv` wrapper
4. **Graceful degradation**: Falls back to non-MCP mode if connection fails
5. **User-friendly**: Clear messages about what functionality is available

import asyncio
import json
import os
import sys
import threading
import logging
from contextlib import AsyncExitStack
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv
from openai import OpenAI, APIError
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

load_dotenv()  # load environment variables from .env

# Configure logging for better debugging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LLMResponseGenerator:
    def __init__(self, api_key: str | None = None, model: str = "gpt-4o-mini"):
        """Initialize the LLM response generator with MCP support.
        
        Args:
            api_key: OpenAI API key. If None, will load from OPENAI_API_KEY env var.
            model: Model to use (default: gpt-4o-mini for faster, cheaper responses)
        """
        # Initialize session and client objects
        self.session: ClientSession | None = None
        self.exit_stack = AsyncExitStack()
        self._loop_thread = None  # Background thread for persistent event loop
        self._loop = None  # The event loop running in the background thread
        self._loop_ready = threading.Event()  # Signal when loop is ready
        
        # Detect subprocess environment for better debugging
        self._is_subprocess = self._detect_subprocess()
        if self._is_subprocess:
            logger.info("Running in subprocess environment")
        
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenAI API key not found. Please set OPENAI_API_KEY in .env file "
                "or pass it directly."
            )
        self.client = OpenAI(api_key=self.api_key)
        self.model = model
        
        # Control signal tracking
        self._last_control_signal = None
        self._last_control_message = None
        
        self.system_prompt = (
            "You are Jarvis, a helpful and friendly AI assistant. Keep your responses short, "
            "concise, and conversational (1-3 sentences max). Avoid long explanations "
            "or verbose text. Be natural and professional. "
            "Always address the user as 'sir' at the end of your responses, mimicking "
            "Jarvis from Iron Man."
        )
    
    def _detect_subprocess(self) -> bool:
        """Detect if we're running in a subprocess environment.
        
        Returns:
            True if running as a subprocess, False otherwise
        """
        try:
            # Check if parent process is not the shell/terminal
            import psutil
            current = psutil.Process()
            parent = current.parent()
            if parent:
                parent_name = parent.name().lower()
                # Common shell/terminal names
                shells = ['bash', 'zsh', 'sh', 'fish', 'tcsh', 'csh', 'terminal', 'iterm']
                is_subprocess = not any(shell in parent_name for shell in shells)
                if is_subprocess:
                    logger.info(f"Detected parent process: {parent_name} (PID: {parent.pid})")
                return is_subprocess
        except (ImportError, Exception) as e:
            logger.debug(f"Could not detect subprocess status: {e}")
            # If psutil is not available, assume we're not in a subprocess
            return False
        return False
    
    def _start_event_loop(self):
        """Start a persistent event loop in a background thread
        
        This method creates a dedicated background thread with its own event loop
        to handle MCP operations. This approach is subprocess-safe and avoids
        conflicts with any existing event loops.
        """
        def run_loop():
            try:
                # Create a fresh event loop for this thread
                self._loop = asyncio.new_event_loop()
                asyncio.set_event_loop(self._loop)
                logger.info("MCP event loop created and set for background thread")
                self._loop_ready.set()  # Signal that loop is ready
                self._loop.run_forever()
            except Exception as e:
                logger.error(f"Error in event loop thread: {e}")
                self._loop_ready.set()  # Signal even on error to prevent deadlock
        
        self._loop_thread = threading.Thread(target=run_loop, daemon=True)
        self._loop_thread.start()
        self._loop_ready.wait()  # Wait for loop to be ready
        
        # Verify the loop was created successfully
        if self._loop is None:
            logger.error("Failed to create event loop in background thread")
            raise RuntimeError("Failed to initialize MCP event loop")

    async def _async_connect_to_server(self, server_script_path: str):
        """Async method to connect to an MCP server (runs in background loop)

        Args:
            server_script_path: Path to the server script (.py or .js)
        """
        is_python = server_script_path.endswith(".py")
        is_js = server_script_path.endswith(".js")
        if not (is_python or is_js):
            raise ValueError("Server script must be a .py or .js file")

        try:
            if is_python:
                path = Path(server_script_path).resolve()
                logger.info(f"Connecting to Python MCP server: {path}")
                
                # Get the current environment and ensure proper paths
                env = os.environ.copy()
                
                # Try to use the same Python interpreter that's running this script
                python_exe = sys.executable
                logger.info(f"Using Python interpreter: {python_exe}")
                
                # For subprocess contexts, use Python directly instead of uv
                # This is more reliable when spawned from another process
                server_params = StdioServerParameters(
                    command=python_exe,
                    args=[str(path)],
                    env=env,
                )
                
                logger.info(f"MCP server command: {python_exe} {path}")
            else:
                logger.info(f"Connecting to Node.js MCP server: {server_script_path}")
                server_params = StdioServerParameters(
                    command="node", 
                    args=[server_script_path], 
                    env=None
                )

            logger.info("Starting MCP server connection...")
            stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
            self.stdio, self.write = stdio_transport
            logger.info("MCP stdio transport established")
            
            self.session = await self.exit_stack.enter_async_context(ClientSession(self.stdio, self.write))
            logger.info("MCP client session created")
            
            await self.session.initialize()
            logger.info("MCP session initialized")

            # List available tools
            response = await self.session.list_tools()
            tools = response.tools
            logger.info(f"Connected to MCP server with {len(tools)} tool(s)")
            print("\nConnected to server with tools:", [tool.name for tool in tools])
            
        except Exception as e:
            logger.error(f"Failed to connect to MCP server: {e}", exc_info=True)
            raise
    
    def connect_to_server(self, server_script_path: str):
        """Connect to an MCP server (synchronous wrapper)

        Args:
            server_script_path: Path to the server script (.py or .js)
            
        Raises:
            RuntimeError: If connection fails
        """
        try:
            # Start background event loop if not already running
            if self._loop is None:
                logger.info("Starting background event loop for MCP")
                self._start_event_loop()
            
            # Verify the loop is running
            if self._loop is None or not self._loop.is_running():
                raise RuntimeError("Event loop failed to start properly")
            
            logger.info(f"Scheduling MCP connection to: {server_script_path}")
            
            # Run the async connect in the background loop
            future = asyncio.run_coroutine_threadsafe(
                self._async_connect_to_server(server_script_path), 
                self._loop
            )
            
            # Wait for connection with timeout
            future.result(timeout=30)  # 30 second timeout for connection
            logger.info("MCP server connection completed successfully")
            
        except Exception as e:
            logger.error(f"Failed to connect to MCP server: {e}", exc_info=True)
            raise RuntimeError(f"MCP server connection failed: {e}") from e

    async def process_query(self, query: str, history: List[Dict[str, str]] | None = None) -> str:
        """Process a query using OpenAI and available tools"""
        if not self.session:
            raise ValueError("No MCP session connected. Call connect_to_server() first.")
            
        messages: List[Dict[str, Any]] = [{"role": "system", "content": self.system_prompt}]
        
        # Add conversation history if provided
        if history:
            messages.extend(history)
            
        messages.append({"role": "user", "content": query})

        response = await self.session.list_tools()
        available_tools = []
        
        for tool in response.tools:
            available_tools.append({
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.inputSchema
                }
            })

        # Initial OpenAI API call
        try:
            if available_tools:
                # Type: ignore the typing issue - this is a known limitation
                response = self.client.chat.completions.create(  # type: ignore
                    model=self.model, 
                    max_tokens=1000, 
                    messages=messages,  # type: ignore
                    tools=available_tools,  # type: ignore
                    temperature=0.7
                )
            else:
                response = self.client.chat.completions.create(  # type: ignore
                    model=self.model, 
                    max_tokens=1000, 
                    messages=messages,  # type: ignore
                    temperature=0.7
                )
        except APIError as e:
            return f"Error calling OpenAI API: {str(e)}"

        # Process response and handle tool calls
        final_text = []
        message = response.choices[0].message

        # Add assistant's text content if present
        if message.content:
            final_text.append(message.content)

        # Handle tool calls if present
        if hasattr(message, 'tool_calls') and message.tool_calls:
            # Add the assistant's message to conversation history
            assistant_message: Dict[str, Any] = {
                "role": "assistant",
                "content": message.content or ""
            }
            
            # Add tool_calls to the message
            tool_calls_data = []
            for tool_call in message.tool_calls:
                tool_calls_data.append({
                    "id": tool_call.id,
                    "type": "function",
                    "function": {
                        "name": tool_call.function.name,
                        "arguments": tool_call.function.arguments
                    }
                })
            
            assistant_message["tool_calls"] = tool_calls_data
            messages.append(assistant_message)

            # Process each tool call
            for tool_call in message.tool_calls:
                try:
                    # Extract tool call information
                    tool_name = tool_call.function.name
                    tool_args_str = tool_call.function.arguments
                    
                    try:
                        tool_args = json.loads(tool_args_str)
                    except json.JSONDecodeError:
                        tool_args = {}

                    # Execute tool call
                    result = await self.session.call_tool(tool_name, tool_args)
                    
                    # Handle structured tool results with control signals
                    # MCP returns result.content as a list of TextContent objects
                    if hasattr(result, 'content') and result.content:
                        if isinstance(result.content, list) and len(result.content) > 0:
                            # Extract text from first TextContent object
                            first_content = result.content[0]
                            result_content_str = first_content.text if hasattr(first_content, 'text') else str(first_content)
                        else:
                            result_content_str = str(result.content)
                    else:
                        result_content_str = str(result)
                    
                    try:
                        # Try to parse as JSON for structured responses
                        result_data = json.loads(result_content_str)
                        if isinstance(result_data, dict) and "control_action" in result_data:
                            self._last_control_signal = result_data["control_action"]
                            self._last_control_message = result_data.get("message", "")
                            result_content = result_data.get("message", result_content_str)
                        else:
                            result_content = result_content_str
                    except (json.JSONDecodeError, TypeError) as e:
                        # Handle non-JSON responses
                        result_content = result_content_str
                    
                    # Add tool result to conversation
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": result_content
                    })
                    
                except Exception as e:
                    # Handle tool execution errors
                    error_msg = f"Error calling tool: {str(e)}"
                    messages.append({
                        "role": "tool",
                        "tool_call_id": getattr(tool_call, 'id', 'unknown'),
                        "content": error_msg
                    })

            # Get next response from OpenAI with tool results
            try:
                response = self.client.chat.completions.create(  # type: ignore
                    model=self.model,
                    max_tokens=1000,
                    messages=messages,  # type: ignore
                    temperature=0.7
                )

                # Add the final response
                final_response = response.choices[0].message.content
                if final_response:
                    final_text.append(final_response)
            except APIError as e:
                final_text.append(f"Error getting follow-up response: {str(e)}")

        return "\n".join(final_text) if final_text else "I apologize, but I couldn't generate a response."

    def get_last_control_signal(self) -> tuple[str | None, str | None]:
        """Get and clear the last control signal from tool execution.
        
        Returns:
            tuple: (control_action, control_message) or (None, None)
        """
        signal = self._last_control_signal
        message = self._last_control_message
        
        # Clear after reading
        self._last_control_signal = None
        self._last_control_message = None
        
        return signal, message

    def get_response(
        self,
        text: str,
        stream: bool = False,
        temperature: float = 0.7,
        history: List[Dict[str, str]] | None = None,
        return_control_signal: bool = False,
    ) -> str | tuple[str, str | None]:
        """Get a response from the LLM for the given text.
        
        Args:
            text: The transcribed text to send to the LLM
            stream: If True, returns a generator for streaming response (NOT IMPLEMENTED for MCP).
                    For now, returns the complete response as a string.
            temperature: Controls randomness (0.0-2.0). Lower = more deterministic.
            history: Previous conversation history
            return_control_signal: If True, returns (response, control_signal) tuple
        
        Returns:
            str: Complete response text (if return_control_signal=False)
            tuple: (response, control_signal) (if return_control_signal=True)
            
        Raises:
            ValueError: If input text is empty
        """
        if not text or not text.strip():
            raise ValueError("Input text cannot be empty")

        # Clear any previous control signals
        self._last_control_signal = None
        self._last_control_message = None

        # If no MCP session is connected, fall back to simple OpenAI chat
        if not self.session:
            response = self._get_simple_response(text, stream, temperature, history)
            if return_control_signal:
                return response, None
            return response
            
        # Note: streaming is not yet implemented for MCP tool calling
        # For now, we'll always return the complete response
        if stream:
            print("Note: Streaming not yet implemented for MCP. Returning complete response.")
            
        # Run the async process_query method
        try:
            # Always use asyncio.run which creates a fresh event loop
            # This avoids the task scope conflicts from nested loops
            response = self._run_async_query(text, history)
                
            # Get control signal after processing
            control_signal, _ = self.get_last_control_signal()
            
            if return_control_signal:
                return response, control_signal
            return response
            
        except Exception as e:
            print(f"Error with MCP processing, falling back to simple response: {e}")
            fallback_response = self._get_simple_response(text, stream, temperature, history)
            if return_control_signal:
                return fallback_response, None
            return fallback_response

    def _run_async_query(self, text: str, history: List[Dict[str, str]] | None = None) -> str:
        """Helper method to run async query in the same event loop as the MCP session
        
        This method ensures that the async query runs in the correct event loop context,
        which is critical for subprocess environments.
        """
        if self._loop is not None and self._loop.is_running():
            # If we have a stored loop and it's running, schedule the coroutine in that loop
            logger.debug("Running query in background event loop")
            try:
                future = asyncio.run_coroutine_threadsafe(self.process_query(text, history), self._loop)
                result = future.result(timeout=60)  # 60 second timeout
                logger.debug("Query completed successfully")
                return result
            except Exception as e:
                logger.error(f"Error running query in background loop: {e}", exc_info=True)
                raise
        else:
            # No stored loop or it's not running - this shouldn't happen if connect_to_server succeeded
            logger.warning("No running event loop found, attempting to create new one")
            try:
                return asyncio.run(self.process_query(text, history))
            except Exception as e:
                logger.error(f"Error creating new event loop: {e}", exc_info=True)
                raise

    def _get_simple_response(
        self,
        text: str,
        stream: bool = False,
        temperature: float = 0.7,
        history: List[Dict[str, str]] | None = None,
    ) -> str:
        """Fallback method for simple OpenAI response without MCP tools."""
        try:
            messages: List[Dict[str, Any]] = [{"role": "system", "content": self.system_prompt}]
            if history:
                messages.extend(history)
            messages.append({"role": "user", "content": text})

            response = self.client.chat.completions.create(  # type: ignore
                model=self.model,
                messages=messages,  # type: ignore
                temperature=temperature,
                stream=stream,
            )

            if stream:
                # Simple streaming for fallback mode
                accumulated_text = ""
                for chunk in response:
                    if chunk.choices[0].delta.content:
                        content = chunk.choices[0].delta.content
                        print(content, end="", flush=True)
                        accumulated_text += content
                return accumulated_text
            else:
                return response.choices[0].message.content or "I apologize, but I couldn't generate a response."
                
        except Exception as e:
            raise Exception(f"Error communicating with OpenAI API: {e}")

    async def cleanup(self):
        """Clean up resources"""
        await self.exit_stack.aclose()

# MCP Server for speak_bot
"""
Model Context Protocol (MCP) server implementation for the speak_bot voice assistant.
This module provides MCP server functionality for the voice-first assistant pipeline.
"""

import sys
import asyncio
import json
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("speak_bot_controller")

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
    return {
        "message": "Goodbye, sir.",
        "control_action": "stop_conversation"
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

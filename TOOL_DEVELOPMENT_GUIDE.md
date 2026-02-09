# Tool Development Guide

## Overview

This guide explains how to add new tools to the speak_bot system. The system uses the Model Context Protocol (MCP) for tool integration, allowing the LLM to call tools based on their descriptions without modifying the system prompt.

## Design Philosophy

**Key Principle**: Tool usage logic should be defined in the tool's description, NOT in the system prompt.

### Why This Approach?

1. **Scalability**: Adding 10 or 100 tools won't bloat the system prompt
2. **Maintainability**: Tool logic lives with tool definitions
3. **Flexibility**: Tools can be dynamically loaded/unloaded at runtime
4. **Standard**: Follows OpenAI's recommended function calling patterns

## System Prompt Design

The system prompt in `mcp_client.py` focuses on general assistant behavior:

```python
self.system_prompt = (
    "You are Jarvis, a helpful and friendly AI assistant. Keep your responses short, "
    "concise, and conversational (1-3 sentences max). Avoid long explanations "
    "or verbose text. Be natural and professional. "
    "Always address the user as 'sir' at the end of your responses, mimicking "
    "Jarvis from Iron Man."
)
```

**Notice**: No tool-specific instructions! The LLM learns about tools from the `tools` parameter in the API call.

## How to Add a New Tool

### Step 1: Define the Tool in MCP Server

Edit `speak_bot/mcp/mcp_server.py` and add your tool:

```python
@mcp.tool()
async def your_tool_name(parameter1: str, parameter2: int = 0):
    """Brief one-line description of what the tool does.
    
    Use this tool when [specific conditions for when to use the tool].
    Be [liberal/conservative/specific] in interpreting [relevant conditions].
    
    Args:
        parameter1: Description of parameter1
        parameter2: Description of parameter2 (default: 0)
    
    Examples of when to use:
    - "specific user phrase example 1"
    - "specific user phrase example 2"
    - "specific user phrase example 3"
    
    Examples of when NOT to use:
    - "situation where tool should not be used"
    
    Returns:
        dict: Description of return value structure
    """
    # Your tool implementation here
    result = do_something(parameter1, parameter2)
    
    # For tools that need to signal control actions back to the main loop
    return {
        "data": result,
        "control_action": "optional_control_signal"  # Optional
    }
```

### Step 2: Tool Description Best Practices

A comprehensive tool description should include:

1. **Brief Purpose** (first line): What the tool does
2. **Usage Conditions**: When to use this tool
3. **Interpretation Guidance**: How liberal/strict to be in interpreting user intent
4. **Parameters**: Clear description of all parameters
5. **Examples**: Specific phrases that should trigger the tool
6. **Counter-Examples**: Situations where the tool should NOT be used (if relevant)
7. **Return Value**: What the tool returns

### Step 3: Test Your Tool

Test the tool independently:

```python
from speak_bot.mcp_client import LLMResponseGenerator
from pathlib import Path

llm = LLMResponseGenerator(model='gpt-4o-mini')
mcp_server_path = Path('speak_bot/mcp/mcp_server.py')
llm.connect_to_server(str(mcp_server_path))

# The tool should appear in the connected tools list
# Test with various user inputs to verify tool is called correctly
```

## Example Tool Implementations

### Example 1: Calendar Event Tool

```python
@mcp.tool()
async def add_calendar_event(
    title: str, 
    date: str, 
    time: str, 
    duration_minutes: int = 60,
    description: str = ""
):
    """Add an event to the user's calendar.
    
    Use this tool when the user wants to schedule, create, or add an event, 
    appointment, meeting, or reminder to their calendar. Be liberal in 
    interpreting scheduling intent.
    
    Args:
        title: The title/name of the event
        date: Date in YYYY-MM-DD format
        time: Time in HH:MM format (24-hour)
        duration_minutes: Duration in minutes (default: 60)
        description: Optional description or notes
    
    Examples of when to use:
    - "schedule a meeting with John tomorrow at 2pm"
    - "add dentist appointment next Friday at 10am"
    - "remind me to call mom on her birthday"
    - "create a calendar event for the team lunch"
    - "I have a doctor's appointment on March 15th at 3:30pm"
    
    Examples of when NOT to use:
    - "what's on my calendar today?" (use read_calendar_events instead)
    - "when is my next meeting?" (use read_calendar_events instead)
    
    Returns:
        dict: Event details and confirmation message
    """
    # Implementation here
    event_id = create_calendar_event(title, date, time, duration_minutes, description)
    
    return {
        "event_id": event_id,
        "message": f"Event '{title}' added to calendar for {date} at {time}, sir."
    }
```

### Example 2: Email Reading Tool

```python
@mcp.tool()
async def read_recent_emails(count: int = 5, unread_only: bool = False):
    """Read recent emails from the user's inbox.
    
    Use this tool when the user wants to check, read, or get information about
    their emails. Be liberal in interpreting email-checking intent.
    
    Args:
        count: Number of emails to retrieve (default: 5, max: 20)
        unread_only: If True, only return unread emails (default: False)
    
    Examples of when to use:
    - "do I have any new emails?"
    - "check my inbox"
    - "what are my latest emails?"
    - "show me unread messages"
    - "any important emails today?"
    
    Examples of when NOT to use:
    - "send an email to John" (use send_email instead)
    - "delete that email" (use delete_email instead)
    
    Returns:
        dict: List of email summaries with sender, subject, and preview
    """
    # Implementation here
    emails = fetch_emails(count, unread_only)
    
    return {
        "emails": emails,
        "count": len(emails),
        "message": f"You have {len(emails)} {'unread' if unread_only else 'recent'} emails, sir."
    }
```

## Control Signals

Some tools may need to signal control actions back to the main conversation loop (like `stop_conversation` does).

### How Control Signals Work:

1. **Tool returns a dict with `control_action` key**:
   ```python
   return {
       "message": "Response text",
       "control_action": "some_signal"
   }
   ```

2. **MCP client extracts the control signal** (in `mcp_client.py`):
   ```python
   self._last_control_signal = result_data["control_action"]
   ```

3. **Main loop checks for the signal** (in `main.py`):
   ```python
   if tool_signal == "some_signal":
       # Handle the control action
       sys.exit(0)
   ```

### Adding New Control Signals:

If you need a new control action:

1. Have your tool return it in the `control_action` field
2. Add handling logic in `main.py:run_conversation()` to respond to it
3. Document the control signal in your tool's docstring

## Testing Guidelines

When adding a new tool:

1. **Test with explicit phrases**: "add a calendar event for tomorrow"
2. **Test with implicit phrases**: "I need to remember to call John tomorrow"
3. **Test with edge cases**: Ambiguous requests that could match multiple tools
4. **Test with negative cases**: Requests that should NOT trigger the tool
5. **Verify graceful failure**: What happens if the tool fails?

## Common Pitfalls to Avoid

1. ❌ **Don't add tool-specific instructions to system prompt**
   - Let the tool description handle usage logic
   
2. ❌ **Don't make tool descriptions too brief**
   - Include examples and usage conditions
   
3. ❌ **Don't forget parameter type hints**
   - FastMCP uses these for validation
   
4. ❌ **Don't forget async/await**
   - All MCP tools must be async functions
   
5. ❌ **Don't return raw strings from tools**
   - Return structured dicts with clear fields

## Tool Categories for Future Development

Based on your plans, here are recommended tool categories:

### Calendar Tools
- `add_calendar_event`: Create new events
- `read_calendar_events`: Query upcoming events
- `edit_calendar_event`: Modify existing events
- `delete_calendar_event`: Remove events
- `check_availability`: Check if user is free at a certain time

### Email Tools
- `read_recent_emails`: Read latest emails
- `search_emails`: Search emails by criteria
- `send_email`: Send new email
- `reply_to_email`: Reply to specific email
- `mark_email_read`: Mark email as read/unread

### Conversation Control Tools (existing)
- `stop_conversation`: End the conversation

## Architecture Notes

- **MCP Server**: Defined in `speak_bot/mcp/mcp_server.py`
- **MCP Client**: Defined in `speak_bot/mcp_client.py`
- **Tool Usage**: LLM decides based on tool descriptions via OpenAI function calling
- **Tool Execution**: Async execution through MCP protocol
- **Result Handling**: Structured responses with optional control signals

## Resources

- [OpenAI Function Calling Guide](https://platform.openai.com/docs/guides/function-calling)
- [Model Context Protocol Documentation](https://modelcontextprotocol.io/)
- [FastMCP Documentation](https://github.com/jlowin/fastmcp)

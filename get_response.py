from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, ToolMessage



def extract_final_response(agent_response: Any) -> str:
    """Extract the final text response from a LangChain agent invocation.

    LangChain agents can return responses in multiple formats depending on the
    agent implementation and model provider. This utility extracts the final
    user-facing response while supporting tool execution workflows.

    Supported response types:
        - str
        - AIMessage
        - BaseMessage
        - dict returned by agent.invoke()
        - LangGraph state dictionaries
        - Responses containing tool calls and tool results

    The function attempts extraction in the following order:

        1. "output" key (classic AgentExecutor)
        2. Last AIMessage in "messages"
        3. AIMessage.content
        4. BaseMessage.content
        5. String conversion as a fallback

    Args:
        agent_response: Raw response returned from ``agent.invoke()``.

    Returns:
        The extracted final response text.

    Raises:
        RuntimeError: If the response cannot be parsed.
    """
    try:
        if agent_response is None:
            return ""

        # Already a string
        if isinstance(agent_response, str):
            return agent_response

        # AIMessage
        if isinstance(agent_response, AIMessage):
            return _extract_message_content(agent_response)

        # Generic BaseMessage
        if isinstance(agent_response, BaseMessage):
            return _extract_message_content(agent_response)

        # Dictionary (most common)
        if isinstance(agent_response, dict):

            # Classic AgentExecutor
            if "output" in agent_response:
                return str(agent_response["output"])

            # LangGraph / create_react_agent
            if "messages" in agent_response:
                messages = agent_response["messages"]

                if isinstance(messages, list):
                    # Walk backwards to find the final AIMessage
                    for message in reversed(messages):
                        # Tool returned directly
                        if isinstance(message, ToolMessage):
                            if message.content:
                                return _extract_message_content(message)
                            
                        if isinstance(message, AIMessage):
                            return _extract_message_content(message)

                    # Fallback: last message
                    if messages:
                        last = messages[-1]
                        if isinstance(last, BaseMessage):
                            return _extract_message_content(last)
                        return str(last)

            # Generic fallback
            for key in ("response", "answer", "result", "text"):
                if key in agent_response:
                    return str(agent_response[key])

        print(
            "Unexpected agent response type: %s",
            type(agent_response).__name__,
        )
        return str(agent_response)

    except Exception as exc:
        print("Failed to extract agent response.")
        raise RuntimeError(
            "Unable to extract the final response from agent output."
        ) from exc


def _extract_message_content(message: BaseMessage) -> str:
    """Extract textual content from a LangChain message.

    Handles plain text, structured content blocks, and tool execution messages.

    Args:
        message: LangChain message.

    Returns:
        Extracted text content.
    """
    content = message.content

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts: list[str] = []

        for item in content:
            if isinstance(item, str):
                parts.append(item)

            elif isinstance(item, dict):
                # OpenAI / Anthropic content blocks
                text = (
                    item.get("text")
                    or item.get("content")
                    or item.get("output")
                )

                if text:
                    parts.append(str(text))

        return "\n".join(parts).strip()

    return str(content)
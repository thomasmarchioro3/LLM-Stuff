from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool

@tool
def add(a: int, b: int) -> int:
    """Add two integers.

    Args:
        a: First integer
        b: Second integer
    """
    return a + b


@tool
def multiply(a: int, b: int) -> int:
    """Multiply two integers.

    Args:
        a: First integer
        b: Second integer
    """
    return a * b

tools = [add, multiply]


if __name__ == "__main__":

    
    DEFAULT_MODEL_NAME = "llama3.1:8b"

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL_NAME)
    args = parser.parse_args()

    model_name: str = f"{args.model}"
    print(f"Using model: {model_name}")

    llm = init_chat_model(model_name, model_provider="ollama")

    llm_with_tools = llm.bind_tools(tools)
    
    query = "What is 3239 * 17?"

    messages = []
    messages.append(HumanMessage(query))

    ai_msg = llm_with_tools.invoke(messages)

    messages.append(ai_msg)
    
    for tool_call in ai_msg.tool_calls:
        selected_tool = {"add": add, "multiply": multiply}[tool_call["name"].lower()]
        tool_msg = selected_tool.invoke(tool_call)
        print("Tool message:", tool_msg)
        messages.append(tool_msg)
    
    answer = llm_with_tools.invoke(messages)

    print(answer)




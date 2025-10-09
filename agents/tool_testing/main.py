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

    # ollama._types.ResponseError: registry.ollama.ai/library/deepseek-r1:latest does not support tools (status code: 400)
    # Issue:  https://github.com/ollama/ollama/issues/8517
    #MODEL_NAME = "deepseek-r1:latest"
    
    MODEL_NAME = "deepseek-gg"

    # content='The result of 3239 multiplied by 17 is 55,063.' additional_kwargs={} response_metadata={'model': 'llama3.1:8b', 'created_at': '2025-09-15T18:30:31.016660597Z', 'done': True, 'done_reason': 'stop', 'total_duration': 706448958, 'load_duration': 24817128, 'prompt_eval_count': 98, 'prompt_eval_duration': 56166750, 'eval_count': 17, 'eval_duration': 621458839, 'model_name': 'llama3.1:8b'} id='run--29ae9643-ab16-4ddf-b1f5-c2d8fb16773b-0' usage_metadata={'input_tokens': 98, 'output_tokens': 17, 'total_tokens': 115}
    #MODEL_NAME = "llama3.1:8b"
    
    # content='The result of 3239 multiplied by 17 is 55063.' additional_kwargs={} response_metadata={'model': 'qwen2.5:latest', 'created_at': '2025-09-15T18:37:28.321741583Z', 'done': True, 'done_reason': 'stop', 'total_duration': 766373329, 'load_duration': 27311868, 'prompt_eval_count': 309, 'prompt_eval_duration': 13275759, 'eval_count': 22, 'eval_duration': 711596858, 'model_name': 'qwen2.5:latest'} id='run--2aec1ed0-9c40-467a-bb07-473e353838e3-0' usage_metadata={'input_tokens': 309, 'output_tokens': 22, 'total_tokens': 331}
    #MODEL_NAME = "qwen2.5:latest"

    # content='<think>\n</think>\n\nThe result of $3239 \\times 17$ is **55063**.' additional_kwargs={} response_metadata={'model': 'qwen3:8b', 'created_at': '2025-09-15T18:35:46.441166846Z', 'done': True, 'done_reason': 'stop', 'total_duration': 1419234056, 'load_duration': 177476771, 'prompt_eval_count': 353, 'prompt_eval_duration': 73993943, 'eval_count': 28, 'eval_duration': 1154809041, 'model_name': 'qwen3:8b'} id='run--5dc7cbc3-0578-4d44-8860-86f55a6fdc99-0' usage_metadata={'input_tokens': 353, 'output_tokens': 28, 'total_tokens': 381}
    #MODEL_NAME = "qwen3:8b"
    
    # content=' The result of multiplying 3239 and 17 is 55063.' additional_kwargs={} response_metadata={'model': 'mistral:latest', 'created_at': '2025-09-15T18:48:00.799439419Z', 'done': True, 'done_reason': 'stop', 'total_duration': 727540051, 'load_duration': 4201840, 'prompt_eval_count': 58, 'prompt_eval_duration': 7496174, 'eval_count': 23, 'eval_duration': 715201745, 'model_name': 'mistral:latest'} id='run--dc4f25f0-e585-41f7-96a9-3d0c1f9d2b25-0' usage_metadata={'input_tokens': 58, 'output_tokens': 23, 'total_tokens': 81}
    #MODEL_NAME = "mistral:latest"

    llm = init_chat_model(MODEL_NAME, model_provider="ollama")

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




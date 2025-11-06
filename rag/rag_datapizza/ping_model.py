from datapizza.clients.openai_like import OpenAILikeClient

# Create client for Ollama
client = OpenAILikeClient(
    api_key="",  # Ollama doesn't require an API key
    model="llama3.1:8b",
    system_prompt="You are a helpful assistant.",
    base_url="http://localhost:11434/v1",
)

response = client.invoke("What is the capital of France?")
print(response.content)

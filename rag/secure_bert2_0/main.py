import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

MODEL_NAME = "cisco-ai/SecureBERT2.0-base"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)
model.eval()

texts = [
    "CVE-2024-12345 allows remote code execution in the web server.",
    "The vulnerability enables arbitrary code execution via crafted HTTP requests.",
]

inputs = tokenizer(
    texts,
    padding=True,
    truncation=True,
    max_length=1024,
    return_tensors="pt",
)

with torch.no_grad():
    outputs = model(**inputs)  # last_hidden_state: [batch, seq_len, hidden]
    token_embeddings = outputs.last_hidden_state
    attention_mask = inputs["attention_mask"].unsqueeze(-1)

    # Mean pooling over non-padding tokens
    sentence_embeddings = (token_embeddings * attention_mask).sum(dim=1) / attention_mask.sum(dim=1)

    # Optional but usually useful for cosine similarity
    sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)

print(sentence_embeddings.shape)  # [2, hidden_size]

similarity = torch.matmul(sentence_embeddings[0], sentence_embeddings[1])
print("Cosine similarity:", similarity.item())

# SecureBERT2.0

Example of using SecureBERT2.0 for embedding generation.

Expected output:


```
Loading weights: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 134/134 [00:00<00:00, 18070.76it/s]
ModernBertModel LOAD REPORT from: cisco-ai/SecureBERT2.0-base
Key               | Status     |  | 
------------------+------------+--+-
head.norm.weight  | UNEXPECTED |  | 
head.dense.weight | UNEXPECTED |  | 
decoder.bias      | UNEXPECTED |  | 

Notes:
- UNEXPECTED:	can be ignored when loading from different task/architecture; not ok if you expect identical arch.
torch.Size([2, 768])
Cosine similarity: 0.8938697576522827
```

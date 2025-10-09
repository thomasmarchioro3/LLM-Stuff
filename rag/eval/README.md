# RAG Evaluation

Evaluation example for Retrieval-Augmented Generation (RAG) systems on the WixQA dataset.

## Setup

Install dependencies:

```sh
uv sync
```

## Run the code

Evaluation of simple TF-IDF:

```sh
uv run eval_wixqa.py --limit 200 --k 10
```

Retrieval examples:

```sh
uv run print_examples.py
```

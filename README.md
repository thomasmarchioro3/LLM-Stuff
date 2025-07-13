# LLM Stuff

Repo where I add non-trivial code for LLM applications (agents, fine-tuning) so that I can reuse it in the future.

## How to run the code

This repo is divided in small projects:
```
.
├── agents
│   └── unit_test
├── fine_tuning
│   └── dpo
└── README.md
```

Each project directory (e.g., `agents/unit_test`) contains a `requirement.txt` with the minimal dependencies to run the code. 

- Optional: Create a virtual environment
```sh
python -m venv .venv
```

- Install dependencies
```sh
pip install -r requirements.txt
```

- Follow further instructions in the `README.md`
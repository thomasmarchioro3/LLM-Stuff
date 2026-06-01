# Simple RAG - Retrieval Augmented Generation

## Requirements (recommended version)

Install the dependencies with

```bash
uv sync
```

## Usage

- Clone the repository and make sure you are using a python environment meeting the requirements above.
- Open a terminal and navigate to the root directory of the project 
- Index PDF chunks to ChromaDB by running (check the file `[database/pdf2chroma.py](database/pdf2chroma.py)` for more options):
```bash
uv run -m database.pdf2chroma
```
- Start a simple RAG-based chatbot (check the file [query.py](query.py) for more options):
```bash
uv run -m query
```

Example output:
```
Enter your question: what is the NIRS acronym short for?
================================
Context: We emphasize three of them, namely the type of response action, evaluation
methodologies, and metrics, that our proposed notion of NIRS aims to address.
1 https://www.snort.org/,https://suricata.io/
2 https://github.com/UNBCIC/CICFlowMeter,https://zeek.org/,
https://www.nfstream.org/

----

Network Intrusion Response Systems 11
Description Rule format
Block source IP or subnet-A FORWARD -s <src_ip>[/<subnet>] -j DROP
Block destination IP or subnet-A FORWARD -d <dst_ip>[/<subnet>] -j DROP
Block specific protocol for a destination IP/-
subnet
-A FORWARD -d <dst_ip>[/<subnet>] -p
<protocol> -j DROP
Block specific protocol’s port for a destination
IP/subnet
-A FORWARD -d <dst_ip>[/<subnet>] -p
<protocol> –dport <dst_port> -j DROP
T able 2.Allowediptablesrules formats.
Heuristic-based NIRS(R 1). This NIRS updates its ruleset through a simple
heuristic process that evaluates both alerts and normal traffic. At each update,
the NIRS produces a new rule of the form
-A FORWARD -s {{src_ip}} -j DROP
Thesrc_ipis selected as the most frequent IP in¯XA. However, to mitigate the
impact on benign traffic, a limit ofϵ= 0.1is set for the occurrence ofsrc_ipin
the current window of¯XN. If that limit is exceeded for the most frequent IP in

----

to be considered in NIRS evaluation is that a ruleset based on a certain NIDS
alert is put in place only after the flow responsible for that alert has ended. In
================================
Response: The NIRS acronym appears to be Network Intrusion Response Systems, according to the provided text.

Sources: ['pdf/ESORICS_2025_paper_756.pdf:3:3', 'pdf/ESORICS_2025_paper_756.pdf:11:1', 'pdf/ESORICS_2025_paper_756.pdf:7:4']
```

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

```
[info] Loaded wixqa_expertwritten: 200 examples
{
  "split": "wixqa_expertwritten",
  "n": 200,
  "retrieval": {
    "hit@1": 0.24,
    "hit@5": 0.525,
    "hit@10": 0.645,
    "mrr@10": 0.36040674603174616,
    "ndcg@10": 0.38846720350440045
  },
  "generation": {
    "exact_match": 0.0,
    "rougeL_f1": 0.13360812182885262
  }
}
[info] Loaded wixqa_simulated: 200 examples
{
  "split": "wixqa_simulated",
  "n": 200,
  "retrieval": {
    "hit@1": 0.175,
    "hit@5": 0.435,
    "hit@10": 0.535,
    "mrr@10": 0.28127380952380965,
    "ndcg@10": 0.3256524307255833
  },
  "generation": {
    "exact_match": 0.0,
    "rougeL_f1": 0.18531563378590524
  }
}

=== Summary ===
wixqa_expertwritten | hit@1 0.240 hit@5 0.525 hit@10 0.645 mrr@10 0.360 ndcg@10 0.388 | EM 0.000 ROUGE-L 0.134
   wixqa_simulated | hit@1 0.175 hit@5 0.435 hit@10 0.535 mrr@10 0.281 ndcg@10 0.326 | EM 0.000 ROUGE-L 0.185
```

Retrieval examples:

```sh
uv run print_examples.py
```

```
========================================================================================================================
[117] ❓ QUESTION:
   I am experiencing issues with fonts not displaying correctly on the mobile version of
   my website. The fonts I am using are default fonts provided by Wix.

✅ GOLD article_ids (1): ['6d5d61b21e2981893d50b1c101694cddc3c88c93e4f2b14b02143c0f19038687']
   → Wix Editor: Troubleshooting Font Issues on Your Mobile Site Differences in the w...

🔍 TOP-5 retrieved docs:
     1. ID=e28fdc7d56d7e729e11b1744cc7c3c29dd8615cdcea8bb30b9a102e99fabad44
          Site Performance: Optimizing Your Text Optimize text on your site to improve the visitor
     experience. Font types, effects, and formatting can impact your site's performance, so we r

 ⭐   2. ID=6d5d61b21e2981893d50b1c101694cddc3c88c93e4f2b14b02143c0f19038687
          Wix Editor: Troubleshooting Font Issues on Your Mobile Site Differences in the way fonts render
     on mobile devices can cause issues on your live mobile site. View the info below to

     3. ID=96b1849e43ab23a9ef45333f6688868193ffd896c2c2ed31fde7e400ff4b8707
          Wix Editor: Uploading and Using Your Own Fonts Get your message read the way it was meant to be
     by uploading and using your own fonts. You can use them in text elements and anywher

     4. ID=a34da1ab2302b61d752632efffb89310add0ce8d2d7cff602d87982cbc54cc34
          Wix Logo: Uploading Your Own Fonts to the Wix Logo Maker Upload your own fonts to the Wix Logo
     Maker to create a unique and memorable look for your brand. You can upload fonts in a

     5. ID=608c34dea52a240d3ddc3001c36d6f1723a9186c750c44fb17c8291a19122b99
          Editor X: Using Different Text Fonts Use your own fonts on Editor X to get your message across
     the way it was meant to be. If your site is in a different language, you can also upl

```

from unsloth import FastLanguageModel
import torch
from datasets import load_dataset
from trl import DPOConfig, DPOTrainer

"""
Sources:
- https://docs.unsloth.ai/get-started/unsloth-notebooks
- https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Llama3.1_(8B)-GRPO.ipynb
"""


max_seq_length = 1024 # Can increase for longer reasoning traces
lora_rank = 32 # Larger rank = smarter, but slower


MODEL_NAME = "unsloth/Llama-3.2-1B-Instruct-bnb-4bit"
# MODEL_NAME = "unsloth/Meta-Llama-3.1-8B-bnb-4bit"

max_seq_length = 2048
random_seed = 42

max_prompt_length = 256

ds = load_dataset("json", data_files="train.jsonl", split="train")

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=max_seq_length,
    load_in_4bit=True,
    dtype=None,
)

model = FastLanguageModel.get_peft_model(
    model,
    r = lora_rank, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ], # Remove QKVO if out of memory
    lora_alpha = lora_rank,
    # use_gradient_checkpointing = "unsloth", # Enable long context finetuning
    random_state = random_seed,
)


training_args = DPOConfig(
    num_train_epochs=1,
    learning_rate = 5e-6,
    adam_beta1 = 0.9,
    adam_beta2 = 0.99,
    weight_decay = 0.1,
    warmup_ratio = 0.1,
    lr_scheduler_type = "cosine",
    optim = "paged_adamw_8bit",
    logging_steps = 1,
    per_device_train_batch_size = 1,
    gradient_accumulation_steps = 4, # Increase to 4 for smoother training
    max_prompt_length = max_prompt_length,
    max_completion_length = max_seq_length - max_prompt_length,
    # num_train_epochs = 1, # Set to 1 for a full training run
    max_steps = 250,
    save_steps = 250,
    max_grad_norm = 0.1,
    report_to = "none", # Can use Weights & Biases
    output_dir = "outputs",
)

trainer = DPOTrainer(
    model=model,
    ref_model=None,
    tokenizer=tokenizer,
    train_dataset=ds,
    eval_dataset=ds,
    args=training_args
)

trainer.train()
# model.save_pretrained("my_llama3_dpo_lora")
model.save_pretrained_gguf("my_llama3_dpo_lora", tokenizer, quantization_method = "q4_k_m")
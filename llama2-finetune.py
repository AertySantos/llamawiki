from peft import PeftModel
from datetime import datetime
import transformers
from peft import LoraConfig, get_peft_model
from peft import prepare_model_for_kbit_training
import matplotlib.pyplot as plt
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from datasets import load_dataset
from accelerate import FullyShardedDataParallelPlugin, Accelerator
from torch.distributed.fsdp.fully_sharded_data_parallel import FullOptimStateDictConfig, FullStateDictConfig
import wandb
import os

# Accelerator
fsdp_plugin = FullyShardedDataParallelPlugin(
    state_dict_config=FullStateDictConfig(
        offload_to_cpu=True, rank0_only=False),
    optim_state_dict_config=FullOptimStateDictConfig(
        offload_to_cpu=True, rank0_only=False),
)

accelerator = Accelerator(fsdp_plugin=fsdp_plugin)

# Load Dataset
train_dataset = load_dataset(
    'json', data_files='json/treino.json', split='train')
eval_dataset = load_dataset(
    'json', data_files='json/teste.json', split='train')

# Formatting prompts


def formatting_func(example):
    text = f"### Instruction: Utilize a Question fornecida para gerar uma Answer sucinta em português, que consiste em uma Answer fundamentada apartir de todo o seu conhecimento sobre a Question, caso nao saiba a resposta, nao gere novas perguntas e responda: nao sei a resposta \n### Question: {example['input']}\n ### Answer: {example['output']}"
    return text


# load Llama 2 7B - `meta-llama/Llama-2-7b-hf` - using 4-bit quantization!
base_model_id = "NousResearch/Llama-2-13b-chat-hf"
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,

)

model = AutoModelForCausalLM.from_pretrained(
    base_model_id, quantization_config=bnb_config)

# Tokenization
tokenizer = AutoTokenizer.from_pretrained(
    base_model_id,
    padding_side="left",
    add_eos_token=True,
    add_bos_token=True,
)
tokenizer.pad_token = tokenizer.eos_token

# Reformat the prompt and tokenize each sample:


def generate_and_tokenize_prompt(prompt):
    return tokenizer(formatting_func(prompt))


tokenized_train_dataset = train_dataset.map(generate_and_tokenize_prompt)
tokenized_val_dataset = eval_dataset.map(generate_and_tokenize_prompt)

max_length = 512


def generate_and_tokenize_prompt2(prompt):
    result = tokenizer(
        formatting_func(prompt),
        truncation=True,
        max_length=max_length,
        padding="max_length",
    )
    result["labels"] = result["input_ids"].copy()
    return result


tokenized_train_dataset = train_dataset.map(generate_and_tokenize_prompt2)
tokenized_val_dataset = eval_dataset.map(generate_and_tokenize_prompt2)

print(tokenized_train_dataset[1]['input_ids'])


# Set Up LoRA
model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )


print(model)

config = LoraConfig(
    r=32,
    lora_alpha=64,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
        "lm_head",
    ],
    bias="none",
    lora_dropout=0.05,  # Conventional
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, config)
print_trainable_parameters(model)
# Apply the accelerator. You can comment this out to remove the accelerator.
model = accelerator.prepare_model(model)

wandb.login()
wandb_project = "teste-finetune"
if len(wandb_project) > 0:
    os.environ["WANDB_PROJECT"] = wandb_project

print(model)

if torch.cuda.device_count() > 1:  # If more than 1 GPU
    model.is_parallelizable = True
    model.model_parallel = True


project = "teste-finetune"
base_model_name = "llama2-13b"
run_name = base_model_name + "-" + project
output_dir = "./" + run_name

tokenizer.pad_token = tokenizer.eos_token

trainer = transformers.Trainer(
    model=model,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_val_dataset,
    args=transformers.TrainingArguments(

        output_dir=output_dir,
        warmup_steps=1,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=1,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={'use_reentrant': False},
        max_steps=1000,
        learning_rate=1e-5,  # ou um valor ainda menor, dependendo do caso
        bf16=True,
        optim="adafactor",
        logging_dir="./logs",        # Directory for storing logs
        save_strategy="steps",       # Save the model checkpoint every logging step
        save_steps=100,                # Save checkpoints every 50 steps
        evaluation_strategy="steps",  # Evaluate the model every logging step
        eval_steps=100,               # Evaluate and save checkpoints every 50 steps
        do_eval=True,  # Perform evaluation at the end of training
        lr_scheduler_type="cosine",
        # report_to="wandb",           # Comment this out if you don't want to use weights & baises
        # Name of the W&B run (optional)
        run_name=f"{run_name}-{datetime.now().strftime('%Y-%m-%d-%H-%M')}"
    ),
    data_collator=transformers.DataCollatorForLanguageModeling(
        tokenizer, mlm=False),
)

# silence the warnings. Please re-enable for inference!
model.config.use_cache = False
trainer.train()


base_model_id = "NousResearch/Llama-2-13b-chat-hf"

base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    quantization_config=bnb_config,  # Same quantization config as before
    device_map="auto",
    trust_remote_code=True,
    use_auth_token=True
)

tokenizer = AutoTokenizer.from_pretrained(
    base_model_id, add_bos_token=True, trust_remote_code=True)

ft_model = PeftModel.from_pretrained(
    base_model, "llama2-13b-teste-finetune/checkpoint-1000")

eval_prompt = " Qual a capital do Brasil?: # "
model_input = tokenizer(eval_prompt, return_tensors="pt").to("cuda")

ft_model.eval()
with torch.no_grad():
    print(tokenizer.decode(ft_model.generate(**model_input,
          max_new_tokens=500)[0], skip_special_tokens=True))

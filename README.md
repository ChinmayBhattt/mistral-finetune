# Fine-tuning Mistral on your own data 🤙

Welcome!

In this notebook and tutorial, we will fine-tune the [Mistral 7B](https://github.com/mistralai/mistral-src) model — which outperforms Llama 2 13B on all tested benchmarks — ***on your own data!***

## Watch the accompanying video walk-through [here](https://youtu.be/kmkcNVvEz-k?si=Ogt1wRFNqYI6zXfw&t=1)!

This was done for **just one dollar ($1)** on an 1x A10G 24GB from Brev.

This tutorial uses **QLoRA**, a fine-tuning method that combines quantization and LoRA. For more information, see [this post](https://developer.nvidia.com/brev/blog/how-qlora-works).

In this notebook, we load the large model in 4-bit using `bitsandbytes` and use LoRA to train using the PEFT library from Hugging Face 🤗.

> **Note:** If you ever have trouble importing something from Huggingface, you may need to run `huggingface-cli login` in a shell.

---

## ⚠️ A Note on OOM Errors

If you get an error like `OutOfMemoryError: CUDA out of memory`, tweak your parameters to make the model less computationally intensive.

To retry, open a Terminal and run `nvidia-smi`. Find the process ID (`PID`) under `Processes` and run `kill [PID]`. Then restart the notebook from the beginning.

---

## Let's Begin!

### 0. Preparing Your Data

Before checking out a GPU, prepare your dataset for loading and training.

You need two `.jsonl` files structured like this:

```json
{"input": "What color is the sky?", "output": "The sky is blue."}
{"input": "Where is the best place to get cloud GPUs?", "output": "Brev"}
```

For input/output pairs, use the second `formatting_func` shown below, which combines all features into one input string.

The data used in this example was formatted as journal entries:

```json
{"note": "journal-entry-for-model-to-predict"}
{"note": "journal-entry-for-model-to-predict-1"}
{"note": "journal-entry-for-model-to-predict-2"}
```

---

### 1. Instantiate GPU & Load Dataset

```python
# You only need to run this once per machine
!pip install -q -U bitsandbytes
!pip install -q -U git+https://github.com/huggingface/transformers.git
!pip install -q -U git+https://github.com/huggingface/peft.git
!pip install -q -U git+https://github.com/huggingface/accelerate.git
!pip install -q -U datasets scipy ipywidgets matplotlib
```

```python
from datasets import load_dataset

train_dataset = load_dataset('json', data_files='notes.jsonl', split='train')
eval_dataset = load_dataset('json', data_files='notes_validation.jsonl', split='train')
```

#### Optional: Track Training with Weights & Biases

```python
!pip install -q wandb -U

import wandb, os
wandb.login()

wandb_project = "journal-finetune"
if len(wandb_project) > 0:
    os.environ["WANDB_PROJECT"] = wandb_project
```

---

### Formatting Prompts

Create a `formatting_func` to structure training examples as prompts:

```python
def formatting_func(example):
    text = f"### The following is a note by Eevee the Dog: {example['note']}"
    return text
```

Another common format for input/output pairs:

```python
def formatting_func(example):
    text = f"### Question: {example['input']}\n ### Answer: {example['output']}"
    return text
```

---

### 2. Load Base Model

Load Mistral (`mistralai/Mistral-7B-v0.1`) using **4-bit quantization**:

```python
from huggingface_hub import notebook_login
notebook_login()
```

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

base_model_id = "mistralai/Mistral-7B-v0.1"
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

model = AutoModelForCausalLM.from_pretrained(base_model_id, quantization_config=bnb_config, device_map="auto")
```

---

### 3. Tokenization

Set up the tokenizer with **left padding** (reduces memory usage during training):

```python
tokenizer = AutoTokenizer.from_pretrained(
    base_model_id,
    padding_side="left",
    add_eos_token=True,
    add_bos_token=True,
)
tokenizer.pad_token = tokenizer.eos_token

def generate_and_tokenize_prompt(prompt):
    return tokenizer(formatting_func(prompt))
```

Tokenize each sample:

```python
tokenized_train_dataset = train_dataset.map(generate_and_tokenize_prompt)
tokenized_val_dataset = eval_dataset.map(generate_and_tokenize_prompt)
```

#### Visualize Data Lengths

```python
import matplotlib.pyplot as plt

def plot_data_lengths(tokenized_train_dataset, tokenized_val_dataset):
    lengths = [len(x['input_ids']) for x in tokenized_train_dataset]
    lengths += [len(x['input_ids']) for x in tokenized_val_dataset]
    print(len(lengths))

    plt.figure(figsize=(10, 6))
    plt.hist(lengths, bins=20, alpha=0.7, color='blue')
    plt.xlabel('Length of input_ids')
    plt.ylabel('Frequency')
    plt.title('Distribution of Lengths of input_ids')
    plt.show()

plot_data_lengths(tokenized_train_dataset, tokenized_val_dataset)
```

#### Tokenize with Padding & Truncation

Choose a `max_length` based on your data distribution, then re-tokenize:

```python
max_length = 512  # Adjust based on your dataset

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
```

> **Verify:** `input_ids` should be left-padded with `eos_token` (2), end with `eos_token` (2), and start with `bos_token` (1).

```python
print(tokenized_train_dataset[1]['input_ids'])
```

---

### How Does the Base Model Do?

Test the base model before fine-tuning:

```python
eval_prompt = " The following is a note by Eevee the Dog: # "

eval_tokenizer = AutoTokenizer.from_pretrained(base_model_id, add_bos_token=True)
model_input = eval_tokenizer(eval_prompt, return_tensors="pt").to("cuda")

model.eval()
with torch.no_grad():
    print(eval_tokenizer.decode(
        model.generate(**model_input, max_new_tokens=256, repetition_penalty=1.15)[0],
        skip_special_tokens=True
    ))
```

---

### 4. Set Up LoRA

Prepare the model for k-bit training:

```python
from peft import prepare_model_for_kbit_training

model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)
```

Helper to inspect trainable parameters:

```python
def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )
```

Configure LoRA and apply it to the model. QLoRA is applied to all linear layers: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`, and `lm_head`.

> **Note:** `r=32` and `lora_alpha=64` are used here. The QLoRA paper used `r=64` and `lora_alpha=16`.

```python
from peft import LoraConfig, get_peft_model

config = LoraConfig(
    r=32,
    lora_alpha=64,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj", "lm_head",
    ],
    bias="none",
    lora_dropout=0.05,
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, config)
print_trainable_parameters(model)
# trainable params: 85041152 || all params: 3837112320 || trainable%: 2.22
```

---

### 5. Run Training!

```python
if torch.cuda.device_count() > 1:
    model.is_parallelizable = True
    model.model_parallel = True
```

```python
import transformers
from datetime import datetime

project = "journal-finetune"
base_model_name = "mistral"
run_name = base_model_name + "-" + project
output_dir = "./" + run_name

trainer = transformers.Trainer(
    model=model,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_val_dataset,
    args=transformers.TrainingArguments(
        output_dir=output_dir,
        warmup_steps=2,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=1,
        gradient_checkpointing=True,
        max_steps=500,
        learning_rate=2.5e-5,
        bf16=True,
        optim="paged_adamw_8bit",
        logging_steps=25,
        logging_dir="./logs",
        save_strategy="steps",
        save_steps=25,
        evaluation_strategy="steps",
        eval_steps=25,
        do_eval=True,
        report_to="wandb",
        run_name=f"{run_name}-{datetime.now().strftime('%Y-%m-%d-%H-%M')}"
    ),
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
)

model.config.use_cache = False  # Silence warnings. Re-enable for inference!
trainer.train()
```

> **Tip on Overfitting:** Start with a high `max_steps` (e.g. 1000) and watch the validation loss. When it starts rising while training loss drops, that's your sweet spot. Use the corresponding checkpoint as your final model.

---

### 6. Try the Trained Model!

> **Important:** Restart the kernel before loading the model again to avoid running out of memory.

Load the base model:

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

base_model_id = "mistralai/Mistral-7B-v0.1"
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)

eval_tokenizer = AutoTokenizer.from_pretrained(base_model_id, add_bos_token=True, trust_remote_code=True)
```

Load the QLoRA adapter from the best checkpoint:

```python
from peft import PeftModel

ft_model = PeftModel.from_pretrained(base_model, "mistral-journal-finetune/checkpoint-300")
```

Run inference:

```python
eval_prompt = " The following is a note by Eevee the Dog, which doesn't share anything too personal: # "
model_input = eval_tokenizer(eval_prompt, return_tensors="pt").to("cuda")

ft_model.eval()
with torch.no_grad():
    print(eval_tokenizer.decode(
        ft_model.generate(**model_input, max_new_tokens=100, repetition_penalty=1.15)[0],
        skip_special_tokens=True
    ))
```

**Example output:**
```
The following is a note by Eevee the Dog, which doesn't share anything too personal: #
I'm grateful for my best friend coming to visit me. I know we'll have so much fun and our
relationship will continue to flourish. We really are each other's number one fan...
```


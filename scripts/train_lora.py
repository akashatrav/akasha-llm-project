import json
import torch
from transformers import LlamaTokenizer, LlamaForCausalLM, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model

# Define your model path.
# This should be the directory of the base DeepSeek model.
model_name_or_path = "deepseek-ai/DeepSeek-V3-Base"

# Load tokenizer and base model.
tokenizer = LlamaTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
model = LlamaForCausalLM.from_pretrained(model_name_or_path, device_map="auto", trust_remote_code=True)


# Set up the LoRA configuration.
lora_config = LoraConfig(
    r=8,                     # Rank of LoRA
    lora_alpha=32,           # Scaling factor
    target_modules=["q_proj", "v_proj"],  # Example target modules; adjust if needed
    lora_dropout=0.1,        # Dropout rate
    bias="none",
)

# Apply LoRA to the model.
model = get_peft_model(model, lora_config)

# Load training data from a JSON file.
# Make sure the file 'sample_training_data.json' exists in the same directory.
with open("sample_training_data.json", "r") as f:
    training_data = json.load(f)

# Create a simple dataset.
class SimpleDataset(torch.utils.data.Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        # Concatenate instruction and output as an example.
        text = item["instruction"] + " " + item["output"]
        encoding = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=128, padding="max_length")
        # Remove batch dimension.
        encoding = {k: v.squeeze(0) for k, v in encoding.items()}
        return encoding

dataset = SimpleDataset(training_data, tokenizer)

# Set up training arguments.
training_args = TrainingArguments(
    output_dir="./lora_adapter",   # Directory to save adapter weights.
    num_train_epochs=1,            # One epoch for testing.
    per_device_train_batch_size=1, # Small batch size.
    logging_steps=1,
    save_steps=100,
    save_total_limit=1,
    prediction_loss_only=True,
    learning_rate=2e-4,
    fp16=True,                     # Enable if your GPU supports it.
)

# Initialize the Trainer.
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
)

# Start training.
print("Starting LoRA training...")
trainer.train()
print("Training complete.")

# Save the LoRA adapter.
model.save_pretrained("./lora_adapter")
print("LoRA adapter saved in ./lora_adapter")

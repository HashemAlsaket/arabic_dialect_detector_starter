from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset, DatasetDict
import pandas as pd

# Load and process the ADI dataset
# Replace 'path_to_adi_dataset.csv' with the actual path to your ADI CSV file
training_data_path = "/home/hashem-alsaket/Desktop/workspace/arabic_dialect_detector_starter/src/data/Arabic_dialect.csv"
dataset = pd.read_csv(training_data_path)
dataset = DatasetDict({
    "train": load_dataset("csv", data_files=training_data_path)["train"],
    "validation": load_dataset("csv", data_files=training_data_path)["train"],
    "test": load_dataset("csv", data_files=training_data_path)["train"]
})

# Load the LLaMA tokenizer and model for sequence classification
model_name = "meta-llama/Llama-3.1-8B"  # Replace with your checkpoint if using a fine-tuned LLaMA model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=5)  # Update `num_labels` if needed

# Set the pad token if it's missing
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token 

# Tokenization function
def preprocess_function(examples):
    return tokenizer(examples["t"], padding="max_length", truncation=True)

# Tokenize the dataset
tokenized_datasets = dataset.map(preprocess_function, batched=True)

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=2,  # Adjust based on your GPU memory
    per_device_eval_batch_size=2,
    num_train_epochs=3,
    weight_decay=0.01,
    fp16=True,  # Use mixed precision for faster training if supported
)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
)

# Fine-tune the model
trainer.train()

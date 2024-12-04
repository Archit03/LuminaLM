import os
import torch
import torch.nn as nn
import torch.optim as optim
import logging
import yaml
from torch.utils.data import DataLoader, ConcatDataset, random_split
from datasets import load_dataset, DatasetDict
from transformers import DataCollatorWithPadding, get_linear_schedule_with_warmup
from tokenizers import Tokenizer
from tqdm import tqdm
from model import LuminaLM, LuminaLMConfig
from torch.utils.tensorboard import SummaryWriter
import argparse

# Argument Parser for YAML Config File
parser = argparse.ArgumentParser()
parser.add_argument("--config_path", type=str, required=True, help="Path to configuration YAML file.")
args = parser.parse_args()

# Load YAML Configurations
with open(args.config_path, 'r') as config_file:
    config_data = yaml.safe_load(config_file)

# Model Configuration from YAML
model_config = LuminaLMConfig(**config_data['model'])

# Training Parameters
training_config = config_data['training']

# Logging Configuration
logging_config = config_data['logging']

# Set up logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Device Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# TensorBoard SummaryWriter
writer = SummaryWriter(log_dir=logging_config['tensorboard_log_dir'])

# Dataset Dictionary: Include Different Datasets and Splits
dataset_dict = {
    "squad": ["train", "validation"],
    "trivia_qa": ["train"],
    "nq_open": ["train"]
}

# Load Tokenizer
tokenizer = Tokenizer.from_file("Medical_tokenizer.json")

# Preprocessing Function for QA Datasets
def preprocess_function(examples):
    input_texts = ["question: " + q + " context: " + c for q, c in zip(examples["question"], examples["context"])]
    inputs = [tokenizer.encode(text).ids[:model_config.block_size] for text in input_texts]
    targets = [tokenizer.encode(answer["text"][0]).ids[:model_config.block_size] if len(answer["text"]) > 0 else [model_config.pad_token_id] for answer in examples["answers"]]
    return {"input_ids": inputs, "decoder_input_ids": targets}

# Load and Preprocess Datasets
datasets = []

for dataset_name, splits in dataset_dict.items():
    for split in splits:
        try:
            dataset = load_dataset(dataset_name, split=split)
            tokenized_dataset = dataset.map(preprocess_function, batched=True)
            datasets.append(tokenized_dataset)
            logger.info(f"Loaded and tokenized dataset '{dataset_name}' split '{split}'")
        except Exception as e:
            logger.error(f"Error loading dataset '{dataset_name}' split '{split}': {e}")

# Combine All Loaded Datasets
combined_dataset = ConcatDataset(datasets)

# Split Combined Dataset into Training and Validation Sets
train_size = int(0.9 * len(combined_dataset))
val_size = len(combined_dataset) - train_size
train_dataset, val_dataset = random_split(combined_dataset, [train_size, val_size])

# DataLoader with Padding Collator
data_collator = DataCollatorWithPadding(tokenizer=tokenizer, pad_to_multiple_of=model_config.block_size)
train_loader = DataLoader(train_dataset, batch_size=training_config['batch_size'], collate_fn=data_collator, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=training_config['batch_size'], collate_fn=data_collator)

# Load Model
model = LuminaLM(model_config).to(device)

# Load Pre-trained Embeddings
embedding_path = "luminalm_embeddings_v1.pt"
if os.path.exists(embedding_path):
    try:
        embeddings = torch.load(embedding_path)
        model.get_input_embeddings().weight.data.copy_(embeddings)
        logger.info("Loaded pre-trained embeddings from {}".format(embedding_path))
    except Exception as e:
        logger.error(f"Error loading embeddings: {e}")

# Multi-GPU Support
if torch.cuda.device_count() > 1:
    logger.info(f"Using {torch.cuda.device_count()} GPUs for training.")
    model = nn.DataParallel(model)

# Optimizer, Scheduler, and Early Stopping
optimizer = optim.AdamW(model.parameters(), lr=training_config['learning_rate'], weight_decay=training_config['weight_decay'])
num_training_steps = len(train_loader) * training_config['num_epochs']
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=training_config['warmup_steps'], num_training_steps=num_training_steps)

# Loss Function
criterion = nn.CrossEntropyLoss(ignore_index=model_config.pad_token_id)

# Early Stopping Mechanism
class EarlyStopping:
    def __init__(self, patience=training_config['early_stopping_patience'], min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

early_stopping = EarlyStopping()

# Training Loop
def train_one_epoch(model, train_loader, optimizer, scheduler, criterion, device, epoch: int, max_grad_norm: float = 1.0):
    model.train()
    total_loss = 0
    for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Training Epoch {epoch+1}")):
        input_ids = torch.tensor(batch["input_ids"]).to(device)
        labels = torch.tensor(batch["decoder_input_ids"]).to(device)

        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, decoder_input_ids=input_ids)
        logits = outputs.view(-1, outputs.size(-1))

        loss = criterion(logits, labels.view(-1))
        loss.backward()

        # Gradient clipping for stability
        if max_grad_norm:
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

        # Log the training loss
        if batch_idx % 10 == 0:
            writer.add_scalar('Training Loss', loss.item(), epoch * len(train_loader) + batch_idx)

    avg_loss = total_loss / len(train_loader)
    logger.info(f"Training Loss after Epoch {epoch+1}: {avg_loss}")
    writer.add_scalar('Average Training Loss per Epoch', avg_loss, epoch)
    return avg_loss

# Validation Loop
def validate_one_epoch(model, val_loader, criterion, device, epoch: int):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in tqdm(val_loader, desc=f"Validating Epoch {epoch+1}"):
            input_ids = torch.tensor(batch["input_ids"]).to(device)
            labels = torch.tensor(batch["decoder_input_ids"]).to(device)

            outputs = model(input_ids=input_ids, decoder_input_ids=input_ids)
            logits = outputs.view(-1, outputs.size(-1))

            loss = criterion(logits, labels.view(-1))
            total_loss += loss.item()

    avg_loss = total_loss / len(val_loader)
    logger.info(f"Validation Loss after Epoch {epoch+1}: {avg_loss}")
    writer.add_scalar('Validation Loss', avg_loss, epoch)
    return avg_loss

# Generate Function - Called after Each Epoch
def generate_response(model, prompt: str, max_length: int = 50):
    model.eval()
    with torch.no_grad():
        try:
            input_ids = tokenizer.encode(prompt).ids
            input_ids = torch.tensor([input_ids], device=device)

            # Generate response using the model
            generated_ids = model.generate(input_ids=input_ids, max_length=max_length)
            generated_tokens = generated_ids[0].cpu().tolist()

            # Decode response
            response = tokenizer.decode(generated_tokens)
            logger.info(f"Prompt: {prompt}\nResponse: {response}")
            return response
        except Exception as e:
            logger.error(f"Error during generation: {e}")
            return "Error generating response."

# Training and Validation Process
num_epochs = training_config['num_epochs']
best_val_loss = float('inf')

for epoch in range(num_epochs):
    # Train and Validate
    try:
        train_loss = train_one_epoch(model, train_loader, optimizer, scheduler, criterion, device, epoch, max_grad_norm=training_config['max_grad_norm'])
        val_loss = validate_one_epoch(model, val_loader, criterion, device, epoch)

        # Early Stopping
        early_stopping(val_loss)
        if early_stopping.early_stop:
            logger.info("Early stopping triggered.")
            break

        # Save model if validation improves
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model.save_pretrained("best_model")
            logger.info("Saved Best Model with Validation Loss: {:.4f}".format(best_val_loss))

        # Generate Response after Epoch
        prompt = "What are the symptoms of diabetes?"
        generate_response(model, prompt)

    except Exception as e:
        logger.error(f"An error occurred during training at epoch {epoch+1}: {e}")
        raise

# Save Final Model
model.save_pretrained("final_model")
logger.info("Final model saved.")

import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AdamW
import embeddings  # Custom embeddings logic in embeddings.py
import tokenizer  # Custom tokenizer logic in tokenizer.py
from model import SentientSculptor, SentientSculptorConfig, TextDataset  # Assuming model is in sentient_sculptor.py


# Function to measure memory usage
def get_memory_usage(device):
    if device.type == 'cuda':
        return torch.cuda.memory_allocated(device) / (1024 ** 3)  # Convert to GB
    return 0.0  # CPU doesn't track memory usage in the same way


# Function to benchmark training
def benchmark_training(model, dataloader, device, epochs=3):
    optimizer = AdamW(model.parameters(), lr=1e-4)
    loss_fn = nn.CrossEntropyLoss()
    model.train()

    print(f"Benchmarking Training on {device}")

    total_training_time = 0
    for epoch in range(epochs):
        start_time = time.time()
        loop = tqdm(dataloader, leave=True)
        total_tokens = 0

        for input_ids, target_ids in loop:
            optimizer.zero_grad()
            
            input_ids, target_ids = input_ids.to(device), target_ids.to(device)

            # Forward pass
            outputs = model(input_ids)

            # Shift logits and labels for next-token prediction
            logits = outputs[..., :-1, :].contiguous()
            shift_labels = target_ids[..., 1:].contiguous()

            # Calculate loss
            loss = loss_fn(logits.view(-1, logits.size(-1)), shift_labels.view(-1))

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            total_tokens += input_ids.numel()  # Count number of tokens processed
            loop.set_postfix(loss=loss.item())

        end_time = time.time()
        epoch_time = end_time - start_time
        total_training_time += epoch_time

        print(f"Epoch {epoch+1} Time: {epoch_time:.2f}s, Memory Usage: {get_memory_usage(device):.2f} GB")

    avg_epoch_time = total_training_time / epochs
    print(f"Average Training Time per Epoch: {avg_epoch_time:.2f}s")
    return avg_epoch_time


# Function to benchmark inference (text generation)
def benchmark_inference(model, device, prompt, max_length=50, num_trials=10):
    model.eval()
    generation_times = []

    for _ in range(num_trials):
        start_time = time.time()
        
        with torch.no_grad():
            generated_text = model.generate_text(prompt, max_length=max_length)

        end_time = time.time()
        generation_time = end_time - start_time
        generation_times.append(generation_time)

    avg_generation_time = sum(generation_times) / num_trials
    print(f"Average Inference Time (over {num_trials} trials): {avg_generation_time:.2f}s")
    return avg_generation_time


# Function to benchmark total memory usage
def benchmark_memory_usage(model, device):
    model.eval()
    
    # Run a dummy forward pass to calculate memory usage
    input_ids = torch.randint(0, model.config.vocab_size, (1, model.config.block_size)).to(device)
    
    with torch.no_grad():
        model(input_ids)
    
    memory_usage = get_memory_usage(device)
    print(f"Peak Memory Usage: {memory_usage:.2f} GB")
    return memory_usage


# Main benchmarking function
if __name__ == "__main__":
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize model and load it to device
    model_config = SentientSculptorConfig()
    model = SentientSculptor(model_config)
    model.to(device)

    # Prepare dataset and dataloader
    texts = ["The medical diagnosis reveals a pattern.", "The MRI scan shows abnormal growth."]
    dataset = TextDataset(texts, model.tokenizer)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    # Benchmark training
    avg_epoch_time = benchmark_training(model, dataloader, device, epochs=3)

    # Benchmark inference
    prompt = "The patient exhibits"
    avg_inference_time = benchmark_inference(model, device, prompt, max_length=50, num_trials=10)

    # Benchmark memory usage
    peak_memory_usage = benchmark_memory_usage(model, device)

    # Print summary of benchmark results
    print("\nBenchmark Summary:")
    print(f"Average Training Time per Epoch: {avg_epoch_time:.2f} seconds")
    print(f"Average Inference Time (50 tokens): {avg_inference_time:.2f} seconds")
    print(f"Peak Memory Usage during Inference: {peak_memory_usage:.2f} GB")

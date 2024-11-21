import model as model
# Function to count the number of parameters
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Example usage with Sentient Sculptor model
model_config = model.LuminaLMConfig()  # Initialize a smaller custom GPT configuration
model = model.LuminaLM(model_config)  # Randomly initialized Sentient Sculptor model

total_params = count_parameters(model)
print(f'Total number of trainable parameters: {total_params}')

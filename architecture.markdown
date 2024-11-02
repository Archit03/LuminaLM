# Architecture of LuminaLM-3B Model
# [EllanorAI](https://www.ellanorai.org/)

```mermaid
flowchart TD
    %% Title
    title["Architecture of the Model"]

    %% Define nodes and flow
    A["Input Tokenization"] --> B["Token Embedding (wte)"]
    B --> C["Positional Embedding (wpe)"]
    C --> D["Transformer Layers"]

    %% Expand Transformer Layers section
    subgraph Transformer_Layers
        D1["Multi-Head Self-Attention"] --> D2["Feedforward Network"]
        D2 --> D3["Layer Normalization"]
        D3 --> D4["Dropout & Residual Connection"]
    end

    D --> E["Final Layer Normalization"]
    E --> F["Output Projection to Vocabulary"]
    F --> G["Predicted Tokens / Text Output"]

    %% Optional Notes
    note1["Each Transformer Layer performs self-attention and transformation."]
    note2["Output layer maps hidden states back to vocabulary for predictions."]
    D --> note1
    F --> note2


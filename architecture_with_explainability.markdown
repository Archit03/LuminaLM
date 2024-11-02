```mermaid
flowchart TD
    A[Input Tokenization] --> B[Token Embedding Layer]
    B --> C[Rotary Positional Embeddings]
    C --> D[Stacked Transformer Layers with Explainability Modules]

    %% Transformer Layer with Explainability Modules
    subgraph Transformer_Layers
        D1[Multi-Head Self-Attention] --> D1A[Attention Visualization]
        D1 --> D2[Feedforward Network]
        D2 --> D3[Feature Attribution - Integrated Gradients & SHAP]
        D3 --> D4[Layer-wise Relevance Propagation]
        D4 --> D5[Layer Normalization and Dropout]
        D1A --> D3
        D3 --> D5
    end

    D --> E[Final Layer Normalization]
    E --> F[Output Projection to Vocabulary]
    F --> G[Generated Tokens or Text Output with Explanations]

    %% Explainability Dashboard and Feedback
    G --> H[Explainability Dashboard with Heatmaps]
    H --> I[User Feedback for Explanation Tuning]

    %% Explanation Details
    note1["Attention Visualization: Highlights focus on specific tokens in each layer."]
    note2["Feature Attribution: Integrated Gradients & SHAP interpret important tokens."]
    note3["Layer-wise Relevance Propagation: Shows relevance across layers."]
    note4["Dashboard: Interactive visualizations for analysis and feedback."]

    %% Link notes to relevant sections
    D1A --> note1
    D3 --> note2
    D4 --> note3
    H --> note4


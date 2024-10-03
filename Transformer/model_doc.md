# Transformer Model: Detailed documentation

This code defines a Transformer model using PyTorch, including both the encoder and decoder components, positional embeddings, multi-head attention, feed-forward networks, and more. It is designed for sequence-to-sequence tasks such as machine translation or text generation. Below is a detailed explanation of the code:

---

## 1. `InputEmbeddings` Class
- This class is used to transform input token indices into dense vectors (embeddings).
- **`nn.Embedding(vocab_size, d_model)`**: Creates an embedding layer where `vocab_size` is the size of the vocabulary and `d_model` is the dimensionality of each embedding vector.
- **`forward(self, x)`**: For the given input `x` (a tensor of token indices), it generates embeddings and scales them by `sqrt(d_model)` for normalization (as suggested in the original Transformer paper).

---

## 2. `PositionalEncoding` Class
- Positional encoding is used to give the model information about the position of tokens in the sequence (since the Transformer architecture lacks inherent sequential awareness like RNNs).
- **`pe = torch.zeros(seq_len, d_model)`**: Creates a matrix to store positional encodings for each position in the sequence.
- **`position` and `div_terms`**: These are used to calculate the sinusoidal positional encodings, where `sin` is applied to even positions and `cos` to odd positions.
- **`register_buffer('pe', pe)`**: Stores the positional encodings as a buffer (non-learnable parameter).
- **`forward(self, x)`**: Adds positional encodings to the input embeddings and applies dropout for regularization.

---

## 3. `LayerNormalization` Class
- This is a custom implementation of layer normalization, which helps stabilize and speed up training by normalizing the inputs across each token's hidden dimension.
- **`alpha` and `bias`**: Learnable parameters used to scale and shift the normalized output.
- **`forward(self, x)`**: Normalizes the input `x` by subtracting the mean and dividing by the standard deviation, then scales by `alpha` and shifts by `bias`.

---

## 4. `FeedForwardBlock` Class
- This is a fully connected feed-forward network that operates on each token independently.
- **`nn.Linear(d_model, d_ff)`**: The first linear layer projects the `d_model`-dimensional input to a higher dimension `d_ff` (usually much larger).
- **`torch.relu`**: Applies the ReLU activation function.
- **`forward(self, x)`**: Passes input through the two linear layers, applying ReLU and dropout in between.

---

## 5. `MultiHeadAttentionBlock` Class
- Multi-head attention allows the model to focus on different parts of the sequence simultaneously.
- **`w_q`, `w_k`, `w_v`**: Linear layers to generate query, key, and value matrices.
- **`attention(query, key, value, mask)`**: The static method implements scaled dot-product attention:
  - Calculates the attention scores between the query and key vectors.
  - Applies a softmax to get attention probabilities.
  - Uses the attention scores to combine the value vectors.
- **`forward(self, q, k, v, mask)`**: Splits the input into multiple heads, applies the attention mechanism, and concatenates the outputs.

---

## 6. `ResidualConnection` Class
- Implements the residual connections with layer normalization, which are essential for training deep models by allowing the gradient to bypass some layers.
- **`forward(self, x, sublayer)`**: Applies the sublayer (such as attention or feed-forward) to the normalized input, adds the residual connection, and applies dropout.

---

## 7. `EncoderBlock` Class
- Defines one block of the encoder, which consists of:
  - A self-attention mechanism.
  - A feed-forward network.
  - Two residual connections.
- **`forward(self, x, src_mask)`**: Passes the input through self-attention and feed-forward layers, with residual connections around each.

---

## 8. `Encoder` Class
- Combines multiple `EncoderBlock`s to create the full encoder.
- **`layers`**: A `ModuleList` containing multiple encoder layers.
- **`forward(self, x, mask)`**: Passes the input sequentially through all the encoder layers and applies final layer normalization.

---

## 9. `DecoderBlock` Class
- Defines one block of the decoder, consisting of:
  - A self-attention mechanism.
  - A cross-attention mechanism (which attends to the encoder output).
  - A feed-forward network.
  - Three residual connections.
- **`forward(self, x, encoder_output, src_mask, tgt_mask)`**: Passes the input through self-attention, cross-attention (to the encoder), and feed-forward layers.

---

## 10. `Decoder` Class
- Combines multiple `DecoderBlock`s to create the full decoder.
- **`forward(self, x, encoder_output, src_mask, tgt_mask)`**: Passes the input through all the decoder layers, attending to both the encoder output and the target input.

---

## 11. `ProjectionLayer` Class
- This layer projects the final decoder output (which is of dimension `d_model`) to the vocabulary size for generating predictions.
- **`forward(self, x)`**: Applies a linear transformation followed by a log softmax to get the final probabilities over the vocabulary.

---

## 12. `Transformer` Class
- The main Transformer model that ties together the encoder, decoder, and embedding components.
- **`encode(self, src, src_mask)`**: Encodes the source sequence.
- **`decode(self, encoder_output, src_mask, tgt, tgt_mask)`**: Decodes the target sequence using the encoder's output.
- **`project(self, x)`**: Projects the decoder's output into the target vocabulary space.

---

## 13. `build_transformer` Function
- This function builds a complete Transformer model with customizable parameters.
- Parameters:
  - `src_vocab_size` and `tgt_vocab_size`: Vocabulary sizes for the source and target languages.
  - `src_seq_len` and `tgt_seq_len`: Maximum sequence lengths for source and target sequences.
  - `d_model`: Dimensionality of embeddings.
  - `N`: Number of layers in the encoder and decoder.
  - `h`: Number of attention heads.
  - `dropout`: Dropout probability.
  - `d_ff`: Dimensionality of the feed-forward network.
- **`return transformer`**: Returns a fully initialized Transformer model.

---

## Explanation of Training Process (not explicitly in the code):
- **Input**: The source sentence is tokenized and embedded, then passed through the encoder.
- **Encoding**: The encoder processes the source sequence using multi-head self-attention and feed-forward layers.
- **Decoding**: The decoder generates the target sequence one token at a time, using both self-attention and cross-attention to the encoder output.
- **Projection**: The decoder's output is projected to the vocabulary space, and the most likely tokens are selected.

---

## Initialization:
- The weights of the transformer model are initialized using Xavier uniform initialization (`nn.init.xavier_uniform_`), which helps with stable training.

---

## Conclusion:
This implementation follows the architecture described in the original Transformer paper: *"Attention is All You Need"*. It includes key components such as multi-head attention, residual connections, layer normalization, and position-wise feed-forward networks. The overall design is modular and allows customization of various aspects like the number of layers, attention heads, and the size of the feed-forward network.

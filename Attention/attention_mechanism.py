import numpy as np

def attention_mechanism(embeddings, W_Q, W_K, W_V, is_decoder=False) -> np.ndarray:
    """
    Compute the attention mechanism for a sequence of word embeddings.

    Parameters:
    - embeddings: numpy array of shape (d, d_emb) containing d word embeddings of dimension d_emb.
    - W_Q, W_K, W_V: numpy arrays representing learned weight matrices for Queries, Keys, and Values.
    - is_decoder: boolean indicating whether to apply masking for the decoder.
    
    Returns:
    - new_embeddings: numpy array of shape (d, d_emb) containing updated word embeddings.
    """

    d = embeddings.shape[0]  # Number of words in the input sequence
    d_k = W_Q.shape[1]       # Dimensionality of the Key vectors
    new_embeddings = np.zeros_like(embeddings)  # Initialize the new embeddings

    # Step 1: Compute Query, Key, and Value for each word
    for i in range(d):
        # Compute Query, Key, and Value vectors for the i-th word
        Q_i = np.dot(W_Q, embeddings[i])  # Query vector for word i
        K_i = np.dot(W_K, embeddings[i])  # Key vector for word i
        V_i = np.dot(W_V, embeddings[i])  # Value vector for word i

        # Step 2: Initialize attention weights for the i-th word
        attention_weights = np.zeros(d)

        # Step 3: Compute attention scores for the i-th word
        for j in range(d):
            if is_decoder and j > i:
                # Causal masking: Ignore future words in the decoder
                attention_weights[j] = 0
            else:
                # Calculate the attention score
                score = np.dot(Q_i, np.dot(W_K, embeddings[j])) / np.sqrt(d_k)  # Scale the dot product
                attention_weights[j] = np.exp(score)  # Exponentiate the score

        # Step 4: Normalize the attention weights
        attention_weights /= np.sum(attention_weights)  # Softmax step

        # Step 5: Compute the new embedding for the i-th word
        # Use the original Value vectors corresponding to the attention weights
        new_embeddings[i] = sum(attention_weights[j] * np.dot(W_V, embeddings[j]) for j in range(d))  # Update the new embedding

    return new_embeddings


# Example usage:
# Define input parameters
d = 5  # Number of words
d_emb = 4  # Dimensionality of word embeddings
embeddings = np.random.rand(d, d_emb)  # Randomly initialized word embeddings
W_Q = np.random.rand(d_emb, d_emb)  # Randomly initialized weight matrix for Queries
W_K = np.random.rand(d_emb, d_emb)  # Randomly initialized weight matrix for Keys
W_V = np.random.rand(d_emb, d_emb)  # Randomly initialized weight matrix for Values

# Call the attention mechanism function
new_embeddings = attention_mechanism(embeddings, W_Q, W_K, W_V, is_decoder=False)

print("Updated word embeddings:\n", new_embeddings)

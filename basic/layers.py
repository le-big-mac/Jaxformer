import jax
import jax.numpy as jnp

from .activations import gelu, softmax

def linear(x, W, b=None):
    """
    x : (..., d_in)
    W : (..., d_in, d_out)
    N.B. Be careful with shapes of x and W, as they are not explicitly checked.
    """
    if b is None:
        return jnp.einsum('...i,...ij->...j', x, W)
    return jnp.einsum('...i,...ij->...j', x, W) + b

def dot_product_attention(Q, K, V, mask):
    """
    Q : (batch, seq_len, num_heads, d_k)
    K : (batch, seq_len, num_heads, d_k)
    V : (batch, seq_len, num_heads, d_model)
    mask : (batch, num_heads, seq_len, seq_len)
    """
    Q = Q.transpose((0, 2, 1, 3)) # (batch, num_heads, seq_len, d_k)
    K = K.transpose((0, 2, 3, 1)) # (batch, num_heads, d_k, seq_len)
    V = V.transpose((0, 2, 1, 3)) # (batch, num_heads, seq_len, d_v)
    d_k = Q.shape[-1]
    scores = jnp.einsum('...ij,...jk->...ik', Q, K) / jnp.sqrt(d_k)
    scores = jnp.where(mask, scores, -jnp.inf)
    attention = softmax(scores, axis=-1) # (batch, num_heads, seq_len, seq_len)
    return jnp.einsum('...ij,...jk->...ik', attention, V).transpose((0, 2, 1, 3)) # (batch, seq_len, num_heads, d_v)

def embedding(x, embeddings):
    return jnp.take(embeddings, x, axis=0)

def multi_head_attention(x, attention_weights, linear_weights, d_k):
    """
    x : (batch, seq_len, d_model)
    attention_weights : (num_heads, d_model, 2*d_k + d_v)
    linear_weights : (num_heads*d_v, d_model)
    """
    x = jnp.expand_dims(x, axis=2)
    combined = linear(x, attention_weights)
    Q, K, V = jnp.split(combined, [d_k, 2*d_k], axis=-1) # d_v can be inferred from attention_weights and d_k
    mask = jnp.triu(jnp.ones((x.shape[1], x.shape[1])), 1) == 0
    mask = jnp.broadcast_to(mask, (x.shape[0], attention_weights.shape[1], *mask.shape))

    attention_matrix = dot_product_attention(Q, K, V, mask)
    ff_out = linear(attention_matrix.reshape(attention_matrix.shape[0], attention_matrix.shape[1], -1), linear_weights) # if num_heads*d_v = d_model, linear_weights could be identity
    return ff_out # (batch, seq_len, d_model)

def layer_normalization(x):
    mean = jnp.mean(x, axis=-1, keepdims=True)
    std = jnp.std(x, axis=-1, keepdims=True)
    return (x - mean) / std

def decoder_layer(x, attention_weights, linear_weights, feedforward_weights, feedforward_biases):
    """
    N.B. Not exactly same as decoder in encoder-decoder architecture, but convention is to call it decoder layer.
    x : (batch, seq_len, d_model)
    attention_weights : (num_heads, d_model, 2*d_k + d_v)
    feedforward_weights : (d_model, d_model)
    feedforward_biases : (d_model,)
    """
    attention_output = multi_head_attention(x, attention_weights, linear_weights) # (batch, seq_len, d_model)
    x = layer_normalization(x + attention_output)
    x = layer_normalization(x + gelu(linear(x, feedforward_weights) + feedforward_biases))
    return x

def init_decoder_layer(rng, num_heads, d_model, d_k, d_v):
    attention_weights = jax.random.normal(rng, (num_heads, d_model, 2*d_k + d_v))
    linear_weights = jax.random.normal(rng, (num_heads*d_v, d_model))
    feedforward_weights = jax.random.normal(rng, (d_model, d_model))
    feedforward_biases = jax.random.normal(rng, (d_model,))
    return attention_weights, linear_weights, feedforward_weights, feedforward_biases

def output_layer(x, output_weights, output_biases):
    """
    x : (batch, seq_len, d_model)
    output_weights : (seq_len*d_model, vocab_size)
    output_biases : (vocab_size,)
    Maybe should mask padding tokens?
    """
    return softmax(linear(x.reshape(x.shape[0], -1), output_weights, output_biases))

def init_output_layer(rng, seq_len, d_model, vocab_size):
    output_weights = jax.random.normal(rng, (seq_len*d_model, vocab_size))
    output_biases = jax.random.normal(rng, (vocab_size,))
    return output_weights, output_biases

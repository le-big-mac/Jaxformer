import jax
import jax.numpy as jnp

from basic.activations import gelu, softmax

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
    Q : (seq_len, num_heads, d_k)
    K : (seq_len, num_heads, d_k)
    V : (seq_len, num_heads, d_model)
    mask : (num_heads, seq_len, seq_len)
    """
    Q = Q.transpose((1, 0, 2)) # (num_heads, seq_len, d_k)
    K = K.transpose((1, 2, 0)) # (num_heads, d_k, seq_len)
    V = V.transpose((1, 0, 2)) # (num_heads, seq_len, d_v)
    d_k = Q.shape[-1]
    scores = jnp.einsum('...ij,...jk->...ik', Q, K) / jnp.sqrt(d_k)
    scores = jnp.where(mask, scores, -jnp.inf)
    attention = softmax(scores, axis=-1) # (num_heads, seq_len, seq_len)
    return jnp.einsum('...ij,...jk->...ik', attention, V).transpose((1, 0, 2)) # (seq_len, num_heads, d_v)

def embedding(x, embeddings):
    return jnp.take(embeddings, x, axis=0)

def multi_head_attention(x, qk_weights, v_weights, linear_weights):
    """
    x : (seq_len, d_model)
    qk_weights : (num_heads, d_model, 2*d_k)
    v_weights : (num_heads, d_model, d_v)
    linear_weights : (num_heads*d_v, d_model)
    """
    x = jnp.expand_dims(x, axis=1)
    qk = linear(x, qk_weights) # (seq_len, num_heads, 2*d_k)
    Q, K = jnp.split(qk, 2, axis=-1)
    V = linear(x, v_weights) # (seq_len, num_heads, d_v)
    mask = jnp.triu(jnp.ones((x.shape[0], x.shape[0])), 1) == 0
    mask = jnp.broadcast_to(mask, (qk_weights.shape[0], *mask.shape)) # (num_heads, seq_len, seq_len)

    attention_matrix = dot_product_attention(Q, K, V, mask) # (seq_len, num_heads, d_v)
    ff_out = linear(attention_matrix.reshape(attention_matrix.shape[0], -1), linear_weights) # if num_heads*d_v = d_model, linear_weights could be identity
    return ff_out # (seq_len, d_model)

def layer_normalization(x):
    mean = jnp.mean(x, axis=-1, keepdims=True)
    std = jnp.std(x, axis=-1, keepdims=True)
    return (x - mean) / std

def decoder_layer(x, qk_weights, v_weights, linear_weights, feedforward_weights, feedforward_biases, layer_norm=True):
    """
    N.B. Not exactly same as decoder in encoder-decoder architecture, but convention is to call it decoder layer.
    x : (seq_len, d_model)
    qk_weights : (num_heads, d_model, 2*d_k)
    v_weights : (num_heads, d_model, d_v)
    feedforward_weights : (d_model, d_model)
    feedforward_biases : (d_model,)
    """
    attention_output = multi_head_attention(x, qk_weights, v_weights, linear_weights) # (seq_len, d_model)
    if layer_norm:
        x = layer_normalization(x + attention_output)
        x = layer_normalization(x + gelu(linear(x, feedforward_weights) + feedforward_biases))
    else:
        x = x + attention_output
        x = x + gelu(linear(x, feedforward_weights) + feedforward_biases)
    return x

def init_decoder_layer(rng, num_heads, d_model, d_k, d_v):
    keys = jax.random.split(rng, 4)
    qk_weights = jax.random.normal(keys[0], (num_heads, d_model, 2*d_k)) / jnp.sqrt(d_model + d_k)
    v_weights = jax.random.normal(keys[1], (num_heads, d_model, d_v)) / jnp.sqrt(d_model + d_v)
    linear_weights = jax.random.normal(keys[1], (num_heads*d_v, d_model)) / jnp.sqrt(num_heads*d_v + d_v)
    feedforward_weights = jax.random.normal(keys[2], (d_model, d_model)) / jnp.sqrt(d_model + d_model)
    feedforward_biases = jax.random.normal(keys[3], (d_model,))
    return qk_weights, v_weights, linear_weights, feedforward_weights, feedforward_biases

def output_layer(x, output_weights, output_biases, temp=1.):
    """
    x : (seq_len, d_model)
    output_weights : (seq_len*d_model, vocab_size)
    output_biases : (vocab_size,)
    Maybe should mask padding tokens?
    """
    return softmax(linear(x.reshape(-1), output_weights, output_biases), temp=temp)

def init_output_layer(rng, seq_len, d_model, vocab_size):
    keys = jax.random.split(rng, 2)
    output_weights = jax.random.normal(keys[0], (seq_len*d_model, vocab_size)) / jnp.sqrt(seq_len*d_model + vocab_size)
    output_biases = jax.random.normal(keys[1], (vocab_size,))
    return output_weights, output_biases

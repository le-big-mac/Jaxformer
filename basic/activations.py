import jax.numpy as jnp

def relu(x):
    return jnp.maximum(0, x)

def gelu(x):
    return 0.5 * x * (1 + jnp.tanh(jnp.sqrt(2 / jnp.pi) * (x + 0.044715 * jnp.power(x, 3))))

def softmax(x, axis=-1, temp=1.):
    exp_x = jnp.exp((x - jnp.max(x, axis=axis, keepdims=True)) / temp) # for numerical stability
    return exp_x / jnp.sum(exp_x, axis=axis, keepdims=True)

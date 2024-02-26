import jax
import jax.numpy as jnp

@jax.jit()
def relu(x):
    return jnp.maximum(0, x)

@jax.jit()
def gelu(x):
    return 0.5 * x * (1 + jnp.tanh(jnp.sqrt(2 / jnp.pi) * (x + 0.044715 * jnp.power(x, 3))))

@jax.jit()
def softmax(x, axis):
    exp_x = jnp.exp(x - jnp.max(x, axis=axis, keepdims=True)) # for numerical stability
    return exp_x / jnp.sum(exp_x, axis=axis, keepdims=True)

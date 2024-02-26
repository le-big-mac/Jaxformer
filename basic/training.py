from jax import grad, jit
import jax.numpy as jnp

from .model import model

def cross_entropy_loss(param_dict, x, y):
    return -jnp.sum(y * jnp.log(model(param_dict, x)))

@jit
def sgd_update(param_dict, x, y, learning_rate):
    loss = cross_entropy_loss(param_dict, x, y)
    grads = grad(cross_entropy_loss)(param_dict, x, y)
    return {k: v - learning_rate * g for (k, v), g in zip(param_dict.items(), grads)}, loss

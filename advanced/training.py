from jax import grad, jit
import jax.numpy as jnp

from basic.training import cross_entropy_loss

@jit
def adam_update(param_dict, x, y, learning_rate, m, v, t, beta1=0.9, beta2=0.999, eps=1e-8):
    for k, v in param_dict.items():
        grads = grad(cross_entropy_loss)(param_dict, x, y)
        m[k] = beta1 * m[k] + (1 - beta1) * grads[k]
        v[k] = beta2 * v[k] + (1 - beta2) * grads[k]**2
        m_hat = m[k] / (1 - beta1**t)
        v_hat = v[k] / (1 - beta2**t)
        param_dict[k] -= learning_rate * m_hat / (jnp.sqrt(v_hat) + eps)

    loss = cross_entropy_loss(param_dict, x, y)
    return param_dict, loss

def init_adam_params(param_dict):
    m = {k: jnp.zeros_like(v) for k, v in param_dict.items()}
    v = {k: jnp.zeros_like(v) for k, v in param_dict.items()}
    return m, v

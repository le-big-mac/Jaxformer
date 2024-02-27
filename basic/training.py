from jax import jit
import jax.numpy as jnp

def get_cross_entropy_loss(model):
    def cross_entropy_loss(param_dict, positional_encoding, x, y):
        return -jnp.sum(y * jnp.log(model(param_dict, positional_encoding, x))) / x.shape[0]
    return jit(cross_entropy_loss)

@jit
def sgd_update(param_dict, grads, learning_rate, train_embeddings=True):
    new_param_dict = {k: v - learning_rate * grads[k] for (k, v) in param_dict.items() if k != 'embeddings'}

    if train_embeddings:
        new_param_dict['embeddings'] = param_dict['embeddings'] - learning_rate * grads['embeddings']
        new_param_dict['embeddings'] = jnp.concatenate([jnp.zeros((1, new_param_dict['embeddings'].shape[1])), new_param_dict['embeddings'][1:]], axis=0)

    return new_param_dict

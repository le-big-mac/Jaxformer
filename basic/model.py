import jax
import jax.numpy as jnp

from basic.layers import *

def get_model(num_decoder_layers, temp=1., layer_norm=True):
    # This is dumb, but lets me make optimizers more general
    def model(param_dict: dict,
              positional_encoding: jnp.ndarray,
              x: jnp.ndarray,
              ):
        x = embedding(x, param_dict['embeddings']) + positional_encoding
        for i in range(num_decoder_layers):
            qkw = param_dict[f'qkw_{i}']
            vw = param_dict[f'vw_{i}']
            lw = param_dict[f'lw_{i}']
            fw = param_dict[f'fw_{i}']
            fb = param_dict[f'fb_{i}']
            x = decoder_layer(x, qkw, vw, lw, fw, fb, layer_norm=layer_norm)
        return output_layer(x, param_dict['ow'], param_dict['ob'], temp=temp)

    return model

def batch_model(model):
    return jax.vmap(model, in_axes=(None, None, 0), out_axes=0)

def init_model(rng, num_decoder_layers, num_heads, d_model, d_k, d_v, seq_len, vocab_size):
    keys = jax.random.split(rng, num_decoder_layers + 1)
    param_dict = {}
    for i in range(num_decoder_layers):
        qkw, vw, lw, fw, fb = init_decoder_layer(keys[i], num_heads, d_model, d_k, d_v)
        param_dict[f'qkw_{i}'] = qkw
        param_dict[f'vw_{i}'] = vw
        param_dict[f'lw_{i}'] = lw
        param_dict[f'fw_{i}'] = fw
        param_dict[f'fb_{i}'] = fb
    param_dict['ow'], param_dict['ob'] = init_output_layer(keys[-1], seq_len, d_model, vocab_size)
    return param_dict

def num_params(param_dict):
    return sum(v.size for v in param_dict.values())

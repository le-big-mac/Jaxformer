import jax
import jax.numpy as jnp

from .layers import *

def get_model(num_decoder_layers):
    # This is dumb, but lets me make optimizers more general
    def model(param_dict: dict,
              x: jnp.ndarray,
              ):
        for i in range(num_decoder_layers):
            aw = param_dict[f'aw_{i}']
            lw = param_dict[f'lw_{i}']
            fw = param_dict[f'fw_{i}']
            fb = param_dict[f'fb_{i}']
            x = decoder_layer(x, aw, lw, fw, fb)
        return output_layer(x, param_dict['ow'], param_dict['ob'])

    return model

def init_model(rng, num_decoder_layers, num_heads, d_model, d_k, d_v, seq_len, vocab_size):
    param_dict = {}
    for i in range(num_decoder_layers):
        aw, lw, fw, fb = init_decoder_layer(rng, num_heads, d_model, d_k, d_v)
        param_dict[f'aw_{i}'] = aw
        param_dict[f'lw_{i}'] = lw
        param_dict[f'fw_{i}'] = fw
        param_dict[f'fb_{i}'] = fb
    param_dict['ow'], param_dict['ob'] = init_output_layer(rng, seq_len, d_model, vocab_size)
    return param_dict

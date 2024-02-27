import jax
import jax.numpy as jnp

def tokenize(text_list):
    tokenized_sentences = []
    word_to_int = {}
    unique_integers = 1

    for sentence in text_list:
        tokenized_sentence = []
        sentence = sentence.lower().strip()
        for word in sentence.split():
            if word not in word_to_int:
                word_to_int[word] = unique_integers
                unique_integers += 1
            tokenized_sentence.append(word_to_int[word])
        tokenized_sentences.append(tokenized_sentence)

    return tokenized_sentences, word_to_int, unique_integers

def create_embeddings(rng, vocab_size, d_model):
    pad = jnp.zeros((1, d_model))
    return jnp.concatenate([pad, jax.random.normal(rng, (vocab_size, d_model))], axis=0)

def onehot(x, vocab_size):
    return jnp.array(x[:, None] == jnp.arange(vocab_size), dtype=jnp.float32)
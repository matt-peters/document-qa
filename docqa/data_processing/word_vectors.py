import gzip
import pickle
from os.path import join, exists
from typing import Iterable, Optional

import numpy as np

from docqa.config import VEC_DIR


""" Loading words vectors """


def load_word_vectors(vec_name: str, vocab: Optional[Iterable[str]]=None, is_path=False, lower=True):
    if not is_path:
        vec_path = join(VEC_DIR, vec_name)
    else:
        vec_path = vec_name
    if exists(vec_path + ".txt"):
        vec_path = vec_path + ".txt"
    elif exists(vec_path + ".txt.gz"):
        vec_path = vec_path + ".txt.gz"
    elif exists(vec_path + ".pkl"):
        vec_path = vec_path + ".pkl"
    else:
        raise ValueError("No file found for vectors %s" % vec_name)
    return load_word_vector_file(vec_path, vocab, lower=lower)


def load_word_vector_file(vec_path: str, vocab: Optional[Iterable[str]] = None, lower=True):
    if vocab is not None:
        if lower:
            vocab = set(x.lower() for x in vocab)
        else:
            vocab = set(vocab)

    # notes some of the large vec files produce utf-8 errors for some words, just skip them
    if vec_path.endswith(".pkl"):
        with open(vec_path, "rb") as f:
            return pickle.load(f)
    elif vec_path.endswith(".txt.gz"):
        handle = lambda x: gzip.open(x, 'r', encoding='utf-8', errors='ignore')
    else:
        handle = lambda x: open(x, 'r', encoding='utf-8', errors='ignore')

    pruned_dict = {}
    with handle(vec_path) as fh:
        for line in fh:
            word_ix = line.find(" ")
            word = line[:word_ix]
            if lower:
                norm_word = word.lower()
            else:
                norm_word = word
            if (vocab is None) or (norm_word in vocab):
                pruned_dict[word] = np.array([float(x) for x in line[word_ix + 1:-1].split(" ")], dtype=np.float32)
    return pruned_dict

'''Preprocessing functions and pipeline'''
import nltk
nltk.download('punkt')
import torch
import logging
import numpy as np
from collections import defaultdict

from allennlp.data import Instance, Vocabulary, Token
from allennlp.data.fields import TextField
from allennlp.data.token_indexers import SingleIdTokenIndexer
from allennlp_mods.numeric_field import NumericField
from datasets.loaddata import STSBTask
from collections import defaultdict


def build_tasks(args, csv_path, max_seq_len, labeled_ratio):
    '''Prepare tasks'''
    task = STSBTask(csv_path, max_seq_len, labeled_ratio)
    max_v_sizes = {'word': args.max_word_v_size}
    token_indexer = {}
    token_indexer["words"] = SingleIdTokenIndexer()  
    word2freq = get_words(task)  
    vocab = get_vocab(word2freq, max_v_sizes)  
    word_embs = get_embeddings(vocab, args.word_embs_file, args.d_word)   
    label, unlabel, val, test = process_task(task, token_indexer, vocab)  
    task.label_data = label
    task.unlabel_data = unlabel
    task.val_data = val
    task.test_data = test

    return task, vocab, word_embs


def get_words(task):
    word2freq = defaultdict(int)

    def count_sentences(sentences):
        for sentence in sentences:
            for word in sentence:
                word2freq[word] += 1

    splits = [task.label_data_text, task.unlabel_data_text, task.val_data_text, task.test_data_text]
    for split in filter(None, splits):  
        for part in split:  
            if isinstance(part, list) and part and isinstance(part[0], list):  
                count_sentences(part)

    logging.info("\tFinished counting words")
    return word2freq


def get_vocab(word2freq, max_v_sizes):
    vocab = Vocabulary(counter=None, max_vocab_size=max_v_sizes['word'])
    words_by_freq = [(word, freq) for word, freq in word2freq.items()]
    words_by_freq.sort(key=lambda x: x[1], reverse=True) 
    for word, _ in words_by_freq[:max_v_sizes['word']]:
        vocab.add_token_to_namespace(word, 'tokens')  
    return vocab


def get_embeddings(vocab, vec_file, d_word):
    word_v_size, unk_idx = vocab.get_vocab_size('tokens'), vocab.get_token_index(vocab._oov_token)  
    embeddings = np.random.randn(word_v_size, d_word)
    with open(vec_file) as vec_fh:
        for line in vec_fh:
            word, vec = line.split(' ', 1)  
            idx = vocab.get_token_index(word)
            if idx != unk_idx:
                idx = vocab.get_token_index(word)
                embeddings[idx] = np.array(list(map(float, vec.split())))
    embeddings[vocab.get_token_index('@@PADDING@@')] = 0.  
    embeddings = torch.FloatTensor(embeddings)

    return embeddings


def process_task(task, token_indexer, vocab):
    splits = {
        "label": task.label_data_text,
        "unlabel": task.unlabel_data_text,
        "val": task.val_data_text,
        "test": task.test_data_text
    }

    processed = {}
    for name, split in splits.items():
        if split is not None:
            processed[name] = process_split(split, token_indexer)
        else:
            processed[name] = None

    for inst_list in processed.values():
        if inst_list is not None:
            for instance in inst_list:
                instance.index_fields(vocab)

    return processed["label"], processed["unlabel"], processed["val"], processed["test"]


def process_split(split, indexers):
    
    def to_textfields(sent_list):
        return [TextField([Token(w) for w in sent], token_indexers=indexers) for sent in sent_list]

    labels = [NumericField(l) for l in split[-1]]

    if len(split) == 3:  
        inputs1, inputs2 = to_textfields(split[0]), to_textfields(split[1])
        return [
            Instance({"input1": i1, "input2": i2, "label": lbl})
            for i1, i2, lbl in zip(inputs1, inputs2, labels)
        ]
    elif len(split) == 5:  
        i1w, i2w, i1s, i2s = map(to_textfields, split[:-1])
        return [
            Instance({
                "input1_weak": iw1, "input2_weak": iw2,
                "input1_strong": is1, "input2_strong": is2,
                "label": lbl
            })
            for iw1, iw2, is1, is2, lbl in zip(i1w, i2w, i1s, i2s, labels)
        ]
    else:
        raise ValueError(f"Unexpected split format with {len(split)} parts")

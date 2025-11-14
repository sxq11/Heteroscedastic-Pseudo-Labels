import os
import codecs
import logging
import string
import numpy as np
import random
import nltk


def process_sentence(sent, max_seq_len):
    """process a sentence using NLTK toolkit"""
    if sent is None:
        return []
    return nltk.word_tokenize(sent)[:max_seq_len]  


def is_punctuation_only(word_list):
    punctuation_set = set(string.punctuation)
    return all(word in punctuation_set for word in word_list)


def load_sts_tsv(data_file, max_seq_len=40, labeled_ratio=0.1, delimiter='\t'):
    sent1s, sent2s, scores, splits, sent_s1s, sent_s2s = [], [], [], [], [], []

    with codecs.open(data_file, 'r', 'utf-8') as f:
        header = f.readline().strip().split(delimiter)
        for row_idx, row in enumerate(f):
            try:
                row = row.strip().split(delimiter)
                s1 = row[7]          
                s2 = row[8]           
                score = float(row[9]) 
                split = row[10]       
                s1_strong = row[11] if row[11] != "None" else None
                s2_strong = row[12] if row[12] != "None" else None

                s1_tok = process_sentence(s1, max_seq_len)
                s2_tok = process_sentence(s2, max_seq_len)
                s1_strong_tok = process_sentence(s1_strong, max_seq_len) if s1_strong else None
                s2_strong_tok = process_sentence(s2_strong, max_seq_len) if s2_strong else None

                if not s1_tok or not s2_tok: 
                    continue

                sent1s.append(s1_tok)
                sent2s.append(s2_tok)
                scores.append(score)
                splits.append(split)
                sent_s1s.append(s1_strong_tok)
                sent_s2s.append(s2_strong_tok)
            except Exception as e:
                logging.warning(f"Error at row {row_idx}: {e}")
                continue

    train_idx = [i for i, sp in enumerate(splits) if sp == "train"]
    dev_idx   = [i for i, sp in enumerate(splits) if sp == "dev"]
    test_idx  = [i for i, sp in enumerate(splits) if sp == "test"]

    np.random.seed(0)
    n_train = len(train_idx)
    n_labeled = int(n_train * labeled_ratio)
    labeled_idx = np.random.choice(train_idx, n_labeled, replace=False)
    unlabeled_idx = set(train_idx) - set(labeled_idx)

    label_data = (
        [sent1s[i] for i in labeled_idx],
        [sent2s[i] for i in labeled_idx],
        [scores[i] for i in labeled_idx],
    )

    unlabel_data = (
        [sent1s[i] for i in unlabeled_idx],   
        [sent2s[i] for i in unlabeled_idx],   
        [sent_s1s[i] for i in unlabeled_idx], 
        [sent_s2s[i] for i in unlabeled_idx], 
        [scores[i] for i in unlabeled_idx],   
    )

    val_data = (
        [sent1s[i] for i in dev_idx],
        [sent2s[i] for i in dev_idx],
        [scores[i] for i in dev_idx],
    )

    test_data = (
        [sent1s[i] for i in test_idx],
        [sent2s[i] for i in test_idx],
        [scores[i] for i in test_idx],
    )

    return label_data, unlabel_data, val_data, test_data


class STSBTask:
    def __init__(self, path, max_seq_len, labeled_ratio):
        super(STSBTask, self).__init__()
        self.path = path
        self.max_seq_len = max_seq_len
        self.labeled_ratio = labeled_ratio
        self.label_data_text, self.unlabel_data_text, self.val_data_text, self.test_data_text = None, None, None, None
        self.load_data()

    def load_data(self):
        label_data, unlabel_data, val_data, test_data = load_sts_tsv(
            self.path,
            self.max_seq_len,
            labeled_ratio=self.labeled_ratio
        )
        self.label_data_text = label_data
        self.unlabel_data_text = unlabel_data
        self.val_data_text = val_data
        self.test_data_text = test_data
        logging.info("\tFinished loading STS Benchmark data.")

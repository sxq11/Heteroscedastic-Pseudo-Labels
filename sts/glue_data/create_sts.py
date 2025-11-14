import os
import random
import nltk
from nltk.corpus import wordnet


def get_synonyms(word):
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonyms.add(lemma.name())
    return list(synonyms)


def synonym_replacement(sentence, n):
    words = nltk.word_tokenize(sentence)
    new_words = words.copy()
    random_word_list = list(set([word for word in words if wordnet.synsets(word)]))
    random.shuffle(random_word_list)
    num_replaced = 0
    for random_word in random_word_list:
        synonyms = get_synonyms(random_word)
        if len(synonyms) >= 1:
            synonym = random.choice(synonyms)
            new_words = [synonym if word == random_word else word for word in new_words]
            num_replaced += 1
        if num_replaced >= n:
            break
    sentence = ' '.join(new_words)
    return sentence


def insert_synonyms(sentence):
    words = nltk.word_tokenize(sentence)
    new_sentence = []
    for word in words:
        synonyms = get_synonyms(word)
        if synonyms and random.choice([True, False]):
            synonym = random.choice(synonyms)
            new_sentence.append(synonym)
        new_sentence.append(word)
    return ' '.join(new_sentence)


def transform_strong(sent):
    sent = synonym_replacement(sent, 1)
    sent = insert_synonyms(sent)
    return sent

data_dir = "./glue_data/STS-B"
splits = {
    "train": "train_new.tsv",
    "dev": "dev_new.tsv",
    "test": "test_new.tsv"
}

output_file = os.path.join(data_dir, "sts.tsv")
all_rows = []

for split, filename in splits.items():
    filepath = os.path.join(data_dir, filename)
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    header = lines[0].strip().split('\t')
    if split == "train":
        if not all_rows:
            header.extend(["split", "sent_s1", "sent_s2"])
            all_rows.append('\t'.join(header))

    for line in lines[1:]:
        row_split = line.strip().split('\t')
        sentence1 = row_split[7]
        sentence2 = row_split[8]

        if split == "train":
            sent_s1 = transform_strong(sentence1)
            sent_s2 = transform_strong(sentence2)
        else:
            sent_s1, sent_s2 = "None", "None"

        row_split.extend([split, sent_s1, sent_s2])
        all_rows.append('\t'.join(row_split))

with open(output_file, 'w', encoding='utf-8') as f:
    f.write('\n'.join(all_rows))

print(f"saved to {output_file}")

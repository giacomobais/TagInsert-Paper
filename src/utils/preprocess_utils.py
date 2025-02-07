import yaml
import torch
import json
import numpy as np

PROP_CONVERTER = {1: "100%", 0.75: "75%", 0.5: "50%", 0.25: "25%", 0.1: "10%"}

def load_config(config_path):
    with open(config_path) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    return config

def prepare_data(train_file, val_file, test_file, tagging, block_size = 52, keep_proportion = 1):

    # reading data from file
    sentence_tokens = []
    test_sentence_tokens = []
    sentence_POS = []
    val_sentence_tokens = []
    val_sentence_POS = []
    test_sentence_POS = []
    vocab = []
    vocab_POS = []
    x = open(train_file)
    for line in x.readlines():
        tokens = []
        POS = []
        pairs = line.split(" ")
        if len(pairs) <= block_size-2:
            for pair in pairs:
                pair = pair.strip('\n')
                if pair != "\n":
                    word = pair.split("|")[0]
                    tag = pair.split("|")[1]
                    tokens.append(word)
                    vocab.append(word)
                    vocab_POS.append(tag)
                    POS.append(tag)
            sentence_tokens.append(tokens)
            sentence_POS.append(POS)
    x.close()

    x = open(val_file)
    for line in x.readlines():
        tokens = []
        POS = []
        pairs = line.split(" ")
        if len(pairs) <= block_size-2:
            for pair in pairs:
                pair = pair.strip('\n')
                if pair != "\n":
                    word = pair.split("|")[0]
                    tag = pair.split("|")[1]
                    tokens.append(word)
                    vocab.append(word)
                    vocab_POS.append(tag)
                    POS.append(tag)
            val_sentence_tokens.append(tokens)
            val_sentence_POS.append(POS)
    x.close()

    x = open(test_file)
    for line in x.readlines():
        tokens = []
        POS = []
        pairs = line.split(" ")
        if len(pairs) <= block_size-2:
            for pair in pairs:
                pair = pair.strip('\n')
                if pair != "\n":
                    word = pair.split("|")[0]
                    tag = pair.split("|")[1]
                    tokens.append(word)
                    vocab.append(word)
                    vocab_POS.append(tag)
                    POS.append(tag)
            test_sentence_tokens.append(tokens)
            test_sentence_POS.append(POS)
    x.close()

    # create mapping for POS tags
    vocab_POS = sorted(list(set(vocab_POS)))
    POS_to_idx = {tag: i+1 for i, tag in enumerate(vocab_POS)}

    # add padding, unknown, and start tokens
    POS_to_idx["<PAD>"] = 0
    POS_to_idx["<START>"] = len(POS_to_idx)
    POS_to_idx["<UNK>"] = len(POS_to_idx)
    POS_to_idx["<END>"] = len(POS_to_idx)
    idx_to_POS = {k: v for v, k in POS_to_idx.items()}

    # convert data to integers
    sentence_POS_idx = []
    for sentence in sentence_POS:
        sentence_idx = [POS_to_idx[tag] for tag in sentence]
        # add start token
        sentence_idx = [POS_to_idx['<START>']] + sentence_idx
        # pad sentence to 100
        sentence_idx += [POS_to_idx['<PAD>']] * (block_size - len(sentence_idx))
        sentence_POS_idx.append(sentence_idx)

    val_sentence_POS_idx = []
    for sentence in val_sentence_POS:
        sentence_idx = [POS_to_idx[tag] for tag in sentence]
        # add start token
        sentence_idx = [POS_to_idx['<START>']] + sentence_idx
        # pad sentence to 100
        sentence_idx += [POS_to_idx['<PAD>']] * (block_size - len(sentence_idx))
        val_sentence_POS_idx.append(sentence_idx)

    test_sentence_POS_idx = []
    for sentence in test_sentence_POS:
        sentence_idx = [POS_to_idx[tag] for tag in sentence]
        # add start token
        sentence_idx = [POS_to_idx['<START>']] + sentence_idx
        # pad sentence to 100
        sentence_idx += [POS_to_idx['<PAD>']] * (block_size - len(sentence_idx))
        test_sentence_POS_idx.append(sentence_idx)

    # create mapping for words
    vocab = sorted(list(set(vocab)))
    word_to_idx = {word: i+1 for i, word in enumerate(vocab)}

    # add pad
    word_to_idx["<PAD>"] = 0
    idx_to_word = {k: v for v, k in word_to_idx.items()}

    # convert data to integers
    sentence_tokens_idx = []
    for sentence in sentence_tokens:
        sentence_idx = [word_to_idx[word] for word in sentence]
        # pad sentence to 100
        sentence_idx += [word_to_idx['<PAD>']] * (block_size - len(sentence_idx))
        sentence_tokens_idx.append(sentence_idx)

    val_sentence_tokens_idx = []
    for sentence in val_sentence_tokens:
        sentence_idx = [word_to_idx[word] for word in sentence]
        # pad sentence to 100
        sentence_idx += [word_to_idx['<PAD>']] * (block_size - len(sentence_idx))
        val_sentence_tokens_idx.append(sentence_idx)

    test_sentence_tokens_idx = []
    for sentence in test_sentence_tokens:
        sentence_idx = [word_to_idx[word] for word in sentence]
        # pad sentence to 100
        sentence_idx += [word_to_idx['<PAD>']] * (block_size - len(sentence_idx))
        test_sentence_tokens_idx.append(sentence_idx)

    # randomly sample a proportion of the data
    if keep_proportion == 1.0:
        keep_proportion = 1
    if keep_proportion != 1:
        sentences_to_keep = np.random.choice(len(sentence_tokens_idx), int(len(sentence_tokens_idx)*keep_proportion), replace = False)
        sentence_tokens_idx = [sentence_tokens_idx[i] for i in sentences_to_keep]
        sentence_POS_idx = [sentence_POS_idx[i] for i in sentences_to_keep]
        sentence_tokens = [sentence_tokens[i] for i in sentences_to_keep]

    # renaming for handiness
    train_words = sentence_tokens_idx
    train_tags = sentence_POS_idx
    val_words = val_sentence_tokens_idx
    val_tags = val_sentence_POS_idx
    test_words = test_sentence_tokens_idx
    test_tags = test_sentence_POS_idx
    train_original_sentences = sentence_tokens
    val_original_sentences = val_sentence_tokens
    test_original_sentences = test_sentence_tokens

    # saving processed data
    prop_path = PROP_CONVERTER[keep_proportion]
    torch.save({'words': train_words, 'tags': train_tags, 'original_sentences': train_original_sentences}, f'data/{tagging}/processed/{prop_path}/train_data.pth')
    torch.save({'words': val_words, 'tags': val_tags, 'original_sentences': val_original_sentences}, f'data/{tagging}/processed/{prop_path}/val_data.pth')
    torch.save({'words': test_words, 'tags': test_tags, 'original_sentences': test_original_sentences}, f'data/{tagging}/processed/{prop_path}/test_data.pth')

    with open(f'data/{tagging}/processed/{prop_path}/word_to_idx.json', 'w') as f:
        json.dump(word_to_idx, f)
    with open(f'data/{tagging}/processed/{prop_path}/idx_to_word.json', 'w') as f:
        json.dump(idx_to_word, f)
    with open(f'data/{tagging}/processed/{prop_path}/{tagging}_to_idx.json', 'w') as f:
        json.dump(POS_to_idx, f)
    with open(f'data/{tagging}/processed/{prop_path}/idx_to_{tagging}.json', 'w') as f:
        json.dump(idx_to_POS, f)
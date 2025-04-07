import yaml
import torch
import json
import numpy as np
from transformers import AutoModel, AutoTokenizer

PROP_CONVERTER = {1: "100%", 0.75: "75%", 0.5: "50%", 0.25: "25%", 0.1: "10%"}
BERT_FINDER = {"en": "bert-base-cased", "de": "bert-base-german-cased", "it": "dbmdz/bert-base-italian-cased", "nl":"GroNLP/bert-base-dutch-cased"}

def load_config(config_path):
    """
    Simple function to load the config file from the path specified
    """
    with open(config_path) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    return config

def prepare_data(train_file, val_file, test_file, tagging, config):
    """
    Preprocessing function to prepare the data for training. It reads from a file formatted as follows:
    word1|tag1 word2|tag2 ... wordn|tagn
    It then tokenizes the words and tags, creates mappings for the words and tags, and extracts BERT embeddings
    The data is then saved in the processed folder in the data directory in a tensor format
    """
    # reading data from file
    sentence_tokens = []
    test_sentence_tokens = []
    sentence_POS = []
    val_sentence_tokens = []
    val_sentence_POS = []
    test_sentence_POS = []
    vocab = []
    vocab_POS = []

    # reading the training data
    x = open(train_file, encoding='utf-8')
    for line in x.readlines():
        tokens = []
        POS = []
        pairs = line.split(" ")
        # ignore sentences that are too long for the specified block size
        if len(pairs) <= config['block_size']-2:
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
    # reading the validation and test data, same as training data
    x = open(val_file, encoding='utf-8')
    for line in x.readlines():
        tokens = []
        POS = []
        pairs = line.split(" ")
        if len(pairs) <= config['block_size']-2:
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

    x = open(test_file, encoding='utf-8')
    for line in x.readlines():
        tokens = []
        POS = []
        pairs = line.split(" ")
        if len(pairs) <= config['block_size']-2:
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

    # add padding, unknown, and start and end tokens
    POS_to_idx["<PAD>"] = 0
    POS_to_idx["<START>"] = len(POS_to_idx)
    POS_to_idx["<UNK>"] = len(POS_to_idx)
    POS_to_idx["<END>"] = len(POS_to_idx)
    idx_to_POS = {k: v for v, k in POS_to_idx.items()}

    # convert data to integers using the mapping, also add start token and pad to block size
    sentence_POS_idx = []
    for sentence in sentence_POS:
        sentence_idx = [POS_to_idx[tag] for tag in sentence]
        # add start token
        sentence_idx = [POS_to_idx['<START>']] + sentence_idx
        # pad sentence to block size
        sentence_idx += [POS_to_idx['<PAD>']] * (config['block_size'] - len(sentence_idx))
        sentence_POS_idx.append(sentence_idx)
    val_sentence_POS_idx = []
    for sentence in val_sentence_POS:
        sentence_idx = [POS_to_idx[tag] for tag in sentence]
        sentence_idx = [POS_to_idx['<START>']] + sentence_idx
        sentence_idx += [POS_to_idx['<PAD>']] * (config['block_size'] - len(sentence_idx))
        val_sentence_POS_idx.append(sentence_idx)
    test_sentence_POS_idx = []
    for sentence in test_sentence_POS:
        sentence_idx = [POS_to_idx[tag] for tag in sentence]
        sentence_idx = [POS_to_idx['<START>']] + sentence_idx
        sentence_idx += [POS_to_idx['<PAD>']] * (config['block_size'] - len(sentence_idx))
        test_sentence_POS_idx.append(sentence_idx)

    # create mapping for words
    vocab = sorted(list(set(vocab)))
    word_to_idx = {word: i+1 for i, word in enumerate(vocab)}
    # add pad
    word_to_idx["<PAD>"] = 0
    idx_to_word = {k: v for v, k in word_to_idx.items()}

    # convert words to integers
    sentence_tokens_idx = []
    for sentence in sentence_tokens:
        sentence_idx = [word_to_idx[word] for word in sentence]
        # pad sentence to block size
        sentence_idx += [word_to_idx['<PAD>']] * (config['block_size'] - len(sentence_idx))
        sentence_tokens_idx.append(sentence_idx)
    val_sentence_tokens_idx = []
    for sentence in val_sentence_tokens:
        sentence_idx = [word_to_idx[word] for word in sentence]
        sentence_idx += [word_to_idx['<PAD>']] * (config['block_size'] - len(sentence_idx))
        val_sentence_tokens_idx.append(sentence_idx)
    test_sentence_tokens_idx = []
    for sentence in test_sentence_tokens:
        sentence_idx = [word_to_idx[word] for word in sentence]
        sentence_idx += [word_to_idx['<PAD>']] * (config['block_size'] - len(sentence_idx))
        test_sentence_tokens_idx.append(sentence_idx)

    # # extract BERT embeddings in batches of batch_size
    # train_embs = []
    # # load the BERT model and tokenizer
    # bert_name = BERT_FINDER[config['language']]
    # bert_model, tokenizer = load_BERT_encoder(bert_name, config['device'])
    # for i in range(0, len(sentence_tokens_idx), config['batch_size']):
    #     embs = extract_BERT_embs(sentence_tokens[i:i+config['batch_size']], bert_model, tokenizer, config)
    #     train_embs.append(embs)
    
    # randomly sample a proportion of the training data, depending on the config file
    keep_proportion = float(config['data_proportion'])
    if keep_proportion == 1.0:
        keep_proportion = 1
    if keep_proportion != 1:
        sentences_to_keep = np.random.choice(len(sentence_tokens_idx), int(len(sentence_tokens_idx)*keep_proportion), replace = False)
        sentence_tokens_idx = [sentence_tokens_idx[i] for i in sentences_to_keep]
        sentence_POS_idx = [sentence_POS_idx[i] for i in sentences_to_keep]
        sentence_tokens = [sentence_tokens[i] for i in sentences_to_keep]
        # train_embs = [train_embs[i] for i in sentences_to_keep]
    # train_embs = torch.cat(train_embs, dim=0)

    # extract BERT embeddings for validation and test data
    # val_embs = []
    # for i in range(0, len(val_sentence_tokens_idx), config['batch_size']):
    #     embs = extract_BERT_embs(val_sentence_tokens[i:i+config['batch_size']], bert_model, tokenizer, config)
    #     val_embs.append(embs)
    # val_embs = torch.cat(val_embs, dim=0)
    # test_embs = []
    # for i in range(0, len(test_sentence_tokens_idx), config['batch_size']):
    #     embs = extract_BERT_embs(test_sentence_tokens[i:i+config['batch_size']], bert_model, tokenizer, config)
    #     test_embs.append(embs)
    # test_embs = torch.cat(test_embs, dim=0)

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

    # torch.save({'words': train_words, 'tags': train_tags, 'original_sentences': train_original_sentences, 'embs': train_embs}, f'data/{tagging}/processed/{prop_path}/train_data.pth')
    # torch.save({'words': val_words, 'tags': val_tags, 'original_sentences': val_original_sentences, 'embs': val_embs}, f'data/{tagging}/processed/{prop_path}/val_data.pth')
    # torch.save({'words': test_words, 'tags': test_tags, 'original_sentences': test_original_sentences, 'embs': test_embs}, f'data/{tagging}/processed/{prop_path}/test_data.pth')
    if tagging == 'PMB':
        language = config['language']
        torch.save({'words': train_words, 'tags': train_tags, 'original_sentences': train_original_sentences}, f'data/{tagging}/{language}/processed/train_data.pth')
        torch.save({'words': val_words, 'tags': val_tags, 'original_sentences': val_original_sentences}, f'data/{tagging}/{language}/processed/val_data.pth')
        torch.save({'words': test_words, 'tags': test_tags, 'original_sentences': test_original_sentences}, f'data/{tagging}/{language}/processed/test_data.pth')
    else:
        torch.save({'words': train_words, 'tags': train_tags, 'original_sentences': train_original_sentences}, f'data/{tagging}/processed/{prop_path}/train_data.pth')
        torch.save({'words': val_words, 'tags': val_tags, 'original_sentences': val_original_sentences}, f'data/{tagging}/processed/{prop_path}/val_data.pth')
        torch.save({'words': test_words, 'tags': test_tags, 'original_sentences': test_original_sentences}, f'data/{tagging}/processed/{prop_path}/test_data.pth')


    # saving mappings
    if tagging == 'PMB':
        language = config['language']
        with open(f'data/{tagging}/{language}/processed/word_to_idx.json', 'w', encoding='utf-8') as f:
            json.dump(word_to_idx, f, ensure_ascii=False)
        with open(f'data/{tagging}/{language}/processed/idx_to_word.json', 'w', encoding='utf-8') as f:
            json.dump(idx_to_word, f, ensure_ascii=False)
        with open(f'data/{tagging}/{language}/processed/{tagging}_to_idx.json', 'w', encoding='utf-8') as f:
            json.dump(POS_to_idx, f, ensure_ascii=False)
        with open(f'data/{tagging}/{language}/processed/idx_to_{tagging}.json', 'w', encoding='utf-8') as f:
            json.dump(idx_to_POS, f, ensure_ascii=False)
    else:
        with open(f'data/{tagging}/processed/{prop_path}/word_to_idx.json', 'w') as f:
            json.dump(word_to_idx, f)
        with open(f'data/{tagging}/processed/{prop_path}/idx_to_word.json', 'w') as f:
            json.dump(idx_to_word, f)
        with open(f'data/{tagging}/processed/{prop_path}/{tagging}_to_idx.json', 'w') as f:
            json.dump(POS_to_idx, f)
        with open(f'data/{tagging}/processed/{prop_path}/idx_to_{tagging}.json', 'w') as f:
            json.dump(idx_to_POS, f)



def extract_BERT_embs(sentences, bert_model, tokenizer, config):
    """
    Function to extract the BERT embeddings for the sentences
    Through the mapping previously created, we can extract the embeddings for the words depending on the subword management
    The embeddings are calculated for the last four layers and summed up to get a single embedding for the word
    """

    device = config['device']
    # string together the sentences
    marked_text = [" ".join(sentence) for sentence in sentences]
    # use tokenizer to get tokenized text and pad up to the bert block size specified in the config
    tokens_tensor = [tokenizer(text, padding="max_length", truncation=True, max_length=config['bert_block_size'], return_tensors="pt") for text in marked_text]
    # get the attention mask and the input ids
    attention_mask = torch.stack([t['attention_mask'] for t in tokens_tensor])
    tokens_tensor = torch.stack([t['input_ids'] for t in tokens_tensor]).squeeze(1)
    # create the mappings for the subwords
    mappings = [create_mapping(sentence, tokens_tensor[i], tokenizer, cased = True, subword_manage=config['embedding_strategy']) for i, sentence in enumerate(sentences)]
    tokens_tensor = tokens_tensor.to(device)
    attention_mask = attention_mask.to(device)
    # forward the tokens tensor and the attention mask to the BERT model
    with torch.no_grad():
        bert_model = bert_model.to(device)
        outputs = bert_model(tokens_tensor, attention_mask = attention_mask)
        bert_model = bert_model.to('cpu')
        hidden_states = outputs.hidden_states
        hidden_states = torch.stack(hidden_states, dim=0)
        hidden_states = hidden_states.to('cpu')
        hidden_states = hidden_states.permute(1,2,0,3)
    del tokens_tensor, attention_mask
    sentence_embeddings = []
    # for each sentence, we sum the last four layers of the embeddings for each word
    for i, token_embeddings in enumerate(hidden_states):
        token_vecs_sum = []
        for j, token in enumerate(token_embeddings):
            if mappings[i][j] is None:
                continue
            sum_vec = torch.sum(token[-4:], dim=0)
            token_vecs_sum.append(sum_vec)
        sentence_embeddings.append(token_vecs_sum)
    del hidden_states, mappings
    # pad sentence embeddings to block size specified in the config
    for i, sentence_embedding in enumerate(sentence_embeddings):
        sentence_embeddings[i] += [torch.zeros(sentence_embedding[0].size(0))] * (config['block_size'] - len(sentence_embedding))
    sentence_embeddings = torch.stack([torch.stack(sentence) for sentence in sentence_embeddings])
    return sentence_embeddings

def create_mapping(sentence, tokens_tensor, tokenizer, cased = False, subword_manage = 'prefix'):
    """ 
    Function to create the mapping from the tokenized sentence to the detokenized sentence
    The function takes the sentence, the tokens tensor, the tokenizer, and the casing and subword management options
    It is essentially useful to calculate the word embeddings from the BERT model so that we get the desired subword
    """
    
    mapping = [None] # for the [CLS] token
    detokenized_index = 1 # for the [CLS] token
    # detokenize the sentence
    detokenized_sentence = tokenizer.convert_ids_to_tokens(tokens_tensor)
    for i, word in enumerate(sentence):
        word = word if cased else word.lower()
        # get the detokenized token
        detokenized_token = tokenizer.convert_ids_to_tokens(tokens_tensor[detokenized_index].item())
        # if the word is not equal to the detokenized token, we need to reconstruct the word
        if word != detokenized_token:
            reconstructed_word = detokenized_token
            # while the reconstructed word is not equal to the word, we need to add the index to the mapping
            while reconstructed_word != word:
                detokenized_index += 1
                reconstructed_word += detokenized_sentence[detokenized_index].strip('##')
                mapping.append(i)
            mapping.append(i)
        else:
            mapping.append(i)
        # move to the next token
        detokenized_index += 1
    # if the mapping is shorter than the detokenized sentence, we need to add None to the mapping
    while len(mapping) < len(detokenized_sentence):
        mapping.append(None)
    # prefix case: we keep the first subword of the word and discard the rest
    if subword_manage == 'prefix':
        for i, idx in enumerate(mapping):
            if idx is not None:
                j = i+1
                while mapping[j] == idx:
                    mapping[j] = None
                    j += 1
    # suffix case: we keep the last subword of the word and discard the rest
    elif subword_manage == 'suffix':
        for i, idx in enumerate(mapping):
            if idx is not None:
                j = i
                while mapping[j+1] == idx:
                    mapping[j] = None
                    j += 1
    return mapping

def load_BERT_encoder(model_name, device = 'cuda'):
    """ Returns the BERT model and tokenizer specified in the config file. """
    model = AutoModel.from_pretrained(model_name, output_hidden_states = True)
    model = model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # add a special marker tokens for PMB
    special_tokens_dict = {'additional_special_tokens': ['<M>']}
    # the Dutch BERT model lacks some special tokens
    if model_name == BERT_FINDER['nl']:
        special_tokens_dict = {'additional_special_tokens': ['<M>', '~', '\u20AC', '[', ']']}
    _ = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))
    return model, tokenizer

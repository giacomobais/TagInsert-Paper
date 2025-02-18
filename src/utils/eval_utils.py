import json
import pandas as pd
import torch
import wandb
import numpy as np
from torch.autograd import Variable
from tqdm import tqdm
from utils.train_utils import get_dataloaders, load_BERT_encoder, dataload, pad_mask, subsequent_mask


PROP_CONVERTER = {1: "100%", 0.75: "75%", 0.5: "50%", 0.25: "25%", 0.1: "10%"}

@torch.no_grad()
def evaluate(model, config, tagging):
    """
    Function that performs evaluation over the validation or set test. The function calculates accuracy for the model
    and writes the predictions to a file. The evaluation run is also tracked by wandb.
    """
    model.eval()
    prop_path = PROP_CONVERTER[config['data_proportion']]
    # loads mappings and dataloader depending on the tagging task and proportion experiment
    idx_to_tgt = json.load(open(f'data/{tagging}/processed/{prop_path}/idx_to_{tagging}.json'))
    tgt_to_idx = json.load(open(f'data/{tagging}/processed/{prop_path}/{tagging}_to_idx.json'))
    idx_to_word = json.load(open(f'data/{tagging}/processed/{prop_path}/idx_to_word.json'))
    _, val_dataloader, _, len_val = get_dataloaders(f"data/{tagging}/processed/{prop_path}/", config, shuffle=False)
    # naming for wandb tracking
    if config['model_name'] == "VanillaTransformer":
        run_model_name = "VT"
    elif config['model_name'] == "TagInsert":
        run_model_name = "TI"
    elif config['model_name'] == "BERT":
        run_model_name = "BE"
    elif config['model_name'] == "TagInsertL2R":
        run_model_name = "TIL2R"
    run_name = run_model_name + "_" + tagging + "_" + str(config['data_proportion']) + "_eval"
    # start evaluation run
    with wandb.init(project="TagInsert", config=config, name=run_name):
        # load BERT model and tokenizer
        bert_model, tokenizer = load_BERT_encoder(config['bert_model'], config['device'])
        wandb.watch(model, log="all")
        data_iter = dataload(val_dataloader, config, bert_model, tokenizer, tgt_to_idx, pad=0)
        test_predictions = []
        test_targets = []
        test_words = []
        test_orders = []

        # Iterate through the dataset
        with tqdm(data_iter, total=len_val // config['batch_size'], desc="eval") as pbar:
            for i, batch in enumerate(pbar):
                # decoding works differently for each model
                if config['model_name'] == "VanillaTransformer":
                    trg = greedy_decode_VT(model, batch.src, batch.src_mask, batch.embs, config['block_size'], tgt_to_idx['<START>'])
                elif config['model_name'] == "TagInsert":
                    trg, orders = greedy_decode_TI(model, batch.src, batch.src_mask, batch.embs, config['block_size'], batch.seq_lengths, tgt_to_idx, config)
                    # we track ordering of insertions for TI
                    orders = orders.squeeze(1)
                    for k, sent in enumerate(orders):
                        sentence_len = batch.seq_lengths[k]
                        test_orders.append(sent[:sentence_len])
                elif config['model_name'] == "TagInsertL2R":
                    trg = greedy_decode_TIL2R(model, batch.src, batch.src_mask, batch.embs, config['block_size'], tgt_to_idx['<START>'])
                # convert indices to words and format them for writing to file
                trg = trg.squeeze(1)
                trg = [[idx_to_tgt[str(idx.item())] for idx in sentence] for sentence in trg]
                # remove paddings and special tokens
                for j in range(len(trg)):
                    sentence_len = batch.seq_lengths[j]
                    trg[j] = trg[j][1:sentence_len+1]
                test_predictions += trg
                test_words += [[idx_to_word[str(idx.item())] for idx in sentence] for sentence in batch.src]
                for j, idx in enumerate(batch.tgt_y):
                    sentence = [idx_to_tgt[str(idx.item())] for idx in idx]
                    sentence_len = batch.seq_lengths[j]
                    test_targets.append(sentence[:sentence_len])

        # Calculate sentence-level accuracy and write predictions to file
        correct = 0
        total = 0
        i = 0
        with open(f"predictions/{config['model_name']}/{tagging}/predictions_{config['model_name']}_{config['data_proportion']}.csv", "w") as f:
            for words, gold, pred in zip(test_words, test_targets, test_predictions):
                # Calculate sentence-level accuracy
                sentence_correct = sum(1 for g, p in zip(gold, pred) if g == p)
                sentence_total = len(gold)
                sentence_accuracy = sentence_correct / sentence_total
                # Accumulate overall accuracy stats
                correct += sentence_correct
                total += sentence_total
                # Write predictions to file, format depends on the model as TI has ordering
                if config['model_name'] == "TagInsert":
                    order = test_orders[i].tolist()
                    for j, (w, g, p, o) in enumerate(zip(words, gold, pred, order)):
                        f.write(f"{w}|{g}|{p}|{o}")
                        if j < len(words) - 1:
                            f.write(",")
                else:
                    for j, (w, g, p) in enumerate(zip(words, gold, pred)):
                        f.write(f"{w}|{g}|{p}")
                        if j < len(words) - 1:
                            f.write(",")
                f.write(f"\n Sentence accuracy: {sentence_accuracy}\n")
                i+=1
                # Calculate overall accuracy
            overall_accuracy = correct / total
            f.write(f"Overall accuracy: {overall_accuracy}\n")

        # Log overall accuracy and finish wandb
        wandb.log({"Accuracy": overall_accuracy})
        wandb.finish()

def greedy_decode_VT(model, src, src_mask, embs, max_len, start_symbol):
    """
    Function that implements greedy decoding for the Vanilla Transformer model.
    """
    # forward pass through the encoder
    memory = model.encode(src, src_mask, embs)
    # prepare the initial input for the decoder, containing only the start symbol
    ys = torch.ones(src.size(0), 1).fill_(start_symbol).type_as(src.data)
    # iterate through the decoder for the maximum length, each iteration predicting the next word
    for _ in range(max_len - 1):
        # forward pass through the decoder
        out = model.decode(memory, src_mask, ys, subsequent_mask(ys.size(1)).type_as(src.data))
        # generate probabilities for the next word
        prob = model.generator(out[:, -1])
        # greedy decode
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.view(src.size(0), 1)
        # expand the output tensor with the predicted word
        ys = torch.cat((ys, next_word), dim=1)
    return ys

def greedy_decode_TI(model, src, src_mask, embs, max_len, seq_length, tgt_to_idx, config):
    """
    Function that implements greedy decoding for the TagInsert model.
    """
    # forward pass through the encoder
    memory = model.encode(src, src_mask, embs)
    batch_size = src.size(0)
    # prepare the initial input for the decoder, containing only <UNK> special tokens to be replaced
    ys = torch.full((batch_size, max_len), tgt_to_idx['<UNK>']).type_as(src.data)
    # add the start and end tokens to the input, as well as padding depending on the words sequence length
    ys[:, 0] = tgt_to_idx['<START>']
    for i in range(batch_size):
        ys[i][seq_length[i]+1] = tgt_to_idx['<END>']
        ys[i][seq_length[i]+2:] = tgt_to_idx['<PAD>']
    # track the positions where insertions have been made, so no more insertions are made in the same position
    done_positions = torch.zeros_like(ys).to(config['device'])
    # initialize the orderings tensor to track the order of insertions
    orderings = torch.zeros((batch_size, max_len), dtype = torch.int64)
    # iterate through the decoder for the maximum length, each iteration predicting the next word
    for pred in range(max_len):
        # forward pass through the decoder
        out = model.decode(memory, src, embs, src_mask, Variable(ys), Variable(pad_mask(tgt_to_idx['<PAD>'], ys)))
        # generate probabilities for the next word
        prob = model.generator(out)
        # mask the probabilities for the positions where insertions have already been made and for padding, start and end tokens
        prob[:, 0, :] = float('-inf')
        for i in range(batch_size):
            prob[i, seq_length[i]+1:, :] = float('-inf')
        for j, sent in enumerate(done_positions):
            for i, pos in enumerate(sent):
                if pos == 1:
                    prob[j, i, :] = float('-inf')

        probs = prob.to('cpu').detach().numpy()
        # unravel logits to get the location and tag of the maximum probability
        result = [np.unravel_index(np.argmax(r), r.shape) for r in probs]
        # update the output tensor with the predicted word and the orderings tensor. Also update the done_positions tensor
        for i, (location, tag) in enumerate(result):
            if torch.sum(done_positions[i]) != seq_length[i]:
                done_positions[i, location] = 1
                ys[i, location] = tag
                orderings[i, location-1] = pred+1
    return ys, orderings

def greedy_decode_TIL2R(model, src, src_mask, embs, max_len, start_symbol):
    """
    Function that implements greedy decoding for the TagInsertL2R model.
    """
    # forward pass through the encoder
    memory = model.encode(src, src_mask, embs)
    # prepare the initial input for the decoder, containing only the start symbol
    ys = torch.ones(src.size(0), 1).fill_(start_symbol).type_as(src.data)
    # iterate through the decoder for the maximum length, each iteration predicting the next word
    for _ in range(max_len - 1):
        # forward pass through the decoder
        out = model.decode(memory, None, embs, src_mask, ys, subsequent_mask(ys.size(1)).type_as(src.data))
        # generate probabilities for the next word
        prob = model.generator(out[:, -1])
        # greedy decode
        _, next_word = torch.max(prob, dim=1)
        # expand the output tensor with the predicted word
        next_word = next_word.view(src.size(0), 1)
        ys = torch.cat((ys, next_word), dim=1)
    return ys
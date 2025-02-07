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
    model.eval()
    prop_path = PROP_CONVERTER[config['data_proportion']]
    if tagging == "POS":
        idx_to_tgt = json.load(open(f'data/POS/processed/{prop_path}/idx_to_POS.json'))
        tgt_to_idx = json.load(open(f'data/POS/processed/{prop_path}/POS_to_idx.json'))
        idx_to_word = json.load(open(f'data/POS/processed/{prop_path}/idx_to_word.json'))
        _, val_dataloader, _, len_val = get_dataloaders(f"data/POS/processed/{prop_path}/", config, shuffle=False)
    elif tagging == "CCG":
        idx_to_tgt = json.load(open(f'data/CCG/processed/{prop_path}/idx_to_CCG.json'))
        tgt_to_idx = json.load(open(f'data/CCG/processed/{prop_path}/CCG_to_idx.json'))
        idx_to_word = json.load(open(f'data/CCG/processed/{prop_path}/idx_to_word.json'))
        _, val_dataloader, _, len_val = get_dataloaders(f"data/CCG/processed/{prop_path}/", config, shuffle=False)
    if config['model_name'] == "VanillaTransformer":
        run_model_name = "VT"
    elif config['model_name'] == "TagInsert":
        run_model_name = "TI"
    elif config['model_name'] == "BERT":
        run_model_name = "BE"
    elif config['model_name'] == "TagInsertL2R":
        run_model_name = "TIL2R"
    run_name = run_model_name + "_" + tagging + "_" + str(config['data_proportion']) + "_eval"
    with wandb.init(project="TagInsert", config=config, name=run_name):
        bert_model, tokenizer = load_BERT_encoder(config['bert_model'], config['device'])
        wandb.watch(model, log="all")
        data_iter = dataload(val_dataloader, config, bert_model, tokenizer, tgt_to_idx, pad=0)
        
        test_predictions = []
        test_targets = []
        test_words = []

        # Iterate through the dataset
        with tqdm(data_iter, total=len_val // config['batch_size'], desc="eval") as pbar:
            for i, batch in enumerate(pbar):
                if config['model_name'] == "VanillaTransformer":
                    trg = greedy_decode_VT(model, batch.src, batch.src_mask, batch.embs, config['block_size'], tgt_to_idx['<START>'])
                elif config['model_name'] == "TagInsert":
                    trg, orders = greedy_decode_TI(model, batch.src, batch.src_mask, batch.embs, config['block_size'], batch.seq_lengths, tgt_to_idx, config)
                elif config['model_name'] == "TagInsertL2R":
                    trg = greedy_decode_TIL2R(model, batch.src, batch.src_mask, batch.embs, config['block_size'], tgt_to_idx['<START>'])
                trg = trg.squeeze(1)
                trg = [[idx_to_tgt[str(idx.item())] for idx in sentence] for sentence in trg]
                
                for j in range(len(trg)):
                    sentence_len = batch.seq_lengths[j]
                    trg[j] = trg[j][1:sentence_len+1]
                
                test_predictions += trg
                test_words += [[idx_to_word[str(idx.item())] for idx in sentence] for sentence in batch.src]
                for j, idx in enumerate(batch.tgt_y):
                    sentence = [idx_to_tgt[str(idx.item())] for idx in idx]
                    sentence_len = batch.seq_lengths[j]
                    test_targets.append(sentence[:sentence_len])

        # Calculate sentence-level accuracy and prepare DataFrame
        data = []
        correct = 0
        total = 0

        for words, gold, pred in zip(test_words, test_targets, test_predictions):
            # Calculate sentence-level accuracy
            sentence_correct = sum(1 for g, p in zip(gold, pred) if g == p)
            sentence_total = len(gold)
            sentence_accuracy = sentence_correct / sentence_total

            # Accumulate overall accuracy stats
            correct += sentence_correct
            total += sentence_total

            # Add row to data
            data.append({
                "Words": " ".join(words),
                "Gold": " ".join(gold),
                "Predicted": " ".join(pred),
                "Accuracy": sentence_accuracy
            })

        # Create a DataFrame
        df = pd.DataFrame(data)

        df.to_excel("predictions.xlsx", index=False)

        # Calculate overall accuracy
        overall_accuracy = correct / total
        print("Overall Accuracy:", overall_accuracy)

        # Log overall accuracy and finish wandb
        wandb.log({"Accuracy": overall_accuracy})
        wandb.finish()

def greedy_decode_VT(model, src, src_mask, embs, max_len, start_symbol):
    memory = model.encode(src, src_mask, embs)
    ys = torch.ones(src.size(0), 1).fill_(start_symbol).type_as(src.data)
    for _ in range(max_len - 1):
        out = model.decode(
            memory, src_mask, ys, subsequent_mask(ys.size(1)).type_as(src.data)
        )
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.view(src.size(0), 1)
        ys = torch.cat((ys, next_word), dim=1)
    return ys

def greedy_decode_TI(model, src, src_mask, embs, max_len, seq_length, tgt_to_idx, config):
    memory = model.encode(src, src_mask, embs)
    batch_size = src.size(0)
    ys = torch.full((batch_size, max_len), tgt_to_idx['<UNK>']).type_as(src.data)
    ys[:, 0] = tgt_to_idx['<START>']
    for i in range(batch_size):
      ys[i][seq_length[i]+1] = tgt_to_idx['<END>']
      ys[i][seq_length[i]+2:] = tgt_to_idx['<PAD>']
    done_positions = torch.zeros_like(ys).to(config['device'])
    orderings = torch.zeros((batch_size, max_len), dtype = torch.int64)
    for pred in range(max_len):
        out = model.decode(memory, src, embs, src_mask,
                           Variable(ys),
                           Variable(pad_mask(tgt_to_idx['<PAD>'], ys)))

        prob = model.generator(out)
        prob[:, 0, :] = float('-inf')
        for i in range(batch_size):
          prob[i, seq_length[i]+1:, :] = float('-inf')
        for j, sent in enumerate(done_positions):
          for i, pos in enumerate(sent):
            if pos == 1:
              prob[j, i, :] = float('-inf')

        # argmax_indices = torch.argmax(prob, dim = 1)
        probs = prob.to('cpu').detach().numpy()
        result = [np.unravel_index(np.argmax(r), r.shape) for r in probs]
        # location, tag = argmax_indices // POS_VOCAB_SIZE, argmax_indices % POS_VOCAB_SIZE
        for i, (location, tag) in enumerate(result):
          if torch.sum(done_positions[i]) != seq_length[i]:
            done_positions[i, location] = 1
            ys[i, location] = tag
            orderings[i, location-1] = pred+1
            # print(f'{ys}, insertion made for tag {tag.item()} at position {location}.')
            # # print(orderings)
            # print('---------------------------------------')

    return ys, orderings

def greedy_decode_TIL2R(model, src, src_mask, embs, max_len, start_symbol):
    memory = model.encode(src, src_mask, embs)
    ys = torch.ones(src.size(0), 1).fill_(start_symbol).type_as(src.data)
    for _ in range(max_len - 1):
        out = model.decode(
            memory, None, embs, src_mask, ys, subsequent_mask(ys.size(1)).type_as(src.data)
        )
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.view(src.size(0), 1)
        ys = torch.cat((ys, next_word), dim=1)
    return ys
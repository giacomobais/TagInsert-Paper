
import torch
import numpy as np
import json
from tqdm import tqdm
import yaml
from torch import nn
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, DistilBertTokenizer
import wandb
from src.models.VanillaTransformer import make_model_VT
from src.models.TagInsert import make_model_TI

def load_config(config_path):
    with open(config_path) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    return config

def prepare_data(train_file, val_file, test_file, block_size = 52):

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
    torch.save({'words': train_words, 'tags': train_tags, 'original_sentences': train_original_sentences}, 'data/POS/processed/train_data.pth')
    torch.save({'words': val_words, 'tags': val_tags, 'original_sentences': val_original_sentences}, 'data/POS/processed/val_data.pth')
    torch.save({'words': test_words, 'tags': test_tags, 'original_sentences': test_original_sentences}, 'data/POS/processed/test_data.pth')

    with open('data/POS/processed/word_to_idx.json', 'w') as f:
        json.dump(word_to_idx, f)
    with open('data/POS/processed/idx_to_word.json', 'w') as f:
        json.dump(idx_to_word, f)
    with open('data/POS/processed/POS_to_idx.json', 'w') as f:
        json.dump(POS_to_idx, f)
    with open('data/POS/processed/idx_to_POS.json', 'w') as f:
        json.dump(idx_to_POS, f)

class TaggingDataset(Dataset):
    def __init__(self, data):
        data = torch.load(data, weights_only=True)
        self.word_sequences = data['words']
        self.tag_sequences = data['tags']
        self.original_sentences = data['original_sentences']

    def shuffle(self):
        shuffled_indices = torch.randperm(len(self.word_sequences))
        self.word_sequences = [self.word_sequences[i] for i in shuffled_indices]
        self.tag_sequences = [self.tag_sequences[i] for i in shuffled_indices]

    def __len__(self):
        return len(self.word_sequences)

    def __getitem__(self, idx):
        words = torch.tensor(self.word_sequences[idx], dtype=torch.long)
        tags = torch.tensor(self.tag_sequences[idx], dtype=torch.long)
        original_sentence = self.original_sentences[idx]
        return words, tags, original_sentence

def load_BERT_encoder(model_name, device = 'cuda'):
    model = AutoModel.from_pretrained(model_name, output_hidden_states = True)
    model = model.to(device)
    tokenizer = DistilBertTokenizer.from_pretrained(model_name)
    return model, tokenizer

def create_mapping(sentence, tokens_tensor, tokenizer, cased = False, subword_manage = 'prefix'):
    mapping = [None] # for the [CLS] token
    detokenized_index = 1 # for the [CLS] token
    detokenized_sentence = tokenizer.convert_ids_to_tokens(tokens_tensor)
    for i, word in enumerate(sentence):
        word = word if cased else word.lower()
        detokenized_token = tokenizer.convert_ids_to_tokens(tokens_tensor[detokenized_index].item())
        if word != detokenized_token:
            reconstructed_word = detokenized_token
            while reconstructed_word != word:
                detokenized_index += 1
                reconstructed_word += detokenized_sentence[detokenized_index].strip('##')
                mapping.append(i)
            mapping.append(i)
        else:
            mapping.append(i)
        detokenized_index += 1
    while len(mapping) < len(detokenized_sentence):
        mapping.append(None)
    if subword_manage == 'prefix':
        for i, idx in enumerate(mapping):
            if idx is not None:
                j = i+1
                while mapping[j] == idx:
                    mapping[j] = None
                    j += 1
    elif subword_manage == 'suffix':
        for i, idx in enumerate(mapping):
            if idx is not None:
                j = i
                while mapping[j+1] == idx:
                    mapping[j] = None
                    j += 1
    return mapping

def extract_BERT_embs(sentences, bert_model, tokenizer, config):
    device = config['device']
    # Add the special tokens.
    marked_text = [" ".join(sentence) for sentence in sentences]
    # use tokenizer to get tokenized text and pad up to 100
    tokens_tensor = [tokenizer(text, padding="max_length", truncation=True, max_length=config['bert_block_size'], return_tensors="pt") for text in marked_text]
    attention_mask = torch.stack([t['attention_mask'] for t in tokens_tensor])
    tokens_tensor = torch.stack([t['input_ids'] for t in tokens_tensor]).squeeze(1)
    mappings = [create_mapping(sentence, tokens_tensor[i], tokenizer, cased = True, subword_manage='prefix') for i, sentence in enumerate(sentences)]
    tokens_tensor = tokens_tensor.to(device)
    attention_mask = attention_mask.to(device)
    with torch.no_grad():
        bert_model = bert_model.to(device)
        outputs = bert_model(tokens_tensor, attention_mask = attention_mask)
        bert_model = bert_model.to('cpu')
        hidden_states = outputs[1]
        hidden_states = torch.stack(hidden_states, dim=0)
        hidden_states = hidden_states.to('cpu')
        hidden_states = hidden_states.permute(1,2,0,3)
    del tokens_tensor, attention_mask
    sentence_embeddings = []
    for i, token_embeddings in enumerate(hidden_states):
        token_vecs_sum = []
        for j, token in enumerate(token_embeddings):
            if mappings[i][j] is None:
                continue
            sum_vec = torch.sum(token[-4:], dim=0)
            token_vecs_sum.append(sum_vec)
        sentence_embeddings.append(token_vecs_sum)
    del hidden_states, mappings
    # pad sentence embeddings to 100
    for i, sentence_embedding in enumerate(sentence_embeddings):
        sentence_embeddings[i] += [torch.zeros(sentence_embedding[0].size(0))] * (config['block_size'] - len(sentence_embedding))

    sentence_embeddings = torch.stack([torch.stack(sentence) for sentence in sentence_embeddings])
    return sentence_embeddings

def rate(step, model_size, factor, warmup):
    """
    we have to default the step to 1 for LambdaLR function
    to avoid zero raising to negative power.
    """
    if step == 0:
        step = 1
    return factor * (
        model_size ** (-0.5) * min(step ** (-0.5), step * warmup ** (-1.5))
    )

def save_model(model, optimizer, lr_scheduler, train_losses, val_losses, epochs, path):
    torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'lr_scheduler' : lr_scheduler.state_dict(),
                'train_losses': train_losses,
                'val_losses': val_losses,
                'epochs': epochs
                }, path)

def load_model(model_name, path, config, tagging):
    if model_name == "VT":
        model = make_model_VT(tagging, d_model = config['hidden_size'], N=config['n_stacks'], h = config['n_heads'])
    elif model_name == "TI":
        model = make_model_TI(tagging, d_model = config['hidden_size'], N=config['n_stacks'], h = config['n_heads'])
    model = model.to(config['device'])
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], betas=(config['betas'][0], config['betas'][1]), eps=config['eps']) 
    lr_scheduler = LambdaLR(optimizer=optimizer,lr_lambda=lambda step: rate(step, model_size=model.src_embed[0].d_model, factor=1.0, warmup=config['warmup']),)
    checkpoint = torch.load(path, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'], strict = True)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
    train_losses = checkpoint['train_losses']
    val_losses = checkpoint['val_losses']
    epochs = checkpoint['epochs']
    logs = {'train_losses': train_losses, 'val_losses': val_losses, 'epochs': epochs}
    return model, optimizer, lr_scheduler, logs

def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1).type(
        torch.uint8
    )
    return subsequent_mask == 0

class Batch_VT:
    """Object for holding a batch of data with mask during training."""

    def __init__(self, src, tgt=None, embs = None, pad=0):  # 2 = <blank>
        self.src = src
        self.src_mask = (src != pad).unsqueeze(-2)
        self.embs = embs
        if tgt is not None:
            self.seq_lengths = [torch.nonzero(tgt[i] == 0, as_tuple = False)[0,0].item()-1 if torch.nonzero(tgt[i] == 0, as_tuple = False).numel() > 0 else 51 for i in range(tgt.size(0))]
            self.tgt = tgt[:, :-1]
            self.tgt_y = tgt[:, 1:]
            self.tgt_mask = self.make_std_mask(self.tgt, pad)
            self.ntokens = (self.tgt_y != pad).data.sum()

    @staticmethod
    def make_std_mask(tgt, pad):
        "Create a mask to hide padding and future words."
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & subsequent_mask(tgt.size(-1)).type_as(
            tgt_mask.data
        )
        return tgt_mask

class Batch_TI:
    "Object for holding a batch of data with mask during training."
    def __init__(self, config, tgt_map, src, trg=None, embs = None, pad=0):
        self.src = src
        self.src_mask = (src != pad).unsqueeze(-2)
        if trg is not None:
            # extracting length of each sentence
            first_pads = [torch.nonzero(trg[i] == 0, as_tuple = False)[0,0].item()-1 if torch.nonzero(trg[i] == 0, as_tuple = False).numel() > 0 else config['block_size']-1 for i in range(trg.size(0))]
            self.sequence_lengths = first_pads
            new_trg = torch.zeros((src.size(0), config['block_size']), dtype = torch.int64).to(config['device'])
            new_trg_y = torch.zeros((src.size(0), config['block_size']), dtype = torch.int64).to(config['device']) # for each slot, the tokens yet to insert
            self.trajectory = []
            self.inserted = torch.zeros(src.size(0), dtype = torch.int64)
            for i, train in enumerate(trg):
              # current_sep = self.sep[i].item()
              all_ixs = torch.arange(start = 1, end = first_pads[i]+1)
              permuted_ixs = torch.randperm(all_ixs.size(0))
              permuted_all_ixs = all_ixs[permuted_ixs]
              self.trajectory.append(permuted_all_ixs)
              # constructing actual y to be forwarded
              # vec = torch.ones(BLOCK_SIZE).to(device)
              vec = torch.full((config['block_size'],), tgt_map['<UNK>'])
              targets = torch.zeros(config['block_size']).to(config['device'])
              vec[0] = tgt_map['<START>']
              vec[self.sequence_lengths[i]+1] = tgt_map['<END>']
              vec[self.sequence_lengths[i]+2:] = tgt_map['<PAD>']
              new_trg[i] = vec
              ins = 0
              for j, ix in enumerate(vec):
                if ix == tgt_map['<UNK>']:
                  targets[ins] = train[j-1]
                  ins+=1
              new_trg_y[i] = targets
            self.trg = new_trg
            self.trg_y = new_trg_y
            self.trg_mask = self.make_std_mask(self.trg, pad)
            nonpads = (self.trg_y != pad).data.sum()
            self.ntokens = nonpads

            # print('trajectory:', self.trajectory)
            self.embs = embs

    def next_trajectory(self):
        for i, ins in enumerate(self.inserted):
            if ins >= self.sequence_lengths[i]:
              continue
            next_tag_pos = self.trajectory[i][ins].item()
            self.trg[i][next_tag_pos] = self.trg_y[i][next_tag_pos-1]
            self.inserted[i] += 1

    @staticmethod
    def make_std_mask(tgt, pad):
        "Create a mask to hide padding and future words."
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask# & Variable(subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data))
        return tgt_mask

class DummyOptimizer(torch.optim.Optimizer):
    def __init__(self):
        self.param_groups = [{"lr": 0}]
        None

    def step(self):
        None

    def zero_grad(self, set_to_none=False):
        None


class DummyScheduler:
    def step(self):
        None

class LabelSmoothing(nn.Module):
    "Implement label smoothing."

    def __init__(self, size, padding_idx, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(reduction="sum")
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None

    def forward(self, x, target):
        assert x.size(1) == self.size
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(x, true_dist.clone().detach())


class LossCompute_VT:
    "A simple loss compute and train function."

    def __init__(self, generator, criterion):
        self.generator = generator
        self.criterion = criterion

    def __call__(self, x, y, norm):
        x = self.generator(x)
        sloss = (
            self.criterion(
                x.contiguous().view(-1, x.size(-1)), y.contiguous().view(-1)
            )
            / norm
        )
        return sloss.data * norm, sloss
    
class TI_Loss(nn.Module):
  def __init__(self, config, tgt_map):
    super(TI_Loss, self).__init__()
    self.tgt_map = tgt_map
    self.config = config

  def sentence_loss(self, logits, forwarded_trgs, targets, sequence_length):
    # logits has shape (51, 52), they are the logits for all slots of the i-th sentence
    # targets has shape (51, 52), they are the actual tokens for each span, only the first k+1 spans are filled
    # k = len(spans_del) # also known as k
    losses = []
    # extracting the number of tokens in a span L
    for i, ix in enumerate(forwarded_trgs):
      if ix == self.tgt_map['<PAD>']:
        break
      if ix == self.tgt_map['<UNK>']:
        tag_to_insert = targets[i-1]
        # print(tag_to_insert)
        p = logits[i][tag_to_insert]
        losses.append(-torch.log(p))
        # print('---------------------')
    if len(losses) == 0:
      tag_to_insert = self.tgt_map['<END>']
      # print(tag_to_insert, sequence_length+2)
      p = logits[sequence_length+2, tag_to_insert]
      losses.append(-torch.log(p))
    out = torch.mean(torch.stack(losses))
    return out

  def forward(self, logits, forwarded_trgs, targets, sequence_lengths, inserted):
    # logits will have shape (256, 51, 52), 2nd dim are slots and 3rd dim is the vocab
    # targets will have shape (256, 50), 1st dim are the k+1 slots to which loss needs to be computed, while 2nd dim are the words to be inserted in that slot
    # ixs will have shape (256, 50), for each sentence the indeces of the spans, k can retrieve how many
    # sep has shape (256), one k for each sentence
    batch_losses = []
    for i, _ in enumerate(targets):
        if inserted[i] >= sequence_lengths[i]:
          continue
      # if i == 0:
        # slot_losses = torch.zeros(k.item()+1).to(device) #k+1 slots
        # spans_del = ixs[i, :k] # the k indeces delimitating the spans 1-2-6
        batch_loss = self.sentence_loss(logits[i, :, :], forwarded_trgs[i, :], targets[i, :], sequence_lengths[i])
        # print("loss of a single sequence: ",batch_loss)
        batch_losses.append(batch_loss)
    loss = torch.mean(torch.stack(batch_losses))
    # print(loss)
    return loss

class LossCompute_TI:
    "A simple loss compute and train function."
    def __init__(self, generator, criterion, opt=None):
        self.generator = generator
        self.criterion = criterion
        self.opt = opt

    def __call__(self, x, forwarded_y, y, sequence_lengths, norm, inserted):
        # SLOTS = 51, VOCAB = 52
        x = self.generator(x) # shape (B, S * V)
        if norm == 0:
          norm = 1
        loss = self.criterion(x, forwarded_y, y, sequence_lengths, inserted)
        return loss.data * norm, loss

def resume_training(model_path, config, model_name, tagging):
    try:
        model, model_opt, lr_scheduler, logs = load_model(model_name, model_path, config)
        trained_epochs = logs['epochs']
        train_losses = logs['train_losses']
        val_losses = logs['val_losses']
        print('Loading a pre-existent model.')
    except:
        if model_name == "VT":
            model = make_model_VT(tagging, d_model = config['hidden_size'], N=config['n_stacks'], h = config['n_heads'])
        elif model_name == "TI":
            model = make_model_TI(tagging, d_model = config['hidden_size'], N=config['n_stacks'], h = config['n_heads'])
        model_opt = torch.optim.Adam(model.parameters(), lr=config['lr'], betas=(config['betas'][0], config['betas'][1]), eps=config['eps']) 
        lr_scheduler = LambdaLR(optimizer=model_opt,lr_lambda=lambda step: rate(step, model_size=model.src_embed[0].d_model, factor=1.0, warmup=config['warmup']),)
        trained_epochs = 0
        train_losses = []
        val_losses = []
        print('No pre-trained model found.')

    model.train()
    model = model.to(config['device'])
    return model, model_opt, lr_scheduler, trained_epochs, train_losses, val_losses

class TrainState:
    """Track number of steps, examples, and tokens processed"""

    step: int = 0  # Steps in the current epoch
    accum_step: int = 0  # Number of gradient accumulation steps
    samples: int = 0  # total # of examples used
    tokens: int = 0  # total # of tokens processed
    eval_step: int = 0  # Steps in the current evaluation


def dataload(data_loader, config, bert_model, tokenizer, tgt_map, pad=0):
    for batch_data in data_loader:
        xb, yb, original_sentences = batch_data
        embs = extract_BERT_embs(original_sentences, bert_model, tokenizer, config)
        src = xb.to(config['device'])
        tgt = yb.to(config['device'])
        embs = embs.to(config['device'])
        if config['model_name'] == "VanillaTransformer":
            yield Batch_VT(src, tgt, pad=pad)
        elif config['model_name'] == "TagInsert":
            yield Batch_TI(config, tgt_map, src, tgt, embs, pad=pad)


def run_epoch_VT(data_iter, model, loss_compute, optimizer, lr_scheduler, config, data_len, mode = 'train', accum_iter = 1, train_state = TrainState()):
    total_tokens = 0
    total_loss = 0
    losses = []
    log_id = "Batch train loss" if mode == "train" else "Batch val loss"
    with tqdm(data_iter, total = data_len // config['batch_size'], desc=mode) as pbar:
        for i, batch in enumerate(pbar):
            out = model.forward(batch.src, batch.tgt, batch.src_mask, batch.tgt_mask, batch.embs)
            loss, loss_node = loss_compute(out, batch.tgt_y, batch.ntokens)
            losses.append(loss_node.item())
            if mode == "train":
                loss_node.backward()
                train_state.step += 1
                train_state.samples += batch.src.size(0)
                train_state.tokens += batch.ntokens
                if (i + 1) % accum_iter == 0:
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    train_state.accum_step += 1
                wandb.log({log_id: loss_node.item(), "train_step": train_state.step})
                lr_scheduler.step()
            else:
                train_state.eval_step += 1
                wandb.log({log_id: loss_node.item(), "val_step": train_state.eval_step})
            total_loss += loss
            total_tokens += batch.ntokens
            pbar.set_postfix(loss=loss_node.item())

    return total_loss / total_tokens, train_state, losses

def run_epoch_TI(data_iter, model, loss_compute, optimizer, lr_scheduler, config, data_len, mode = 'train', accum_iter = 1, train_state = TrainState()):
    total_tokens = 0
    total_loss = 0
    current_losses = []
    log_id = "Batch train loss" if mode == "train" else "Batch val loss"
    with tqdm(data_iter, total = data_len // config['batch_size'], desc=mode) as pbar:
        for i, batch in enumerate(pbar):
            max_len = np.max(batch.sequence_lengths) - 1
            losses = torch.zeros(max_len).to(config['device'])
            for traj_step in range(max_len):
                # print(f'Calculating loss for trajectory {traj_step}.')
                out = model.forward(batch.src, batch.trg,
                                    batch.src_mask, batch.trg_mask, batch.embs)
                _, loss_node = loss_compute(out, batch.trg, batch.trg_y, batch.sequence_lengths, batch.ntokens, batch.inserted)
                # loss_node = loss_node / accum_iter
                if mode == "train":
                    # loss_node = Variable(loss_node, requires_grad = True)
                    loss_node.backward()
                    train_state.samples += batch.src.shape[0]
                    train_state.tokens += batch.ntokens
                    if i % accum_iter == 0:
                        optimizer.step()
                        optimizer.zero_grad(set_to_none=True)
                    lr_scheduler.step()
                norm = batch.ntokens
                if norm == 0:
                    norm = 1
                losses[traj_step] = loss_node.item()
                batch.next_trajectory()
            del batch.embs
            loss_node_backward = torch.mean(losses)
            if mode == "train":
                train_state.step += 1
                wandb.log({log_id: loss_node_backward.item(), "train_step": train_state.step})
            else:
                train_state.eval_step += 1
                wandb.log({log_id: loss_node_backward.item(), "val_step": train_state.eval_step})
            current_losses.append(loss_node_backward.item())
            total_loss += loss_node_backward.data
            total_tokens += batch.ntokens
            pbar.set_postfix(loss=loss_node_backward.item())
    return total_loss / total_tokens, train_state, current_losses

def collate_fn(batch):
    words, tags, original_sentences = zip(*batch)
    words = torch.stack(words)
    tags = torch.stack(tags)
    return words, tags, original_sentences

def get_dataloaders(data_path, config, shuffle = True):
    train_dataset = TaggingDataset(data_path + "train_data.pth")
    val_dataset = TaggingDataset(data_path + "val_data.pth")
    train_dataloader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=shuffle, collate_fn=collate_fn)
    val_dataloader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=shuffle, collate_fn=collate_fn)
    len_train = len(train_dataset)
    len_val = len(val_dataset)
    return train_dataloader, val_dataloader, len_train, len_val

def define_wandb_metrics():
    wandb.define_metric("train_step")
    wandb.define_metric("Batch train loss", step_metric="train_step")
    wandb.define_metric("val_step")
    wandb.define_metric("Batch val loss", step_metric="val_step")
    wandb.define_metric("epoch")
    wandb.define_metric("train_loss", step_metric="epoch")
    wandb.define_metric("val_loss", step_metric="epoch")

def train(model_package, config, tagging, save = True):
    model, model_opt, lr_scheduler, trained_epochs, train_losses, val_losses = model_package
    with wandb.init(project="TagInsert", config=config, name = config['model_name']+ "_" + tagging):
        bert_model, tokenizer = load_BERT_encoder(config['bert_model'], config['device'])
        tgt_map = json.load(open(f'data/{tagging}/processed/{tagging}_to_idx.json'))
        train_dataloader, val_dataloader, len_train, len_val = get_dataloaders(f"data/{tagging}/processed/", config)
        if config['model_name'] == "VanillaTransformer":
            criterion = LabelSmoothing(size=len(tgt_map), padding_idx=0, smoothing=0.0)
        elif config['model_name'] == "TagInsert":
            criterion = TI_Loss(config, tgt_map)
        train_state = TrainState()
        define_wandb_metrics()
        wandb.watch(model, log="all")
        all_train_losses = []
        all_val_losses = []
        for epoch in range(trained_epochs, config['epochs']):
            print(f"Epoch {epoch}")
            epoch_data_iter = dataload(train_dataloader, config, bert_model, tokenizer, tgt_map, pad=0)
            val_epoch_data_iter = dataload(val_dataloader, config, bert_model, tokenizer, tgt_map, pad=0)
            model.train()
            if config['model_name'] == "VanillaTransformer":
                train_losses = run_epoch_VT(epoch_data_iter, model, LossCompute_VT(model.generator, criterion), model_opt, lr_scheduler, config, len_train, mode = 'train', train_state=train_state)[2]
            elif config['model_name'] == "TagInsert":
                train_losses = run_epoch_TI(epoch_data_iter, model, LossCompute_TI(model.generator, criterion), model_opt, lr_scheduler, config, len_train, mode = 'train', train_state=train_state)[2]
            all_train_losses += train_losses
            with torch.no_grad():
                model.eval()
                if config['model_name'] == "VanillaTransformer":
                    val_losses = run_epoch_VT(val_epoch_data_iter, model, LossCompute_VT(model.generator, criterion), DummyOptimizer(),DummyScheduler(), config, len_val, mode = 'eval', train_state=train_state)[2]
                elif config['model_name'] == "TagInsert":
                    val_losses = run_epoch_TI(val_epoch_data_iter, model, LossCompute_TI(model.generator, criterion), DummyOptimizer(),DummyScheduler(), config, len_val, mode = 'eval', train_state=train_state)[2]
                all_val_losses += val_losses
            wandb.log({"train_loss": np.mean(train_losses) , "val_loss": np.mean(val_losses), "epoch": epoch})
    wandb.finish()
    if save:
        model_name = config['model_name'] + "_" + tagging
        save_model(model, model_opt, lr_scheduler, all_train_losses, all_val_losses, epoch, f"models/{model_name}")
    return model

@torch.no_grad()
def evaluate(model, config, tagging):
    model.eval()
    if tagging == "POS":
        idx_to_tgt = json.load(open('data/POS/processed/idx_to_POS.json'))
        tgt_to_idx = json.load(open('data/POS/processed/POS_to_idx.json'))
        _, val_dataloader, _, len_val = get_dataloaders("data/POS/processed/", config, shuffle = False)
    elif tagging == "CCG":
        idx_to_tgt = json.load(open('data/CCG/processed/idx_to_CCG.json'))
        tgt_to_idx = json.load(open('data/CCG/processed/CCG_to_idx.json'))
        _, val_dataloader, _, len_val = get_dataloaders("data/CCG/processed/", config, shuffle = False)
    with wandb.init(project="TagInsert", config=config, name = f"{config['model_name']}_{tagging}_eval"):
        bert_model, tokenizer = load_BERT_encoder(config['bert_model'], config['device'])
        wandb.watch(model, log="all")
        data_iter = dataload(val_dataloader, config, bert_model, tokenizer, pad=0)
        test_predictions = []
        test_targets = []
        with tqdm(data_iter, total = len_val // config['batch_size'], desc="eval") as pbar:
            for i, batch in enumerate(pbar):
                if config['model_name'] == "VanillaTransformer":
                    trg = greedy_decode_VT(model, batch.src, batch.src_mask, batch.embs, config['block_size'], tgt_to_idx['<START>'])
                elif config['model_name'] == "TagInsert":
                    trg = greedy_decode_TI(model, batch.src, batch.src_mask, batch.embs, config['block_size'], tgt_to_idx['<START>'])
                # aligning predictions with targets while removing padding
                trg = trg.squeeze(1)
                trg = [[idx_to_tgt[str(idx.item())] for idx in sentence] for sentence in trg]
                for j in range(len(trg)):
                    sentence_len = batch.seq_lengths[j]
                    trg[j] = trg[j][1:sentence_len+1]
                test_predictions += trg
                for j, idx in enumerate(batch.tgt_y):
                    sentence = [idx_to_tgt[str(idx.item())] for idx in idx]
                    sentence_len = batch.seq_lengths[j]
                    test_targets.append(sentence[:sentence_len])

        # calculate accuracy
        correct = 0
        total = 0
        for i in range(len(test_predictions)):
            for j in range(len(test_predictions[i])):
                total += 1
                if test_predictions[i][j] == test_targets[i][j]:
                    correct += 1

        accuracy = correct / total
        print("Accuracy:", accuracy)
        wandb.log({"Accuracy": accuracy})
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

def greedy_decode_TI(model, src, src_mask, embs, max_len, start_symbol):
        pass
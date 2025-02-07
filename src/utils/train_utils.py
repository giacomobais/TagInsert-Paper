
import torch
import numpy as np
import json
from tqdm import tqdm
from torch import nn
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer
import wandb
from src.models.VanillaTransformer import make_model_VT
from src.models.TagInsert import make_model_TI
from src.models.TagInsertL2R import make_model_TIL2R
from datasets import DatasetDict
from datasets import Dataset as HuggingFaceDataset
from transformers import DataCollatorForTokenClassification, AutoModelForTokenClassification, TrainingArguments, Trainer
import evaluate as ev
import pandas as pd

############# Logic for training of all models, including processing of data before forwarding, datasets classes and batches #############

# Converter for the path of the processed data, proportion to percentage
PROP_CONVERTER = {1: "100%", 0.75: "75%", 0.5: "50%", 0.25: "25%", 0.1: "10%"}


#### General Training utils #####

class TaggingDataset(Dataset):
    """ 
    Iterable Dataset class for the tagging task.
    Tracks the indexed words, tags and original sentences.
    """
    def __init__(self, data):
        data = torch.load(data, weights_only=True)
        self.word_sequences = data['words']
        self.tag_sequences = data['tags']
        self.original_sentences = data['original_sentences']

    def shuffle(self):
        """ Shuffle util function if needed. """
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
    """ Returns the BERT model and tokenizer specified in the config file. """
    model = AutoModel.from_pretrained(model_name, output_hidden_states = True)
    model = model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer

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

def rate(step, model_size, factor, warmup):
    """ Function to calculate the learning rate for the optimizer """
    
    if step == 0:
        step = 1
    return factor * (
        model_size ** (-0.5) * min(step ** (-0.5), step * warmup ** (-1.5))
    )

def save_model(model, optimizer, lr_scheduler, train_losses, val_losses, epochs, path):
    """ Function to save the model, the optimizer, the learning rate scheduler, the training and validation losses and the number of epochs """
    
    torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'lr_scheduler' : lr_scheduler.state_dict(),
                'train_losses': train_losses,
                'val_losses': val_losses,
                'epochs': epochs
                }, path)

def load_model(model_name, path, config, tagging):
    """ Function to load the model, the optimizer, the learning rate scheduler, the training and validation losses and the number of epochs """
    
    if model_name == "VT":
        model = make_model_VT(tagging, d_model = config['hidden_size'], N=config['n_stacks'], h = config['n_heads'])
    elif model_name == "TI":
        model = make_model_TI(tagging, d_model = config['hidden_size'], N=config['n_stacks'], h = config['n_heads'])
    elif model_name == "TIL2R":
        model = make_model_TIL2R(tagging, d_model = config['hidden_size'], N=config['n_stacks'], h = config['n_heads'])
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
    """ Function that creates a Left-to-Right square mask mask of a given size """
    attn_shape = (1, size, size)
    subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1).type(
        torch.uint8
    )
    return subsequent_mask == 0

def pad_mask(pad, tgt):
    """ Function that creates a mask to hide padding tokens """
    tgt_mask = (tgt != pad).unsqueeze(-2)
    return tgt_mask

##### Training Utils for Vanilla Transformer and TagInsert L2R #####

class Batch_VT:
    """ Object for holding a batch of data with Left-to-Right mask during training of the Vanilla Transformer and TagInsert L2R.  """

    def __init__(self, src, tgt=None, embs = None, pad=0):
        # store indexed words and its pad mask
        self.src = src
        self.src_mask = (src != pad).unsqueeze(-2)
        # store pre-calculated BERT embeddings
        self.embs = embs
        if tgt is not None:
            # calculate the original sentence lengths
            self.seq_lengths = [torch.nonzero(tgt[i] == 0, as_tuple = False)[0,0].item()-1 if torch.nonzero(tgt[i] == 0, as_tuple = False).numel() > 0 else src.size(1)-1 for i in range(tgt.size(0))]
            # store targets to forward to the model, we remove the last token from the target because the model learns to predict the next token
            self.tgt = tgt[:, :-1]
            # store the targets to calculate the loss, we remove the first token (start) from the target because the model learns to predict the next token
            self.tgt_y = tgt[:, 1:]
            # store the mask to hide padding and future words (L2R)
            self.tgt_mask = self.make_std_mask(self.tgt, pad)
            # store the number of tokens in the batch for normalizing the loss
            self.ntokens = (self.tgt_y != pad).data.sum()

    @staticmethod
    def make_std_mask(tgt, pad):
        """ Create a mask to hide padding and future words (L2R). """
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & subsequent_mask(tgt.size(-1)).type_as(
            tgt_mask.data
        )
        return tgt_mask

class Batch_TI:
    """ 
    Object for holding a batch of data with only pad mask during training of TagInsert.
    Data is also prepared for training the model, which includes preparng the targets and the trajectories.
    """
    
    def __init__(self, config, tgt_map, src, trg=None, embs = None, pad=0):
        # store the words and its pad mask
        self.src = src
        self.src_mask = (src != pad).unsqueeze(-2)
        if trg is not None:
            # extracting length of each sentence
            self.seq_lengths = [torch.nonzero(trg[i] == 0, as_tuple = False)[0,0].item()-1 if torch.nonzero(trg[i] == 0, as_tuple = False).numel() > 0 else config['block_size']-2 for i in range(trg.size(0))]
            # initialize target vector to forward to the model, (BATCH_SIZE, BLOCK_SIZE)
            new_trg = torch.zeros((src.size(0), config['block_size']), dtype = torch.int64).to(config['device'])
            # initialize target vector to calculate the loss, (BATCH_SIZE, BLOCK_SIZE)
            new_trg_y = torch.zeros((src.size(0), config['block_size']), dtype = torch.int64).to(config['device'])
            # initialize the trajectory for each sentence
            self.trajectory = []
            # for each sentence, we keep track of how many tags have been inserted thus far
            self.inserted = torch.zeros(src.size(0), dtype = torch.int64)
            # populating the target vectors
            for i, train in enumerate(trg):
                # randomly sampling the trajectory for the sentence
                all_ixs = torch.arange(start = 1, end = self.seq_lengths[i]+1)
                permuted_ixs = torch.randperm(all_ixs.size(0))
                permuted_all_ixs = all_ixs[permuted_ixs]
                self.trajectory.append(permuted_all_ixs)

                # populating targets to be forwarded to the model, at first all tokens are <UNK>, after each trajectory step, one tag will be added in place of one <UNK>
                # first token is always <START>, last token is always <END>, everything in between is <UNK>, the rest are <PAD>
                vec = torch.full((config['block_size'],), tgt_map['<UNK>'])
                vec[0] = tgt_map['<START>']
                vec[self.seq_lengths[i]+1] = tgt_map['<END>']
                vec[self.seq_lengths[i]+2:] = tgt_map['<PAD>']
                # store the targets to be forwarded to the model
                new_trg[i] = vec

                # populating targets to calculate the loss, this is a static vector that always have the gold tags
                ins = 0
                targets = torch.zeros(config['block_size']).to(config['device'])
                for j, ix in enumerate(vec):
                        # if its between <START> and <END>, we insert the actual tag from the training data
                        if ix == tgt_map['<UNK>']:
                            targets[ins] = train[j]
                            ins+=1
                # store the targets to calculate the loss
                new_trg_y[i] = targets
            self.tgt = new_trg
            self.tgt_y = new_trg_y
            # the mask for TagInsert is only for pad tokens
            self.tgt_mask = self.make_pad_mask(self.tgt, pad)
            # the number of tokens in the batch for normalizing the loss
            nonpads = (self.tgt_y != pad).data.sum()
            self.ntokens = nonpads
            # store the pre-calculated BERT embeddings
            self.embs = embs

    def next_trajectory(self):
        """ Function to move to the next trajectory step. Effectively inserts a tag in place of a <UNK> token following the pre-sampled trajectory. """
        
        for i, ins in enumerate(self.inserted):
            # if all tags have been inserted, we skip the sentence
            if ins >= self.seq_lengths[i]:
              continue
            # extract tag to be inserted and insert it in the target vector
            next_tag_pos = self.trajectory[i][ins].item()
            self.tgt[i][next_tag_pos] = self.tgt_y[i][next_tag_pos-1]
            self.inserted[i] += 1

    @staticmethod
    def make_pad_mask(tgt, pad):
        """ Create a mask to hide padding tokens. """

        tgt_mask = (tgt != pad).unsqueeze(-2)
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
        model, model_opt, lr_scheduler, logs = load_model(model_name, model_path, config, tagging)
        trained_epochs = logs['epochs']
        train_losses = logs['train_losses']
        val_losses = logs['val_losses']
        print('Loading a pre-existent model.')
    except:
        if model_name == "VT":
            model = make_model_VT(tagging, d_model = config['hidden_size'], N=config['n_stacks'], h = config['n_heads'])
        elif model_name == "TI":
            model = make_model_TI(tagging, d_model = config['hidden_size'], N=config['n_stacks'], h = config['n_heads'])
        elif model_name == "TIL2R":
            model = make_model_TIL2R(tagging, d_model = config['hidden_size'], N=config['n_stacks'], h = config['n_heads'])
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
        if config['model_name'] == "VanillaTransformer" or config['model_name'] == "TagInsertL2R":
            yield Batch_VT(src, tgt, embs, pad=pad)
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
            max_len = np.max(batch.seq_lengths) - 1
            losses = torch.zeros(max_len).to(config['device'])
            for traj_step in range(max_len):
                # print(f'Calculating loss for trajectory {traj_step}.')
                out = model.forward(batch.src, batch.tgt,
                                    batch.src_mask, batch.tgt_mask, batch.embs)
                _, loss_node = loss_compute(out, batch.tgt, batch.tgt_y, batch.seq_lengths, batch.ntokens, batch.inserted)
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
    train_dataset = TaggingDataset(data_path + f"train_data.pth")
    val_dataset = TaggingDataset(data_path + f"val_data.pth")
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
    if config['model_name'] == "VanillaTransformer":
        run_model_name = "VT"
    elif config['model_name'] == "TagInsert":
        run_model_name = "TI"
    elif config['model_name'] == "BERT":
        run_model_name = "BE"
    elif config['model_name'] == "TagInsertL2R":
        run_model_name = "TIL2R"
    run_name = run_model_name + "_" + tagging + "_" + str(config['data_proportion'])
    with wandb.init(project="TagInsert", config=config, name = run_name):
        bert_model, tokenizer = load_BERT_encoder(config['bert_model'], config['device'])
        prop_path = PROP_CONVERTER[config['data_proportion']]
        tgt_map = json.load(open(f'data/{tagging}/processed/{prop_path}/{tagging}_to_idx.json'))
        train_dataloader, val_dataloader, len_train, len_val = get_dataloaders(f"data/{tagging}/processed/{prop_path}/", config)
        if config['model_name'] == "VanillaTransformer" or config['model_name'] == "TagInsertL2R":
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
            if config['model_name'] == "VanillaTransformer" or config['model_name'] == "TagInsertL2R":
                train_losses = run_epoch_VT(epoch_data_iter, model, LossCompute_VT(model.generator, criterion), model_opt, lr_scheduler, config, len_train, mode = 'train', train_state=train_state)[2]
            elif config['model_name'] == "TagInsert":
                train_losses = run_epoch_TI(epoch_data_iter, model, LossCompute_TI(model.generator, criterion), model_opt, lr_scheduler, config, len_train, mode = 'train', train_state=train_state)[2]
            all_train_losses += train_losses
            with torch.no_grad():
                model.eval()
                if config['model_name'] == "VanillaTransformer" or config['model_name'] == "TagInsertL2R":
                    val_losses = run_epoch_VT(val_epoch_data_iter, model, LossCompute_VT(model.generator, criterion), DummyOptimizer(),DummyScheduler(), config, len_val, mode = 'eval', train_state=train_state)[2]
                elif config['model_name'] == "TagInsert":
                    val_losses = run_epoch_TI(val_epoch_data_iter, model, LossCompute_TI(model.generator, criterion), DummyOptimizer(),DummyScheduler(), config, len_val, mode = 'eval', train_state=train_state)[2]
                all_val_losses += val_losses
            wandb.log({"train_loss": np.mean(train_losses) , "val_loss": np.mean(val_losses), "epoch": epoch})
    wandb.finish()
    if save:
        model_name = config['model_name'] + "_" + tagging + "_" + str(config['data_proportion'])
        save_model(model, model_opt, lr_scheduler, all_train_losses, all_val_losses, epoch, f"models/{model_name}")
    return model

########## BERT Encoder Train+Eval logic ##########

def preprocess_and_train_BERT_Encoder(config, tagging):
    prop_path = PROP_CONVERTER[config['data_proportion']]
    _, tokenizer = load_BERT_encoder(config['bert_model'], config['device'])
    prop_path = PROP_CONVERTER[config['data_proportion']]
    idx_to_tgt = json.load(open(f'data/{tagging}/processed/{prop_path}/idx_to_{tagging}.json'))
    tgt_to_idx = json.load(open(f'data/{tagging}/processed/{prop_path}/{tagging}_to_idx.json'))
    model = AutoModelForTokenClassification.from_pretrained(config['bert_model'], num_labels=len(tgt_to_idx), id2label=idx_to_tgt, label2id=tgt_to_idx)
    train_data = torch.load(f"data/{tagging}/processed/{prop_path}/train_data.pth", weights_only=True)
    val_data = torch.load(f"data/{tagging}/processed/{prop_path}/val_data.pth", weights_only=True)
    words, tags, original_sentences = train_data['words'], train_data['tags'], train_data['original_sentences']
    val_words, val_tags, val_original_sentences = val_data['words'], val_data['tags'], val_data['original_sentences']
    # print(len(tags), len(words), tags[0], words[0])
    sentence_POS_idx = [sent_tags[1:len(original_sentences[i])+1] for i, sent_tags in enumerate(tags)]
    val_sentence_POS_idx = [sent_tags[1:len(val_original_sentences[i])+1] for i, sent_tags in enumerate(val_tags)]
    text_sents = [' '.join(sent) for sent in original_sentences]
    val_text_sents = [' '.join(sent) for sent in val_original_sentences]
    tokenized_data = tokenizer(text_sents, padding="max_length", truncation=True, max_length=config['bert_block_size'], return_tensors="pt")
    val_tokenized_data = tokenizer(val_text_sents, padding="max_length", truncation=True, max_length=config['bert_block_size'], return_tensors="pt")
    mappings = get_BERT_mappings(tokenized_data, original_sentences, tokenizer)
    val_mappings = get_BERT_mappings(val_tokenized_data, val_original_sentences, tokenizer)

    train_labels = map_data(mappings, sentence_POS_idx)
    val_labels = map_data(val_mappings, val_sentence_POS_idx)
    train_tokens = original_sentences
    val_tokens = val_original_sentences
    train_POS = sentence_POS_idx
    val_POS = val_sentence_POS_idx
    # create a dictionart with the training data
    data = {'id': list(range(len(train_tokens))), 'tokens': train_tokens, 'POS': train_POS, 'attention_mask': tokenized_data['attention_mask'].tolist(), 'input_ids': tokenized_data['input_ids'].tolist(), 'labels': train_labels}
    df = pd.DataFrame(data)
    train_dataset = HuggingFaceDataset.from_pandas(df)

    # create a dictionary with the validation data
    val_data = {'id': list(range(len(val_tokens))), 'tokens': val_tokens, 'POS': val_POS, 'attention_mask': val_tokenized_data['attention_mask'].tolist(), 'input_ids': val_tokenized_data['input_ids'].tolist(), 'labels': val_labels}
    val_df = pd.DataFrame(val_data)
    val_dataset = HuggingFaceDataset.from_pandas(val_df)
    datasets = DatasetDict({'train': train_dataset, 'validation': val_dataset})
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
    training_args = TrainingArguments(disable_tqdm=False, output_dir="models/BERT_Encoder", learning_rate=config['lr'], per_device_train_batch_size=config['batch_size'], per_device_eval_batch_size=config['batch_size'], num_train_epochs=config['epochs'], weight_decay=config['weight_decay'], evaluation_strategy="epoch", report_to="wandb")
    trainer = Trainer(model=model, args=training_args, train_dataset=datasets["train"], eval_dataset=datasets["validation"], tokenizer=tokenizer, data_collator=data_collator, compute_metrics=compute_metrics)
    result = trainer.train()
    trainer.save_model("models/BERT_Encoder")
    return model

def bad_tokens_and_map(tokenized_sentence, original_sentence, tokenizer, cased=True):
    bad_tokens = []
    detokenized_sentence = tokenizer.convert_ids_to_tokens(tokenized_sentence)
    mapping_to_original = [None] # For the [CLS] token
    detokenized_index = 1 # Skip the [CLS] token
    for i, word in enumerate(original_sentence):
        word = word if cased else word.lower()
        detokenized_word = detokenized_sentence[detokenized_index]
        next_detokenized_word = detokenized_sentence[detokenized_index+1]
        if word != detokenized_word:
            if not next_detokenized_word.startswith('##') and not detokenized_word.startswith('##'):
                bad_tokens.append(word)
            reconstructed_word = detokenized_word
            while word != reconstructed_word:
                detokenized_index += 1
                reconstructed_word += detokenized_sentence[detokenized_index].strip('##')
                mapping_to_original.append(i)
            mapping_to_original.append(i)
        else:
            mapping_to_original.append(i)
        detokenized_index += 1
    while len(mapping_to_original) < len(detokenized_sentence):
        mapping_to_original.append(None)
    return bad_tokens, mapping_to_original

def get_BERT_mappings(tokenized_data, original_sentences, tokenizer):
    mappings = []
    for i in range(len(tokenized_data['input_ids'])):
        bad_tokens, mapping = bad_tokens_and_map(tokenized_data['input_ids'][i], original_sentences[i], tokenizer)
        mappings.append(mapping)
    return mappings

def map_data(mappings, data):
    all_labels = []
    for i, mapping in enumerate(mappings):
        labels = []
        previous_word_idx = None
        orig_j = 0
        for j, idx in enumerate(mapping):
            if idx is None:
                labels.append(-100)
            elif idx == previous_word_idx:
                labels.append(-100)
            else:
                labels.append(data[i][orig_j])
                orig_j += 1
                previous_word_idx = idx
        all_labels.append(labels)
    return all_labels

def compute_metrics(p):
    metric = ev.load('accuracy')
    pred, labels = p
    pred = np.argmax(pred, axis=2)
    # ignore predictions where we don't have a valid label
    mask = labels != -100
    pred = pred[mask]
    labels = labels[mask]
    return metric.compute(predictions=pred, references=labels)
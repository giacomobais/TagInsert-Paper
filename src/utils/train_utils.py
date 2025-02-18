
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
        self.embs = data['embs']

    def shuffle(self):
        """ Shuffle util function if needed. """
        shuffled_indices = torch.randperm(len(self.word_sequences))
        self.word_sequences = [self.word_sequences[i] for i in shuffled_indices]
        self.tag_sequences = [self.tag_sequences[i] for i in shuffled_indices]
        self.original_sentences = [self.original_sentences[i] for i in shuffled_indices]
        self.embs = [self.embs[i] for i in shuffled_indices]

    def __len__(self):
        return len(self.word_sequences)

    def __getitem__(self, idx):
        words = torch.tensor(self.word_sequences[idx], dtype=torch.long)
        tags = torch.tensor(self.tag_sequences[idx], dtype=torch.long)
        original_sentence = self.original_sentences[idx]
        embs = self.embs[idx]
        return words, tags, original_sentence, embs

def load_BERT_encoder(model_name, device = 'cuda'):
    """ Returns the BERT model and tokenizer specified in the config file. """
    model = AutoModel.from_pretrained(model_name, output_hidden_states = True)
    model = model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer



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
    """ 
    Dummy optimizer class to be used when no optimizer is needed.    
    """
    def __init__(self):
        self.param_groups = [{"lr": 0}]
        None

    def step(self):
        None

    def zero_grad(self, set_to_none=False):
        None


class DummyScheduler:
    """
    Dummy scheduler class to be used when no scheduler is needed.
    """
    def step(self):
        None

class LabelSmoothing(nn.Module):
    """
    Calculate the Kullback-Leibler divergence loss between the predicted and the actual distribution.
    Label smoothing is also included.
    """

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
    """
    A simple class that calculates the loss for the Vanilla Transformer and TagInsert L2R.
    The loss is normalized by the number of tokens in the batch.
    """

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
    """
    Class to calculate the loss for TagInsert. Each sentence loss is individually calculated. The loss is defined as the negative log likelihood
    of inserting the correct tags at all the positions corresponding to an <UNK> token. The sentence loss is the average of the losses for each position.
    The full batch loss is the average of all sentence losses for the batch.
    """
    def __init__(self, config, tgt_map):
            super(TI_Loss, self).__init__()
            self.tgt_map = tgt_map
            self.config = config

    def sentence_loss(self, logits, forwarded_trgs, targets, sequence_length):
            # function to calculate the loss for a single sentence
            losses = []
            # iterate over the tags present in the current trajectory step
            for i, ix in enumerate(forwarded_trgs):
                # if the tag is <PAD>, the sentence is over and we interrupt the loop
                if ix == self.tgt_map['<PAD>']:
                    break
                # loss is calculated in positions corresponding to <UNK> tokens
                if ix == self.tgt_map['<UNK>']:
                    # extract gold tag for the position
                    tag_to_insert = targets[i-1]
                    # extract logit of inserting the gold tag in the position
                    p = logits[i][tag_to_insert]
                    # calculate and log the loss
                    losses.append(-torch.log(p))
            # This should never happen, but just in case, if the full sentence is part of the trajectory steps, we calculate the loss of inserting an <END> tag
            if len(losses) == 0:
                tag_to_insert = self.tgt_map['<END>']
                p = logits[sequence_length+2, tag_to_insert]
                losses.append(-torch.log(p))
            # average the losses for the sentence
            out = torch.mean(torch.stack(losses))
            return out

    def forward(self, logits, forwarded_trgs, targets, sequence_lengths, inserted):
            # function to calculate the loss for the full batch
            batch_losses = []
            # iterate over the sentences in the batch
            for i, _ in enumerate(targets):
                # if every tag has been inserted, we skip the sentence
                if inserted[i] >= sequence_lengths[i]:
                    continue
                # calculate the loss of the sentence, passing the according logits, gold targets and forwarded targets
                batch_loss = self.sentence_loss(logits[i, :, :], forwarded_trgs[i, :], targets[i, :], sequence_lengths[i])
                # log the loss
                batch_losses.append(batch_loss)
            # average the losses for the batch
            loss = torch.mean(torch.stack(batch_losses))
            return loss

class LossCompute_TI:
    """
    A simple class that calculates the loss for TagInsert.
    """
    def __init__(self, generator, criterion, opt=None):
        self.generator = generator
        self.criterion = criterion
        self.opt = opt

    def __call__(self, x, forwarded_y, y, sequence_lengths, norm, inserted):
        # generate logits throught the output layer
        x = self.generator(x)
        # this should not happen, but just in case
        if norm == 0:
            norm = 1
        # calculate the loss
        loss = self.criterion(x, forwarded_y, y, sequence_lengths, inserted)
        return loss.data * norm, loss

def resume_training(model_path, config, model_name, tagging):
    """
    Function that loads a pre-saved model from a checkpoint. If the model does not exist, it initiliazes a new untrained one.
    """
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
    """
    Function that iterates through the data loader and prepares the batches for the model.
    """
    for batch_data in data_loader:
        xb, yb, _, embs = batch_data
        #embs = extract_BERT_embs(original_sentences, bert_model, tokenizer, config)
        src = xb.to(config['device'])
        tgt = yb.to(config['device'])
        embs = torch.stack(embs, dim=0)
        embs = embs.to(config['device'])
        if config['model_name'] == "VanillaTransformer" or config['model_name'] == "TagInsertL2R":
            yield Batch_VT(src, tgt, embs, pad=pad)
        elif config['model_name'] == "TagInsert":
            yield Batch_TI(config, tgt_map, src, tgt, embs, pad=pad)


def run_epoch_VT(data_iter, model, loss_compute, optimizer, lr_scheduler, config, data_len, mode = 'train', accum_iter = 1, train_state = TrainState()):
    """
    Function that runs a single epoch for the Vanilla Transformer and TagInsert L2R.
    It includes backward prop and wandb tracking for the training mode. The function is also used for evaluation.
    """
    total_tokens = 0
    total_loss = 0
    losses = []
    log_id = "Batch train loss" if mode == "train" else "Batch val loss"
    with tqdm(data_iter, total = data_len // config['batch_size'], desc=mode) as pbar:
        # for each batch in the epoch
        for i, batch in enumerate(pbar):
            # forward to the model and get the logits
            out = model.forward(batch.src, batch.tgt, batch.src_mask, batch.tgt_mask, batch.embs)
            # calculate the loss for the batch
            loss, loss_node = loss_compute(out, batch.tgt_y, batch.ntokens)
            losses.append(loss_node.item())
            if mode == "train":
                # if in training mode, we calculate the gradients and update the model
                loss_node.backward()
                train_state.step += 1
                train_state.samples += batch.src.size(0)
                train_state.tokens += batch.ntokens
                if (i + 1) % accum_iter == 0:
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    train_state.accum_step += 1
                # log the train loss for the batch
                wandb.log({log_id: loss_node.item(), "train_step": train_state.step})
                # step the learning rate scheduler
                lr_scheduler.step()
            else:
                train_state.eval_step += 1
                # log the validation loss for the batch
                wandb.log({log_id: loss_node.item(), "val_step": train_state.eval_step})
            # update the total loss
            total_loss += loss
            total_tokens += batch.ntokens
            pbar.set_postfix(loss=loss_node.item())
    # return the loss normalized by the number of tokens
    return total_loss / total_tokens, train_state, losses

def run_epoch_TI(data_iter, model, loss_compute, optimizer, lr_scheduler, config, data_len, mode = 'train', accum_iter = 1, train_state = TrainState()):
    """
    Function that runs a single epoch for TagInsert.
    It includes backward prop and wandb tracking for the training mode. The function is also used for evaluation.
    """
    total_tokens = 0
    total_loss = 0
    current_losses = []
    log_id = "Batch train loss" if mode == "train" else "Batch val loss"
    with tqdm(data_iter, total = data_len // config['batch_size'], desc=mode) as pbar:
        # for each batch in the epoch
        for i, batch in enumerate(pbar):
            # get the length of the longest sentence in the batch minus one, so that we don't calculate more losses than needed
            max_len = np.max(batch.seq_lengths) - 1
            losses = torch.zeros(max_len).to(config['device'])
            # for each trajectory step
            for traj_step in range(max_len):
                # forward to the model and get the logits
                out = model.forward(batch.src, batch.tgt,
                                    batch.src_mask, batch.tgt_mask, batch.embs)
                # calculate the loss for the batch
                _, loss_node = loss_compute(out, batch.tgt, batch.tgt_y, batch.seq_lengths, batch.ntokens, batch.inserted)
                if mode == "train":
                    # if in training mode, we calculate the gradients and update the model
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
                # log the loss for the trajectory step
                losses[traj_step] = loss_node.item()
                # move to the next trajectory step, a new pre-sampled tag is inserted in place of a <UNK> token
                batch.next_trajectory()
            # free up memory
            del batch.embs
            # average the losses for the batch
            loss_node_backward = torch.mean(losses)
            if mode == "train":
                train_state.step += 1
                # log the train loss for the batch
                wandb.log({log_id: loss_node_backward.item(), "train_step": train_state.step})
            else:
                train_state.eval_step += 1
                # log the validation loss for the batch
                wandb.log({log_id: loss_node_backward.item(), "val_step": train_state.eval_step})
            # log the loss for the batch
            current_losses.append(loss_node_backward.item())
            total_loss += loss_node_backward.data
            total_tokens += batch.ntokens
            pbar.set_postfix(loss=loss_node_backward.item())
    # return the loss normalized by the number of tokens
    return total_loss / total_tokens, train_state, current_losses

def collate_fn(batch):
    """
    Collate function for the dataloader. It stacks the words and tags and returns the original sentences and word embeddings.
    """
    words, tags, original_sentences, embs = zip(*batch)
    words = torch.stack(words)
    tags = torch.stack(tags)
    return words, tags, original_sentences, embs

def get_dataloaders(data_path, config, shuffle = True):
    """
    Function that creates Datasets and Dataloaders for the training and validation data.
    """
    train_dataset = TaggingDataset(data_path + f"train_data.pth")
    val_dataset = TaggingDataset(data_path + f"val_data.pth")
    train_dataloader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=shuffle, collate_fn=collate_fn)
    val_dataloader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=shuffle, collate_fn=collate_fn)
    len_train = len(train_dataset)
    len_val = len(val_dataset)
    return train_dataloader, val_dataloader, len_train, len_val

def define_wandb_metrics():
    """
    Simple function that returns metric to be logged in wandb.
    """
    wandb.define_metric("train_step")
    wandb.define_metric("Batch train loss", step_metric="train_step")
    wandb.define_metric("val_step")
    wandb.define_metric("Batch val loss", step_metric="val_step")
    wandb.define_metric("epoch")
    wandb.define_metric("train_loss", step_metric="epoch")
    wandb.define_metric("val_loss", step_metric="epoch")

def train(model_package, config, tagging, save = True):
    """
    Function that trains any of the models included in this package.
    """
    model, model_opt, lr_scheduler, trained_epochs, train_losses, val_losses = model_package
    # naming conversions for the wandb run
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
        # load the BERT encoder, to remove if using pre calculated embeddings
        bert_model, tokenizer = load_BERT_encoder(config['bert_model'], config['device'])
        # naming for paths logic
        prop_path = PROP_CONVERTER[config['data_proportion']]
        # load the mapping for the tags
        tgt_map = json.load(open(f'data/{tagging}/processed/{prop_path}/{tagging}_to_idx.json'))
        # get the dataloaders for the training and validation data
        train_dataloader, val_dataloader, len_train, len_val = get_dataloaders(f"data/{tagging}/processed/{prop_path}/", config)
        # initialize losses depending on the model
        if config['model_name'] == "VanillaTransformer" or config['model_name'] == "TagInsertL2R":
            criterion = LabelSmoothing(size=len(tgt_map), padding_idx=0, smoothing=0.0)
        elif config['model_name'] == "TagInsert":
            criterion = TI_Loss(config, tgt_map)
        
        # training setup
        train_state = TrainState()
        define_wandb_metrics()
        wandb.watch(model, log="all")
        all_train_losses = []
        all_val_losses = []
        # for each epoch, starts by the last trained epoch in case of checkpoint
        for epoch in range(trained_epochs, config['epochs']):
            print(f"Epoch {epoch}")
            # get the iterators for the training and validation data
            epoch_data_iter = dataload(train_dataloader, config, bert_model, tokenizer, tgt_map, pad=0)
            val_epoch_data_iter = dataload(val_dataloader, config, bert_model, tokenizer, tgt_map, pad=0)
            model.train()
            # run the epoch for the model
            if config['model_name'] == "VanillaTransformer" or config['model_name'] == "TagInsertL2R":
                train_losses = run_epoch_VT(epoch_data_iter, model, LossCompute_VT(model.generator, criterion), model_opt, lr_scheduler, config, len_train, mode = 'train', train_state=train_state)[2]
            elif config['model_name'] == "TagInsert":
                train_losses = run_epoch_TI(epoch_data_iter, model, LossCompute_TI(model.generator, criterion), model_opt, lr_scheduler, config, len_train, mode = 'train', train_state=train_state)[2]
            # log losses
            all_train_losses += train_losses
            # validation epoch
            with torch.no_grad():
                model.eval()
                if config['model_name'] == "VanillaTransformer" or config['model_name'] == "TagInsertL2R":
                    val_losses = run_epoch_VT(val_epoch_data_iter, model, LossCompute_VT(model.generator, criterion), DummyOptimizer(),DummyScheduler(), config, len_val, mode = 'eval', train_state=train_state)[2]
                elif config['model_name'] == "TagInsert":
                    val_losses = run_epoch_TI(val_epoch_data_iter, model, LossCompute_TI(model.generator, criterion), DummyOptimizer(),DummyScheduler(), config, len_val, mode = 'eval', train_state=train_state)[2]
                # log losses
                all_val_losses += val_losses
            # log the losses for the epoch in wandb
            wandb.log({"train_loss": np.mean(train_losses) , "val_loss": np.mean(val_losses), "epoch": epoch})
    wandb.finish()

    # save model if specified
    if save:
        model_name = config['model_name'] + "_" + tagging + "_" + str(config['data_proportion'])
        save_model(model, model_opt, lr_scheduler, all_train_losses, all_val_losses, epoch, f"models/{model_name}")
    return model

########## BERT Encoder Train+Eval logic ##########

def preprocess_and_train_BERT_Encoder(config, tagging):
    """
    Function that slightly modifies the data to be used with the BERT encoder and trains the model.
    Training is done through the HuggingFace Trainer API, which also includes validation and tracking through wandb.
    """
    prop_path = PROP_CONVERTER[config['data_proportion']]
    # load the tokenizer and mapping for the tags
    _, tokenizer = load_BERT_encoder(config['bert_model'], config['device'])
    idx_to_tgt = json.load(open(f'data/{tagging}/processed/{prop_path}/idx_to_{tagging}.json'))
    tgt_to_idx = json.load(open(f'data/{tagging}/processed/{prop_path}/{tagging}_to_idx.json'))
    # load the BERT model for token classification
    model = AutoModelForTokenClassification.from_pretrained(config['bert_model'], num_labels=len(tgt_to_idx), id2label=idx_to_tgt, label2id=tgt_to_idx)
    # load the training and validation data
    train_data = torch.load(f"data/{tagging}/processed/{prop_path}/train_data.pth", weights_only=True)
    val_data = torch.load(f"data/{tagging}/processed/{prop_path}/val_data.pth", weights_only=True)
    _, tags, original_sentences = train_data['words'], train_data['tags'], train_data['original_sentences']
    _, val_tags, val_original_sentences = val_data['words'], val_data['tags'], val_data['original_sentences']
    # remove start token and paddings from tags
    sentence_POS_idx = [sent_tags[1:len(original_sentences[i])+1] for i, sent_tags in enumerate(tags)]
    val_sentence_POS_idx = [sent_tags[1:len(val_original_sentences[i])+1] for i, sent_tags in enumerate(val_tags)]
    # convert list of words into normal strings for the BERT forward pass
    text_sents = [' '.join(sent) for sent in original_sentences]
    val_text_sents = [' '.join(sent) for sent in val_original_sentences]
    # tokenize words for the BERT model
    tokenized_data = tokenizer(text_sents, padding="max_length", truncation=True, max_length=config['bert_block_size'], return_tensors="pt")
    val_tokenized_data = tokenizer(val_text_sents, padding="max_length", truncation=True, max_length=config['bert_block_size'], return_tensors="pt")
    # map each subword token to the original word index
    mappings = get_BERT_mappings(tokenized_data, original_sentences, tokenizer)
    val_mappings = get_BERT_mappings(val_tokenized_data, val_original_sentences, tokenizer)
    #  mapping the tags to the words using the prefix method
    train_labels = map_data(mappings, sentence_POS_idx)
    val_labels = map_data(val_mappings, val_sentence_POS_idx)
    # renaming for clarity
    train_tokens = original_sentences
    val_tokens = val_original_sentences
    train_POS = sentence_POS_idx
    val_POS = val_sentence_POS_idx
    # create a huggingface dataset with the training data and all the necessary information for training
    data = {'id': list(range(len(train_tokens))), 'tokens': train_tokens, 'POS': train_POS, 'attention_mask': tokenized_data['attention_mask'].tolist(), 'input_ids': tokenized_data['input_ids'].tolist(), 'labels': train_labels}
    df = pd.DataFrame(data)
    train_dataset = HuggingFaceDataset.from_pandas(df)
    # create a huggingface dataset with the validation data and all the necessary information for validation
    val_data = {'id': list(range(len(val_tokens))), 'tokens': val_tokens, 'POS': val_POS, 'attention_mask': val_tokenized_data['attention_mask'].tolist(), 'input_ids': val_tokenized_data['input_ids'].tolist(), 'labels': val_labels}
    val_df = pd.DataFrame(val_data)
    val_dataset = HuggingFaceDataset.from_pandas(val_df)
    # merge the datasets and define collator
    datasets = DatasetDict({'train': train_dataset, 'validation': val_dataset})
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
    # setup training
    training_args = TrainingArguments(disable_tqdm=False, output_dir="models/BERT_Encoder", learning_rate=config['lr'], per_device_train_batch_size=config['batch_size'], per_device_eval_batch_size=config['batch_size'], num_train_epochs=config['epochs'], weight_decay=config['weight_decay'], evaluation_strategy="epoch", report_to="wandb")
    trainer = Trainer(model=model, args=training_args, train_dataset=datasets["train"], eval_dataset=datasets["validation"], tokenizer=tokenizer, data_collator=data_collator, compute_metrics=compute_metrics)
    # train, result are tracked through wandb
    _ = trainer.train()
    # save model
    trainer.save_model("models/BERT_Encoder")
    return model

def bad_tokens_and_map(tokenized_sentence, original_sentence, tokenizer, cased=True):
    """
    Function that tracks the subwords from the BERT tokenizer to their original words indeces.
    The function also tracks weirdly tokenized words.
    """
    bad_tokens = []
    # get the original subwords from the tokenized sentence
    detokenized_sentence = tokenizer.convert_ids_to_tokens(tokenized_sentence)
    mapping_to_original = [None] # For the [CLS] token
    detokenized_index = 1 # Skip the [CLS] token
    # for each word in the original sentence
    for i, word in enumerate(original_sentence):
        # convert to lower case if needed
        word = word if cased else word.lower()
        # get the two next subwords for the current word
        detokenized_word = detokenized_sentence[detokenized_index]
        next_detokenized_word = detokenized_sentence[detokenized_index+1]
        # reconstructing the word from the subwords in order to track how many subwords are part of the word
        if word != detokenized_word:
            # if neither the current subword nor the next one start with '##', it means that this is a weirdly tokenized word, we save it in case it is needed for inspection
            if not next_detokenized_word.startswith('##') and not detokenized_word.startswith('##'):
                bad_tokens.append(word)
            # progressively reconstruct the word from the subwords
            reconstructed_word = detokenized_word
            while word != reconstructed_word:
                # go to the next subword
                detokenized_index += 1
                reconstructed_word += detokenized_sentence[detokenized_index].strip('##')
                # track the index for the original word
                mapping_to_original.append(i)
            # last subword to track for this word
            mapping_to_original.append(i)
        # if there was no subword, just track the original word index
        else:
            mapping_to_original.append(i)
        # go next subword
        detokenized_index += 1
    # make sure the mapping is the same length as the original sentence
    while len(mapping_to_original) < len(detokenized_sentence):
        mapping_to_original.append(None)
    return bad_tokens, mapping_to_original

def get_BERT_mappings(tokenized_data, original_sentences, tokenizer):
    """
    Function that gets the mapping from the subwords to the original words for each sentence.
    """
    mappings = []
    for i in range(len(tokenized_data['input_ids'])):
        _, mapping = bad_tokens_and_map(tokenized_data['input_ids'][i], original_sentences[i], tokenizer)
        mappings.append(mapping)
    return mappings

def map_data(mappings, data):
    """
    Returns a mapping that links the first subword of each word to the tag of the word (prefix method).
    """
    all_labels = []
    # for each sentence
    for i, mapping in enumerate(mappings):
        labels = []
        previous_word_idx = None
        orig_j = 0
        # for each subword
        for _, idx in enumerate(mapping):
            # if we logged a none, it is either the [CLS] or paddings, which we ignore
            if idx is None:
                labels.append(-100)
            # if the mapping show the same index as the previous one, it means that the subword is part of the same word and we only map the first subword
            elif idx == previous_word_idx:
                labels.append(-100)
            # first subward, this one will be mapped to the tag of the word
            else:
                # get the tag of the word
                labels.append(data[i][orig_j])
                orig_j += 1
                previous_word_idx = idx
        all_labels.append(labels)
    return all_labels

def compute_metrics(p):
    """
    Simple function to compute the accuracy of the BERT model.
    """
    metric = ev.load('accuracy')
    pred, labels = p
    pred = np.argmax(pred, axis=2)
    # ignore predictions where we don't have a valid label
    mask = labels != -100
    pred = pred[mask]
    labels = labels[mask]
    return metric.compute(predictions=pred, references=labels)
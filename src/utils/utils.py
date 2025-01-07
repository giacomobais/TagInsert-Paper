
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
from src.models.VanillaTransformer import make_model

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

def load_model(path, config):
    word_to_idx = json.load(open('data/POS/processed/word_to_idx.json'))
    POS_to_idx = json.load(open('data/POS/processed/POS_to_idx.json'))
    model = make_model(len(word_to_idx), len(POS_to_idx), d_model = 768, N=config['n_heads'])
    model = model.to(config['device'])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.5, betas=(0.9, 0.98), eps=1e-9) #TODO: change hyperparameters with config
    lr_scheduler = LambdaLR(optimizer=optimizer,lr_lambda=lambda step: rate(step, model_size=model.src_embed[0].d_model, factor=1.0, warmup=400),)
    checkpoint = torch.load(path)
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

class Batch:
    """Object for holding a batch of data with mask during training."""

    def __init__(self, src, tgt=None, embs = None, pad=0):  # 2 = <blank>
        self.src = src
        self.src_mask = (src != pad).unsqueeze(-2)
        self.embs = embs
        if tgt is not None:
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

class SimpleLossCompute:
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

def resume_training(config):
    try:
        model, model_opt, lr_scheduler, logs = load_model("models/VanillaTransformer_POS")
        trained_epochs = logs['epochs']
        train_losses = logs['train_losses']
        val_losses = logs['val_losses']
        print('Loading a pre-existent model.')
    except:
        word_to_idx = json.load(open('data/POS/processed/word_to_idx.json'))
        POS_to_idx = json.load(open('data/POS/processed/POS_to_idx.json'))
        model = make_model(len(word_to_idx), len(POS_to_idx), d_model = 768, N=config['n_heads']) #TODO: change hyperparameters with config
        model_opt = torch.optim.Adam(model.parameters(), lr=0.5, betas=(0.9, 0.98), eps=1e-9)
        lr_scheduler = LambdaLR(optimizer=model_opt,lr_lambda=lambda step: rate(step, model_size=model.src_embed[0].d_model, factor=1.0, warmup=400),)
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



def get_embs(original_sentences, bert_model, tokenizer, config):
    src_embs = extract_BERT_embs(original_sentences, bert_model, tokenizer, config)
    return src_embs


def dataload(data_loader, config, bert_model, tokenizer, pad=0):
    for batch_data in data_loader:
        xb, yb, original_sentences = batch_data
        embs = get_embs(original_sentences, bert_model, tokenizer, config)
        src = xb.to(config['device'])
        tgt = yb.to(config['device'])
        embs = embs.to(config['device'])
        yield Batch(src, tgt, embs, pad=pad)


def run_epoch(data_iter, model, loss_compute, optimizer, lr_scheduler, config, data_len, mode = 'train', accum_iter = 1, train_state = TrainState()):
    total_tokens = 0
    total_loss = 0
    n_accum = 0
    losses = []
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
                    n_accum += 1
                    train_state.accum_step += 1
                lr_scheduler.step()
            total_loss += loss
            total_tokens += batch.ntokens
            log_id = "Batch train loss" if mode == "train" else "Batch val loss"
            wandb.log({log_id: loss_node.item()})
            pbar.set_postfix(loss=loss_node.item())

    return total_loss / total_tokens, train_state, losses

def collate_fn(batch):
    words, tags, original_sentences = zip(*batch)
    words = torch.stack(words)
    tags = torch.stack(tags)
    return words, tags, original_sentences

def train(config):
    with wandb.init(project="TagInsert", config=config):
        model, model_opt, lr_scheduler, trained_epochs, train_losses, val_losses = resume_training(config)
        bert_model, tokenizer = load_BERT_encoder(config['bert_model'], config['device'])
        POS_to_idx = json.load(open('data/POS/processed/POS_to_idx.json'))
        criterion = LabelSmoothing(size=len(POS_to_idx), padding_idx=0, smoothing=0.0)
        train_dataset = TaggingDataset("data/POS/processed/train_data.pth")
        val_dataset = TaggingDataset("data/POS/processed/val_data.pth")
        train_dataloader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, collate_fn=collate_fn)
        val_dataloader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=True, collate_fn=collate_fn)
        wandb.watch(model, log="all")
        for epoch in range(trained_epochs, config['epochs']):
            print(f"Epoch {epoch}")
            epoch_data_iter = dataload(train_dataloader, config, bert_model, tokenizer, pad=0)
            val_epoch_data_iter = dataload(val_dataloader, config, bert_model, tokenizer, pad=0)
            model.train()
            train_losses = run_epoch(epoch_data_iter, model, SimpleLossCompute(model.generator, criterion), model_opt, lr_scheduler, config, len(train_dataset), mode = 'train')[2]
            with torch.no_grad():
                model.eval()
                val_losses = run_epoch(val_epoch_data_iter, model, SimpleLossCompute(model.generator, criterion), DummyOptimizer(),DummyScheduler(), config, len(val_dataset), mode = 'eval')[2]
            wandb.log({"Epoch": epoch, "train_loss": np.mean(train_losses) , "val_loss": np.mean(val_losses)}, step = epoch)
    wandb.finish()
    return model, model_opt, lr_scheduler, train_losses, val_losses, epoch
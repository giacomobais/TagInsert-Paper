import copy
from torch import nn
import torch.nn.functional as F
import json
from src.models.VanillaTransformer import MultiHeadedAttention, PositionwiseFeedForward, PositionalEncoding, Embeddings, Encoder, EncoderLayer, Decoder, DecoderLayer, POS_Embeddings

class TagInsert(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many
    other models.
    """
    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(TagInsert, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

    def forward(self, src, tgt, src_mask, tgt_mask, embs):
      "Take in and process masked src and target sequences."
      out = self.decode(self.encode(src, src_mask, embs), src, embs, src_mask, tgt, tgt_mask)
      return out

    def encode(self, src, src_mask, embs):
      emb = self.src_embed(embs)
      out = self.encoder(emb, src_mask)
      return out

    def decode(self, memory, src, embs, src_mask, tgt, tgt_mask):
        src_emb = self.src_embed(embs)
        emb = self.tgt_embed(tgt)
        emb = self.addsrc(emb, src_emb)
        out = self.decoder(emb, memory, src_mask, tgt_mask)
        # print(out.shape)
        return out

    def addsrc(self, emb, src_emb):
        emb[:, 1:] += src_emb[:, :-1]
        return emb
    
class GeneratorTagInsert(nn.Module):
    "Define standard linear + softmax generation step."
    def __init__(self, d_model, vocab):
        super(GeneratorTagInsert, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        out = F.softmax(self.proj(x), dim=-1)
        return out

def make_model_TI(tagging, N=8, d_model=768, d_ff=768*4, h=8, dropout=0.1):
    "Helper: Construct a model from hyperparameters."
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)
    if tagging == "POS":
        word_to_idx = json.load(open("data/POS/processed/100%/word_to_idx.json"))
        src_vocab = len(word_to_idx)
        POS_to_idx = json.load(open("data/POS/processed/100%/POS_to_idx.json"))
        tgt_vocab = len(POS_to_idx)
    elif tagging == "CCG":
        word_to_idx = json.load(open("data/CCG/processed/100%/word_to_idx.json"))
        src_vocab = len(word_to_idx)
        CCG_to_idx = json.load(open("data/CCG/processed/100%/CCG_to_idx.json"))
        tgt_vocab = len(CCG_to_idx)
    model = TagInsert(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N),
        nn.Sequential(Embeddings(d_model, src_vocab), c(position)),
        nn.Sequential(POS_Embeddings(d_model, tgt_vocab), c(position)),
        GeneratorTagInsert(d_model, tgt_vocab)
        )
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model
import logging
import torch.nn as nn
import pandas as pd
import torch

from allennlp.common import Params
from allennlp.models.model import Model
from allennlp.modules import Highway
from allennlp.modules import TimeDistributed
from allennlp.nn import util, InitializerApplicator
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.token_embedders import Embedding
from allennlp.modules.seq2seq_encoders import Seq2SeqEncoder as s2s_e
import warnings
warnings.filterwarnings("ignore", category=UserWarning, message="RNN module weights are not part of single contiguous chunk of memory")


def build_model(args, vocab, pretrained_embs):
    if args.glove:
        word_embs, train_embs = pretrained_embs, bool(args.train_words)
    else:
        logging.info("Learning embeddings from scratch!")
        word_embs, train_embs = None, True

    word_embedder = Embedding(
        embedding_dim=args.d_word,
        num_embeddings=vocab.get_vocab_size("tokens"),
        weight=word_embs,
        trainable=train_embs,
        padding_index=vocab.get_token_index("@@PADDING@@")
    )

    text_field_embedder = BasicTextFieldEmbedder({"words": word_embedder})

    phrase_layer = s2s_e.by_name("lstm").from_params(Params({
        "input_size": args.d_word,
        "hidden_size": args.d_hid,
        "num_layers": args.n_layers_enc,
        "bidirectional": True
    }))

    pair_encoder = HeadlessPairEncoder(
        vocab, text_field_embedder, args.n_layers_highway,
        phrase_layer, dropout=args.dropout
    )

    d_task = 8 * args.d_hid
    return MultiTaskModel(pair_encoder, d_task)


class MultiTaskModel(nn.Module):
    def __init__(self, pair_encoder, d_task):
        super().__init__()
        self.pair_encoder = pair_encoder
        self.pred_layer = nn.Linear(d_task, 1)
        self.pred_layer_1 = nn.Linear(d_task, 1)   
        self.redu_layer = nn.Linear(8192, 64)      

    def forward(self, input1=None, input2=None, mask1=None, mask2=None, return_feat=False, **kwargs):
        pair_emb = self.pair_encoder(input1, input2, mask1, mask2)
        logits = self.pred_layer(pair_emb)

        out = {"logits": logits}
        if return_feat:
            out["feat"] = pair_emb
        return out


class HeadlessPairEncoder(Model): 
    def __init__(self, vocab, text_field_embedder, num_highway_layers, phrase_layer,
                 dropout=0.2, mask_lstms=True, initializer=InitializerApplicator()):
        super(HeadlessPairEncoder, self).__init__(vocab)
        
        d_emb = text_field_embedder.get_output_dim()
        self._text_field_embedder = text_field_embedder
        self._highway_layer = TimeDistributed(Highway(d_emb, num_highway_layers))
        self._phrase_layer = phrase_layer
        self.pad_idx = vocab.get_token_index(vocab._padding_token)
        self.output_dim = phrase_layer.get_output_dim()  
        self._mask_lstms = mask_lstms
        if dropout > 0:
            self._dropout = nn.Dropout(p=dropout)
        else:
            self._dropout = lambda x: x
            
        initializer(self)

    def forward(self, s1, s2, m1=None, m2=None):
        s1_embs = self._highway_layer(self._text_field_embedder(s1) if m1 is None else s1)  
        s2_embs = self._highway_layer(self._text_field_embedder(s2) if m2 is None else s2)
        
        s1_embs = self._dropout(s1_embs)
        s2_embs = self._dropout(s2_embs)

        s1_mask = util.get_text_field_mask(s1) if m1 is None else m1.long()
        s2_mask = util.get_text_field_mask(s2) if m2 is None else m2.long()

        s1_lstm_mask = s1_mask.float() if self._mask_lstms else None
        s2_lstm_mask = s2_mask.float() if self._mask_lstms else None

        s1_enc = self._phrase_layer(s1_embs, s1_lstm_mask) 
        s2_enc = self._phrase_layer(s2_embs, s2_lstm_mask)

        s1_enc = self._dropout(s1_enc)
        s2_enc = self._dropout(s2_enc)

        s1_mask = s1_mask.unsqueeze(dim=-1)
        s2_mask = s2_mask.unsqueeze(dim=-1)

        s1_enc.data.masked_fill_((1 - s1_mask.byte().data).bool(), -float('inf'))
        s2_enc.data.masked_fill_((1 - s2_mask.byte().data).bool(), -float('inf'))

        s1_enc, _ = s1_enc.max(dim=1)
        s2_enc, _ = s2_enc.max(dim=1)

        return torch.cat([s1_enc, s2_enc, torch.abs(s1_enc - s2_enc), s1_enc * s2_enc], 1)
    
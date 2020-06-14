# -*- coding: utf-8 -*-
#!/usr/bin/env python

import torch
import torch.nn as nn
from fastai.text.models import AWD_LSTM, RNNDropout

class Shared_Langage_Model(nn.Module):

    def __init__(self, n_layer, emb_size, h_size, dr_rate, vocab_dict,*args):
        super().__init__()

        Max_Word_idx = max(vocab_dict.id2vocab_input[-1].keys())+1 #[0,1,2,3], max = 3, idx_len = 3+1

        ####______shared params_______####
        self.dr_rate = dr_rate
        self.Ws_share = nn.Linear(emb_size, 1, bias=False) #W for EOS

        self.rnn_fwd = AWD_LSTM(
            vocab_sz=Max_Word_idx,
            emb_sz=emb_size,
            n_hid=h_size,
            n_layers=n_layer,
            hidden_p=dr_rate,
            input_p=0,
            embed_p=0,
            weight_p=0,
            qrnn=True
        )

        self.rnn_bkw = AWD_LSTM(
            vocab_sz=Max_Word_idx,
            emb_sz=emb_size,
            n_hid=h_size,
            n_layers=n_layer,
            hidden_p=dr_rate,
            input_p=0,
            embed_p=0,
            weight_p=0,
            qrnn=True
        )

        self.dropout = RNNDropout(p=dr_rate)
        ####______shared params_______####

        ####______specific params_______####
        self.emb = nn.Embedding(Max_Word_idx, emb_size, padding_idx= vocab_dict.vocab2id_input[0]["<PAD>"]) #lookup table for all languages
        layer = []
        for lang in range(len(vocab_dict.id2vocab_output)):
            layer.append(nn.Linear(h_size, vocab_dict.V_size[lang]-1, bias=False))

        self.Ws_i = nn.ModuleList(layer)
        ####______specific params_______####

    def __call__(self, BOS_t_id, t_lengths, *args):
        return self.forward(BOS_t_id, t_lengths, *args)

    def Switch_Lang(self, lang):

        self.lang = lang #switch output layer


    def Switch_fwdbkw(self,type):
        if (type == "fwd"):
            self.rnn = self.rnn_fwd

        elif (type == "bkw"):
            self.rnn = self.rnn_bkw

        else:
            raise Exception("Invalid type")

    def forward(self,input_id, input_id_len, *args):

        ht = self.decode(input_id, input_id_len, *args)
        score_V = self.Ws_i[self.lang](self.dropout(ht))
        score_eos = self.Ws_share(self.dropout(ht))
        score = torch.cat([score_eos, score_V], dim=2)  # (bs, maxlen_t, tgtV)
        return score

    def decode(self, input_id, input_id_len, *args):
        input_id_emb = self.emb(input_id)  #bs, max_s_len, * emb_size
        ht, _ = self.rnn(input_id_emb, from_embeddings=True)  # ht: bt * len_t * demb(最上位層の出力 from each hidden state)
        return  ht[-1]

    def set_device(self,is_cuda):
        if is_cuda:
            self.torch = torch.cuda
        else:
            self.torch = torch

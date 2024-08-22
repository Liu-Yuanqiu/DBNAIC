import math
import torch
from torch import nn
from torch.nn import functional as F
import copy
import numpy as np

class ScaledDotProductAttention(nn.Module):
    '''
    Scaled dot-product attention
    '''

    def __init__(self, d_model, d_k, d_v, h, dropout=.1):
        '''
        :param d_model: Output dimensionality of the model
        :param d_k: Dimensionality of queries and keys
        :param d_v: Dimensionality of values
        :param h: Number of heads
        '''
        super(ScaledDotProductAttention, self).__init__()
        self.fc_q = nn.Linear(d_model, h * d_k)
        self.fc_k = nn.Linear(d_model, h * d_k)
        self.fc_v = nn.Linear(d_model, h * d_v)
        self.fc_o = nn.Linear(h * d_v, d_model)
        self.dropout = nn.Dropout(dropout)

        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.h = h

        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.fc_q.weight)
        nn.init.xavier_uniform_(self.fc_k.weight)
        nn.init.xavier_uniform_(self.fc_v.weight)
        nn.init.xavier_uniform_(self.fc_o.weight)
        nn.init.constant_(self.fc_q.bias, 0)
        nn.init.constant_(self.fc_k.bias, 0)
        nn.init.constant_(self.fc_v.bias, 0)
        nn.init.constant_(self.fc_o.bias, 0)

    def forward(self, queries, keys, values, attention_mask=None, attention_weights=None):
        '''
        Computes
        :param queries: Queries (b_s, nq, d_model)
        :param keys: Keys (b_s, nk, d_model)
        :param values: Values (b_s, nk, d_model)
        :param attention_mask: Mask over attention values (b_s, h, nq, nk). True indicates masking.
        :param attention_weights: Multiplicative weights for attention values (b_s, h, nq, nk).
        :return:
        '''
        b_s, nq = queries.shape[:2]
        nk = keys.shape[1]

        q = self.fc_q(queries).view(b_s, nq, self.h, self.d_k).permute(0, 2, 1, 3)  # (b_s, h, nq, d_k)
        k = self.fc_k(keys).view(b_s, nk, self.h, self.d_k).permute(0, 2, 3, 1)  # (b_s, h, d_k, nk)
        v = self.fc_v(values).view(b_s, nk, self.h, self.d_v).permute(0, 2, 1, 3)  # (b_s, h, nk, d_v)

        att = torch.matmul(q, k) / np.sqrt(self.d_k)  # (b_s, h, nq, nk)
        if attention_weights is not None:
            att = att * attention_weights
        if attention_mask is not None:
            att = att.masked_fill(attention_mask, -np.inf)
        att = torch.softmax(att, -1)
        att = self.dropout(att)

        out = torch.matmul(att, v).permute(0, 2, 1, 3).contiguous().view(b_s, nq, self.h * self.d_v)  # (b_s, nq, h*d_v)
        out = self.fc_o(out)  # (b_s, nq, d_model)
        return out

class MultiHeadAttention(nn.Module):
    '''
    Multi-head attention layer with Dropout and Layer Normalization.
    '''

    def __init__(self, d_model, d_k, d_v, h, dropout=.1, identity_map_reordering=False):
        super(MultiHeadAttention, self).__init__()
        self.identity_map_reordering = identity_map_reordering
        self.attention = ScaledDotProductAttention(d_model=d_model, d_k=d_k, d_v=d_v, h=h)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, queries, keys, values, attention_mask=None, attention_weights=None):
        if self.identity_map_reordering:
            q_norm = self.layer_norm(queries)
            k_norm = self.layer_norm(keys)
            v_norm = self.layer_norm(values)
            out = self.attention(q_norm, k_norm, v_norm, attention_mask, attention_weights)
            out = queries + self.dropout(torch.relu(out))
        else:
            out = self.attention(queries, keys, values, attention_mask, attention_weights)
            out = self.dropout(out)
            out = self.layer_norm(queries + out)
        return out

class PositionWiseFeedForward(nn.Module):
    '''
    Position-wise feed forward layer
    '''

    def __init__(self, d_model=512, d_ff=2048, dropout=.1, identity_map_reordering=False):
        super(PositionWiseFeedForward, self).__init__()
        self.identity_map_reordering = identity_map_reordering
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(p=dropout)
        self.dropout_2 = nn.Dropout(p=dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, input):
        if self.identity_map_reordering:
            out = self.layer_norm(input)
            out = self.fc2(self.dropout_2(F.relu(self.fc1(out))))
            out = input + self.dropout(torch.relu(out))
        else:
            out = self.fc2(self.dropout_2(F.relu(self.fc1(input))))
            out = self.dropout(out)
            out = self.layer_norm(input + out)
        return out

class EncoderLayer(nn.Module):
    def __init__(self, d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=.1, identity_map_reordering=False):
        super(EncoderLayer, self).__init__()
        self.mhatt = MultiHeadAttention(d_model, d_k, d_v, h, dropout, identity_map_reordering=identity_map_reordering)
        self.dropout = nn.Dropout(dropout)
        self.lnorm = nn.LayerNorm(d_model)
        self.pwff = PositionWiseFeedForward(d_model, d_ff, dropout, identity_map_reordering=identity_map_reordering)

    def forward(self, queries, keys, values, attention_mask=None, attention_weights=None):
        att = self.mhatt(queries, keys, values, attention_mask, attention_weights)
        att = self.lnorm(queries + self.dropout(att))
        ff = self.pwff(att)
        return ff

class DecoderLayer(nn.Module):
    def __init__(self, d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=.1):
        super(DecoderLayer, self).__init__()
        self.self_att = MultiHeadAttention(d_model, d_k, d_v, h, dropout)
        self.enc_att = MultiHeadAttention(d_model, d_k, d_v, h, dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.lnorm1 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)
        self.lnorm2 = nn.LayerNorm(d_model)
        self.pwff = PositionWiseFeedForward(d_model, d_ff, dropout)

    def forward(self, input, enc_output, mask_cross=None):
        self_att = self.self_att(input, input, input)
        self_att = self.lnorm1(input + self.dropout1(self_att))
        enc_att = self.enc_att(self_att, enc_output, enc_output, mask_cross)
        enc_att = self.lnorm2(self_att + self.dropout2(enc_att))
        ff = self.pwff(enc_att)
        return ff

class DLNA_KDXE(nn.Module):
    def __init__(self, N, d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=0.1, 
                max_len=17, vocab_size=9587, identity_map_reordering=False, attention_module=None, attention_module_kwargs=None):
        super(DLNA_KDXE, self).__init__()
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.h = h
        self.d_ff = d_ff
        self.dropout = dropout
        self.max_len = max_len
        self.vocab_size = vocab_size

        self.fc_region_feature = nn.Linear(512, self.d_model)
        self.dropout_region = nn.Dropout(p=self.dropout)
        self.layer_norm_region = nn.LayerNorm(self.d_model)
        self.fc_grid_feature = nn.Linear(1024, self.d_model)
        self.dropout_grid = nn.Dropout(p=self.dropout)
        self.layer_norm_grid = nn.LayerNorm(self.d_model)

        self.encoder_region = nn.ModuleList([EncoderLayer(d_model, d_k, d_v, h, d_ff, dropout, identity_map_reordering=identity_map_reordering) for _ in range(N)])
        self.encoder_grid = nn.ModuleList([EncoderLayer(d_model, d_k, d_v, h, d_ff, dropout, identity_map_reordering=identity_map_reordering) for _ in range(N)])

        self.region2text = nn.Linear(self.d_model, self.d_model)
        self.grid2text = nn.Linear(self.d_model, self.d_model)
        self.region_clip = self.get_clip_mat(self.max_len, 150).cuda().unsqueeze(0)
        self.grid_clip = self.get_clip_mat(self.max_len, 60).cuda().unsqueeze(0)
        self.word_embed = nn.Embedding(vocab_size, self.d_model)

        self.decoder_region = nn.ModuleList([DecoderLayer(d_model, d_k, d_v, h, d_ff, dropout) for _ in range(N)])
        self.decoder_grid = nn.ModuleList([DecoderLayer(d_model, d_k, d_v, h, d_ff, dropout) for _ in range(N)])
        self.fc_region = nn.Linear(d_model, vocab_size, bias=False)
        self.fc_grid = nn.Linear(d_model, vocab_size, bias=False)

    def get_clip_mat(self, length, size):
        w = torch.zeros((length, size), dtype=torch.float32)
        t = 0.5
        for j in range(length):
            for i in range(size):
                w[j][i] = np.exp(-(j-i*length/size)**2/t)
        w = w / torch.full(w.shape, size, dtype=torch.float32)
        return w

    def encoder_pre(self, region, grid):
        region = F.relu(self.fc_region_feature(region))
        region = self.dropout_region(region)
        region = self.layer_norm_region(region)
        grid = F.relu(self.fc_grid_feature(grid))
        grid = self.dropout_grid(grid)
        grid = self.layer_norm_grid(grid)
        return region, grid

    def encoder(self, region, region_mask, grid, grid_mask):
        out_region = region
        out_grid = grid
        for l_region, l_grid in zip(self.encoder_region, self.encoder_grid):
            out_region = l_region(out_region, out_region, out_region, region_mask)
            out_grid = l_grid(out_grid, out_grid, out_grid, grid_mask)
        return out_region, out_grid

    def decoder_pre(self, out_region, region_mask, out_grid, grid_mask):
        vocab = self.word_embed.weight
        text_region = self.region2text(out_region)
        text_region_prob =  F.log_softmax(text_region @ vocab.t(), dim=-1)
        _, word_region = torch.max(text_region_prob, -1)
        region = text_region + self.word_embed(word_region)
        region = torch.matmul(self.region_clip, region)

        text_grid = self.grid2text(out_grid)
        text_grid_prob = F.log_softmax(text_grid @ vocab.t(), dim=-1)
        _, word_grid = torch.max(text_grid_prob, -1)
        grid = text_grid + self.word_embed(word_grid)
        grid = torch.matmul(self.grid_clip, grid)
        return region, grid

    def decoder(self, text_region, text_grid, out_region, out_grid, region_mask, grid_mask):
        for l_region, l_grid in zip(self.decoder_region, self.decoder_grid):
            text_region = l_region(text_region, out_region, region_mask)
            text_grid = l_grid(text_grid, out_grid, grid_mask)
        dec_region = self.fc_region(text_region)
        dec_grid = self.fc_grid(text_grid)
        return F.log_softmax(dec_region, dim=-1), F.log_softmax(dec_grid, dim=-1)

    def forward(self, region, region_mask, grid, grid_mask):
        region, grid = self.encoder_pre(region, grid)
        out_region, out_grid = self.encoder(region, region_mask, grid, grid_mask)
        text_region, text_grid = self.decoder_pre(out_region, region_mask, out_grid, grid_mask)
        logit_region, logit_grid = self.decoder(text_region, text_grid, out_region, out_grid, region_mask, grid_mask)
        return logit_region, logit_grid
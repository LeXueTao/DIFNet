import torch
from torch import nn
from torch.nn import functional as F

from .attention import MultiHeadAttention
from .utils import sinusoid_encoding_table, PositionWiseFeedForward
from models.containers import Module, ModuleList
from ..layers_lrp import *
from collections import defaultdict


class DecoderLayer(Module):
    def __init__(self, d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=.1, self_att_module=None,
                 enc_att_module=None, self_att_module_kwargs=None, enc_att_module_kwargs=None):
        super(DecoderLayer, self).__init__()
        self.self_att = MultiHeadAttention(d_model, d_k, d_v, h, dropout, can_be_stateful=True,
                                           attention_module=self_att_module,
                                           attention_module_kwargs=self_att_module_kwargs, beam_flag=True)
        self.enc_att = MultiHeadAttention(d_model, d_k, d_v, h, dropout, can_be_stateful=False,
                                          attention_module=enc_att_module,
                                          attention_module_kwargs=enc_att_module_kwargs)

        self.dropout1 = Dropout(dropout)
        self.lnorm1 = LayerNorm(d_model)

        self.dropout2 = Dropout(dropout)
        self.lnorm2 = LayerNorm(d_model)
        self.add1 = Add()
        self.clone1 = Clone()
        self.add2 = Add()
        self.clone2 = Clone()
        self.clone3 = Clone()
        self.pwff = PositionWiseFeedForward(d_model, d_ff, dropout)

    def forward(self, input, enc_output, mask_pad, mask_self_att, mask_enc_att):
        # MHA+AddNorm
        res1, input_q, input_k, input_v = self.clone1(input, 4)
        self_att = self.self_att(input_q, input_k, input_v, mask_self_att)
        self_att = self.lnorm1(self.add1([res1, self.dropout1(self_att)]))
        self_att = self_att * mask_pad
        # MHA+AddNorm
        res2, self_att = self.clone2(self_att, 2)
        enc_output_k, enc_output_v = self.clone3(enc_output, 2)
        enc_att = self.enc_att(self_att, enc_output_k, enc_output_v, mask_enc_att)
        enc_att = self.lnorm2(self.add2([res2, self.dropout2(enc_att)]))
        enc_att = enc_att * mask_pad
        # FFN+AddNorm
        ff = self.pwff(enc_att)
        ff = ff * mask_pad
        return ff
    
    def relprop(self, R, **kwargs):
        R = self.pwff.relprop(R, **kwargs)
        R = self.lnorm2.relprop(R, **kwargs)

        (R2, R_ea) = self.add2.relprop(R, **kwargs)
        R_ea = self.dropout2.relprop(R_ea, **kwargs)
        R_sa, R_eak, R_eav = self.enc_att.relprop(R_ea, **kwargs)
        R_e = self.clone3.relprop((R_eak, R_eav), **kwargs)
        R_sa = self.clone2.relprop((R2, R_sa), **kwargs)

        R_sa = self.lnorm1.relprop(R_sa, **kwargs)
        (R1, R_sa) = self.add1.relprop(R_sa, **kwargs)
        R_sa = self.dropout1.relprop(R_sa, **kwargs)
        R_iq, R_ik, R_iv = self.self_att.relprop(R_sa, **kwargs)
        R_i = self.clone1.relprop((R1, R_iq, R_ik, R_iv), **kwargs)
        
        return R_i, R_e


class TransformerDecoder_LRP(Module):
    def __init__(self, vocab_size, max_len, N_dec, padding_idx, d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=.1,
                 self_att_module=None, enc_att_module=None, self_att_module_kwargs=None, enc_att_module_kwargs=None):
        super(TransformerDecoder_LRP, self).__init__()
        self.d_model = d_model
        self.word_emb = nn.Embedding(vocab_size, d_model, padding_idx=padding_idx)
        self.pos_emb = nn.Embedding.from_pretrained(sinusoid_encoding_table(max_len + 1, d_model, 0), freeze=True)
        self.layers = ModuleList(
            [DecoderLayer(d_model, d_k, d_v, h, d_ff, dropout, self_att_module=self_att_module, enc_att_module=enc_att_module, self_att_module_kwargs=self_att_module_kwargs, enc_att_module_kwargs=enc_att_module_kwargs) for _ in range(N_dec)])
        self.fc = Linear(d_model, vocab_size, bias=False)
        self.max_len = max_len
        self.padding_idx = padding_idx
        self.N = N_dec
        self.default_state = defaultdict(int)
        self.register_state('running_mask_self_attention', torch.zeros((1, 1, 0)).byte())
        self.register_state('running_seq', torch.zeros((1,)).long())

    def forward(self, input, encoder_output, mask_encoder):
        # input (b_s, seq_len)
        b_s, seq_len = input.shape[:2]
        mask_queries = (input != self.padding_idx).unsqueeze(-1).float()  # (b_s, seq_len, 1)
        mask_self_attention = torch.triu(torch.ones((seq_len, seq_len), dtype=torch.uint8, device=input.device),
                                         diagonal=1)
        mask_self_attention = mask_self_attention.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, seq_len)
        mask_self_attention = mask_self_attention + (input == self.padding_idx).unsqueeze(1).unsqueeze(1).byte()
        mask_self_attention = mask_self_attention.gt(0)  # (b_s, 1, seq_len, seq_len)
        if self._is_stateful:
            self.running_mask_self_attention = torch.cat([self.running_mask_self_attention.type_as(mask_self_attention), mask_self_attention], -1)
            mask_self_attention = self.running_mask_self_attention
        
        if len(self.running_mask_self_attention.shape) == 4:
            self.running_mask_self_attention = torch.cat([self.running_mask_self_attention.type_as(mask_self_attention), mask_self_attention], -1)
            mask_self_attention = self.running_mask_self_attention

        seq = torch.arange(1, seq_len + 1).view(1, -1).expand(b_s, -1).to(input.device)  # (b_s, seq_len)
        seq = seq.masked_fill(mask_queries.squeeze(-1) == 0, 0)
        if self._is_stateful:
            self.running_seq.add_(1)
            seq = self.running_seq
        
        if len(self.running_seq.shape) == 2:
            self.running_seq.add_(1)
            seq = self.running_seq

        out = self.word_emb(input) + self.pos_emb(seq)
        
        for i, l in enumerate(self.layers):
            out = l(out, encoder_output, mask_queries, mask_self_attention, mask_encoder)

        out = self.fc(out)
        return F.log_softmax(out, dim=-1)
    
    def relprop(self, R, **kwargs):
        R_enc = 0.0
        R_enc_scale = 0.0
        R_o = self.fc.relprop(R, **kwargs)
        for blk in reversed(self.layers):
            R_o, R_e = blk.relprop(R_o, **kwargs)
            R_enc += R_e
            # R_enc_scale += (abs(R_e)).sum()
        # return R_o, R_enc * R_enc_scale / (abs(R_enc)).sum()
        return R_o, R_enc

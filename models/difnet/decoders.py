import torch
from torch import nn
from torch.nn import functional as F

from .attention import MultiHeadAttention
from .utils import sinusoid_encoding_table, PositionWiseFeedForward
from models.containers import Module, ModuleList


class DecoderLayer(Module):
    def __init__(self, d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=.1, self_att_module=None,
                 enc_att_module=None, self_att_module_kwargs=None, enc_att_module_kwargs=None):
        super(DecoderLayer, self).__init__()
        self.self_att = MultiHeadAttention(d_model, d_k, d_v, h, dropout, can_be_stateful=True,
                                           attention_module=self_att_module,
                                           attention_module_kwargs=self_att_module_kwargs)
        self.enc_att = MultiHeadAttention(d_model, d_k, d_v, h, dropout, can_be_stateful=False,
                                          attention_module=enc_att_module,
                                          attention_module_kwargs=enc_att_module_kwargs)

        self.dropout1 = nn.Dropout(dropout)
        self.lnorm1 = nn.LayerNorm(d_model)

        self.dropout2 = nn.Dropout(dropout)
        self.lnorm2 = nn.LayerNorm(d_model)
        self.pwff = PositionWiseFeedForward(d_model, d_ff, dropout)

    def forward(self, input, enc_output, mask_pad, mask_self_att, mask_enc_att):
        # MHA+AddNorm
        self_att = self.self_att(input, input, input, mask_self_att)
        self_att = self.lnorm1(input + self.dropout1(self_att))
        self_att = self_att * mask_pad
        # MHA+AddNorm
        enc_att = self.enc_att(self_att, enc_output, enc_output, mask_enc_att)
        enc_att = self.lnorm2(self_att + self.dropout2(enc_att))
        enc_att = enc_att * mask_pad
        # FFN+AddNorm
        ff = self.pwff(enc_att)
        ff = ff * mask_pad
        return ff


class DifnetDecoder(Module):
    def __init__(self, vocab_size, max_len, N_dec, padding_idx, d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=.1,
                 self_att_module=None, enc_att_module=None, self_att_module_kwargs=None, enc_att_module_kwargs=None):
        super(DifnetDecoder, self).__init__()
        self.d_model = d_model
        self.word_emb = nn.Embedding(vocab_size, d_model, padding_idx=padding_idx)
        self.pos_emb = nn.Embedding.from_pretrained(sinusoid_encoding_table(max_len + 1, d_model, 0), freeze=True)
        self.layers = ModuleList(
            [DecoderLayer(d_model, d_k, d_v, h, d_ff, dropout, self_att_module=self_att_module, enc_att_module=enc_att_module, self_att_module_kwargs=self_att_module_kwargs, enc_att_module_kwargs=enc_att_module_kwargs) for _ in range(N_dec)])
        self.fc = nn.Linear(d_model, vocab_size, bias=False)
        self.max_len = max_len
        self.padding_idx = padding_idx
        self.N = N_dec

        self.register_state('running_mask_self_attention', torch.zeros((1, 1, 0)).byte())
        self.register_state('running_seq', torch.zeros((1,)).long())

    def forward(self, input, encoder_output, mask_encoder):
        # input (b_s, seq_len)
        b_s, seq_len = input.shape[:2]
        # (b_s, seq_len, 1) 找到pad的位置开头/尾巴
        mask_queries = (input != self.padding_idx).unsqueeze(-1).float()  
        # (seq_len, seq_len)
        mask_self_attention = torch.triu(torch.ones((seq_len, seq_len), dtype=torch.uint8, device=input.device),diagonal=1)
        # (1, 1, seq_len, seq_len)
        mask_self_attention = mask_self_attention.unsqueeze(0).unsqueeze(0)  
        # (1, 1, seq_len, seq_len) + (b_s, 1, 1, seq_len)，针对将'<pad>'符号前置的情况，后置不会影响
        mask_self_attention = mask_self_attention + (input == self.padding_idx).unsqueeze(1).unsqueeze(1).byte()
         # (b_s, 1, seq_len, seq_len)
        mask_self_attention = mask_self_attention.gt(0) 
        if self._is_stateful: # (1, 1, 1, 0)值是空的 (b_s, 1, seq_len, seq_len)
            self.running_mask_self_attention = torch.cat([self.running_mask_self_attention.type_as(mask_self_attention), mask_self_attention], -1)
            mask_self_attention = self.running_mask_self_attention

        # (b_s, seq_len), '<pad>'位置填入0
        seq = torch.arange(1, seq_len + 1).view(1, -1).expand(b_s, -1).to(input.device)  
        seq = seq.masked_fill(mask_queries.squeeze(-1) == 0, 0)
        if self._is_stateful:
            self.running_seq.add_(1) # (1, 1) 0
            seq = self.running_seq
        # seq是为了位置嵌入
        out = self.word_emb(input) + self.pos_emb(seq)
        
        for i, l in enumerate(self.layers):
            # mask_queries限制q，可以将q加到k进一步限制
            # mask_self_attention通过注意力矩阵赋值'-inf'限制k
            out = l(out, encoder_output, mask_queries, mask_self_attention, mask_encoder)

        out = self.fc(out)
        return F.log_softmax(out, dim=-1)

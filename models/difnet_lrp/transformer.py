import torch
from torch import nn
import copy
from models.containers import ModuleList
from ..captioning_model1 import CaptioningModel1
from ..layers_lrp import *

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten,self).__init__()
 
    def forward(self, x):
        return x.view(x.size(0), 133, -1).permute(0, 2, 1)

class Difnet_LRP(CaptioningModel1):
    def __init__(self, bos_idx, encoder, decoder):
        super(Difnet_LRP, self).__init__()
        self.bos_idx = bos_idx
        self.encoder = encoder
        self.decoder = decoder
        self.buffer_module = list()
        # image embed
        self.embed_image = nn.Sequential(
                nn.Linear(2048, 512),
                nn.ReLU(),
                nn.Dropout(p=0.1),
                nn.LayerNorm(512))
        self.embed_pixel = nn.Sequential(
                # nn.AdaptiveAvgPool2d((7, 7)),
                Flatten(),
                nn.Linear(133, self.decoder.d_model),
                # nn.Sigmoid(),
                nn.ReLU(),
                nn.Dropout(p=0.1),
                nn.LayerNorm(self.decoder.d_model))
        # self.register_state('enc_output', None)
        # self.register_state('mask_enc', None)
        self.init_weights()

    def forward(self, images, seq, pixel, *args):
        images = self.embed_image(images)
        pixel = self.embed_pixel(pixel)
        enc_output, mask_enc = self.encoder(images, pixel)
        dec_output = self.decoder(seq, enc_output, mask_enc)
        return dec_output


    def init_state_1(self, batch_size):
        for mm in self.modules():
            # 这个模块有buffer
            if bool(mm._buffers):
                self.buffer_module.append(mm)
                for name in mm._buffers:
                    # 扩展第0位，batch_size*beam_size
                    mm.default_state[name] = mm._buffers[name]
                    mm._buffers[name] = mm._buffers[name].clone().detach().to(mm._buffers[name].device)
                    mm._buffers[name] = mm._buffers[name].unsqueeze(0)
                    mm._buffers[name] = mm._buffers[name].expand([batch_size, ] + list(mm._buffers[name].shape[1:]))
                    mm._buffers[name] = mm._buffers[name].contiguous()
            

    def update_state_1(self, selected_beam, time_step):
        # 第一时间步后扩展状态维度，后续时间步只变换行
        # selected_beam:(batch_size, beam_size)
        beam_size = selected_beam.shape[-1] 
        batch_size  = selected_beam.shape[0]
        cur_beam_size = beam_size
        if time_step==0:
            cur_beam_size = 1
        for mm in self.buffer_module:
            for name in mm._buffers:
                buf = mm._buffers[name]
                shape = [int(sh) for sh in buf.shape]
                beam = selected_beam
                for _ in shape[1:]:
                    beam = beam.unsqueeze(-1)
                buf = torch.gather(buf.view(*([batch_size, cur_beam_size] + shape[1:])), 1,
                                beam.expand(*([batch_size, beam_size] + shape[1:])))
                buf = buf.view(*([-1, ] + shape[1:]))
                mm._buffers[name] = buf


    def update_visual_1(self, feats, mask, beam_size):
        feats = torch.repeat_interleave(feats, beam_size, dim=0)
        mask = torch.repeat_interleave(mask, beam_size, dim=0)
        return feats, mask


    def recover_state_1(self):
        for mm in self.buffer_module:
            for name in mm._buffers:
                mm._buffers[name] = mm.default_state[name]
        # 清空，下个batch继续使用，default_state字典会自动覆盖
        self.buffer_module.clear()


    def beam_search_1(self, images, pixels, device):
        beam_size = 5
        batch_size =  images.shape[0]
        # 整个句子概率，要用均值(后面值再大也大不过1)
        seq_logprob = images.new_full((batch_size, 1, 1), 0.)
        # 句子中每个词语的概率
        stepword_logprob = list()
        # 句子是否遇到'<eos>'结束，0表示结束
        seq_mask = images.new_full((batch_size, beam_size, 1), 1.)
        # 每个时间步的单词
        step_word = list()
        # 当前time_step句子，第0时间步
        cur_seq = images.new_full((batch_size, 1), self.bos_idx).long()
        # encoder输出
        images = self.embed_image(images)
        pixels = self.embed_pixel(pixels)
        enc_feats, enc_mask = self.encoder(images, pixels)
        self.init_state_1(batch_size)
        for time_step in range(20):
            # 只取最后一个时间步(50, 1, 10198)->(50, 10198); cur: (50, 1)
            word_logprob = self.decoder(cur_seq, enc_feats, enc_mask).squeeze(1)
            # 中间-1维度，0_step是1， 后续是beam_size=5
            word_logprob = word_logprob.view(batch_size, -1, 10201)
            # 加上前面时间步prob
            candidate_logprob = seq_logprob + word_logprob

            # 到达EOS，生成mask
            if time_step != 0:
                cur_mask = (cur_seq.view(batch_size, beam_size) != 3).float().unsqueeze(-1)
                seq_mask = seq_mask * cur_mask
                word_logprob = word_logprob * seq_mask.expand_as(word_logprob)
                old_seq_logprob = seq_logprob.expand_as(candidate_logprob).contiguous()
                old_seq_logprob[:, :, 1:] = -999
                candidate_logprob = seq_mask * candidate_logprob + old_seq_logprob * (1 - seq_mask)
            
            selected_logprob, selected_idx = torch.sort(candidate_logprob.view(batch_size, -1), -1, descending=True)
            selected_logprob, selected_idx = selected_logprob[:, :beam_size], selected_idx[:, :beam_size]
            selected_beam = torch.div(selected_idx, candidate_logprob.shape[-1], rounding_mode='trunc')
            selected_words = selected_idx - selected_beam * candidate_logprob.shape[-1]

            seq_logprob = selected_logprob.unsqueeze(-1)
            seq_mask = torch.gather(seq_mask, 1, selected_beam.unsqueeze(-1))

            step_word = list(torch.gather(word, 1, selected_beam.unsqueeze(-1)) for word in step_word)
            step_word.append(selected_words.unsqueeze(-1))

            this_word_logprob = torch.gather(word_logprob, 1,
                                            selected_beam.unsqueeze(-1).expand(batch_size, beam_size,
                                                                                word_logprob.shape[-1]))
            this_word_logprob = torch.gather(this_word_logprob, 2, selected_words.unsqueeze(-1))
            stepword_logprob = list(
                torch.gather(word_logprob, 1, selected_beam.unsqueeze(-1).expand(batch_size, beam_size, 1)) for word_logprob in stepword_logprob)
            stepword_logprob.append(this_word_logprob)
            cur_seq = selected_words.view(-1, 1)
            if time_step==0:
                enc_feats, enc_mask = self.update_visual_1(enc_feats, enc_mask, beam_size)
            self.update_state_1(selected_beam, time_step)

        self.recover_state_1()
        seq_logprob, sort_idxs = torch.sort(seq_logprob, 1, descending=True)
        step_word = torch.cat(step_word, -1) 
        step_word = torch.gather(step_word, 1, sort_idxs.expand(batch_size, beam_size, 20))
        stepword_logprob = torch.cat(stepword_logprob, -1)
        stepword_logprob = torch.gather(stepword_logprob, 1, sort_idxs.expand(batch_size, beam_size, 20))
        oneseq = step_word.contiguous()[:, 1] #(10, 54)
        allseq = step_word.view(-1, 20).contiguous() #(50, 54)
        logprob = stepword_logprob.contiguous()[:, :beam_size]  # (10, 5, 54)
        return oneseq, allseq, logprob


    @property
    def d_model(self):
        return self.decoder.d_model

    def init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, images, seq, pixel, *args):
        images = self.embed_image(images)
        pixel = self.embed_pixel(pixel)
        enc_output, mask_enc = self.encoder(images, pixel)
        dec_output = self.decoder(seq, enc_output, mask_enc)
        return dec_output

    def init_state(self, b_s, device):
        return [torch.zeros((b_s, 0), dtype=torch.long, device=device),
                None, None]

    def step(self, t, prev_output, visual, pixel, seq, mode='teacher_forcing', **kwargs):
        it = None
        if mode == 'teacher_forcing':
            raise NotImplementedError
        elif mode == 'feedback':
            if t == 0:
                visual = self.embed_image(visual)
                pixel = self.embed_pixel(pixel)
                self.enc_output, self.mask_enc = self.encoder(visual, pixel)
                if isinstance(visual, torch.Tensor):
                    it = visual.data.new_full((visual.shape[0], 1), self.bos_idx).long()
                else:
                    it = visual[0].data.new_full((visual[0].shape[0], 1), self.bos_idx).long()
            else:
                it = prev_output

        return self.decoder(it, self.enc_output, self.mask_enc)
    
    def relprop_decode(self, R, **kwargs):
        """ propagates relevances from rdo to output embeddings and encoder state """
        # if self.normalize_out:
            # R = self.dec_out_norm.relprop(R)

        # R_enc = 0.0
        # R_enc_scale = 0.0
        R_crop = 0.0
        # for layer in range(self.decodernum_layers_dec)[::-1]:
            # R = self.dec_ffn[layer].relprop(R)

            # relevance_dict = self.dec_enc_attn[layer].relprop(R, main_key='query_inp')
            # R = relevance_dict['query_inp']
            # R_enc += relevance_dict['kv_inp']
            # R_enc_scale += tf.reduce_sum(abs(relevance_dict['kv_inp']))

            # R = self.dec_attn[layer].relprop(R)
        R, R_enc = self.decoder.relprop(R, **kwargs)
        R_crop = R
        # shift left: compensate for right shift
        # R_crop = tf.pad(R, [[0, 0], [0, 1], [0, 0]])[:, 1:, :]

        return {'emb_out': R_crop, 'enc_out': R_enc,
                'emb_out_before_crop': R}

    def relprop_encode(self, R, **kwargs):
        """ propagates relevances from enc_out to emb_inp """
        # if self.normalize_out:
            # R = self.enc_out_norm.relprop(R)
        # for layer in range(self.num_layers_enc)[::-1]:
            # R = self.enc_ffn[layer].relprop(R)
            # R = self.enc_attn[layer].relprop(R)
        R = self.encoder.relprop(R, **kwargs)
        return R

    def relprop_encode_decode(self, R, **kwargs):
        """ propagates relevances from rdo to input and optput embeddings """
        relevances = self.relprop_decode(R, **kwargs)
        # relevances['emb_inp'] = self.relprop_encode(relevances['enc_out'])
        relevances['emb_inp'] = relevances['enc_out']
        return relevances


class TransformerEnsemble(CaptioningModel1):
    def __init__(self, model: Difnet_LRP, weight_files):
        super(TransformerEnsemble, self).__init__()
        self.n = len(weight_files)
        self.models = ModuleList([copy.deepcopy(model) for _ in range(self.n)])
        for i in range(self.n):
            state_dict_i = torch.load(weight_files[i])['state_dict']
            self.models[i].load_state_dict(state_dict_i)

    def step(self, t, prev_output, visual, seq, mode='teacher_forcing', **kwargs):
        out_ensemble = []
        for i in range(self.n):
            out_i = self.models[i].step(t, prev_output, visual, seq, mode, **kwargs)
            out_ensemble.append(out_i.unsqueeze(0))

        return torch.mean(torch.cat(out_ensemble, 0), dim=0)

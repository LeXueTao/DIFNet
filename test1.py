import os
import random
# from data import ImageDetectionsField1, TextField, RawField, PixelField, ImageDetectionsField152
# from data import COCO, DataLoader
from data import ImageDetectionsField1, TextField, RawField, PixelField, ImageDetectionsField152
from data import COCO, DataLoader
import evaluation
from models.transformer import TransformerEncoder, TransformerDecoder, ScaledDotProductAttention, Transformer
from models.transformer_lrp import TransformerEncoder_LRP, TransformerDecoder_LRP, ScaledDotProductAttention_LRP, Transformer_LRP
from models.difnet import Difnet, DifnetEncoder, DifnetDecoder
from models.difnet_lrp import Difnet_LRP, DifnetEncoder_LRP
import torch
from tqdm import tqdm
import argparse
import pickle
import numpy as np
# import cv2
import json
# from torch.utils.data import DataLoader

random.seed(1234)
torch.manual_seed(1234)
np.random.seed(1234)



def beam_search(model, seq_len, beam_size, bos_idx, eos_idx, pad_idx, image):
    # batch,后面几步会在第0步重新赋值
    batch_size =  image.shape[0]
    # 整个句子概率，要用均值(后面值再大也大不过1)
    seq_logprob = image.new_full((batch_size, beam_size, 1), 0.)
    # 句子中每个词语的概率
    seqword_logprob = image.new_full((batch_size, beam_size, 1), 0.)
    # 句子是否遇到'<eos>'结束，0表示结束
    seq_mask = image.new_full((batch_size, beam_size, 1), 1.)
    # 句子记录集合，提前填充('<pad>')，开始只有'<cls>'符号
    seq = image.new_full((batch_size, beam_size, 1), bos_idx).long()
    # 当前time_step句子，这是第0时间步
    cur_seq = seq.reshape(batch_size*beam_size, -1)
    # encoder输出
    enc_feats, enc_mask = model.encoder(model.embed_image(image))
    enc_feats = torch.repeat_interleave(enc_feats, beam_size, dim=0)
    enc_mask = torch.repeat_interleave(enc_mask, beam_size, dim=0)
    model.init_state(batch_size, beam_size)
    for time_step in range(seq_len):
        # 输出只取最后一个时间步 word_logprob -> (50, 30522)  (b_s*5, 30522)
        # 输入最后一个时间步
        word_logprob = model.decoder(cur_seq, enc_feats, enc_mask).squeeze(1)
        # time_step0时刻，需要特殊处理，把横向数据竖过来
        if time_step == 0:
            selected_logprob, selected_idx = torch.sort(word_logprob[0::5], -1, descending=True)
            selected_logprob, selected_idx = selected_logprob[:, :beam_size], selected_idx[:, :beam_size]
            seq_logprob = selected_logprob.unsqueeze(-1)
            seqword_logprob = seq_logprob
            seq = selected_idx.unsqueeze(-1)
            seq = torch.cat((seq, selected_idx.unsqueeze(-1)), dim=-1)
            cur_seq = seq.reshape(batch_size*beam_size, -1)[:, -1].unsqueeze(-1)
            continue
        word_logprob = word_logprob.reshape(batch_size, beam_size, -1)
        # 加上整个seq的概率
        candidate_logprob = seq_logprob + word_logprob
        # (batch_s, beam_s, 1), 0表示结束
        mask = (cur_seq[:, -1].reshape(batch_size, beam_size) != eos_idx).float().unsqueeze(-1)
        # 更新seq_mask，这一步让为eos的持续为0
        seq_mask = seq_mask * mask
        # 已经被mask的beam上的预测概率被赋值0
        word_logprob = word_logprob * seq_mask.expand_as(word_logprob) 
        old_seq_logprob = seq_logprob.expand_as(candidate_logprob).contiguous()
        # 整体复制的，只需要保留第一个值
        old_seq_logprob[:, :, 1:] = -999
        # 未被遮盖的继续更新，被遮盖的保留原值(-999的那部分会参与到整体筛选概率中)
        candidate_logprob = seq_mask * candidate_logprob + old_seq_logprob * (1 - seq_mask)
        # 选词，选beam，开始更新所有数据
        selected_logprob, selected_idx = torch.sort(candidate_logprob.view(batch_size, -1), -1, descending=True)
        selected_logprob, selected_idx = selected_logprob[:, :beam_size], selected_idx[:, :beam_size]
        selected_beam = torch.div(selected_idx, candidate_logprob.shape[-1], rounding_mode='trunc')
        selected_words = selected_idx - selected_beam * candidate_logprob.shape[-1]
        # 更新seq_logprob
        seq_logprob = selected_logprob.unsqueeze(-1)
        # 加入单词logprob
        this_word_logprob = torch.gather(word_logprob, 1,
                                         selected_beam.unsqueeze(-1).expand(batch_size, beam_size,
                                                                            word_logprob.shape[-1]))
        this_word_logprob = torch.gather(this_word_logprob, 2, selected_words.unsqueeze(-1))
        seqword_logprob = torch.gather(seqword_logprob, 1, 
                                       selected_beam.unsqueeze(-1).expand(batch_size, beam_size, 
                                                                          seqword_logprob.shape[-1]))
        seqword_logprob = torch.cat((seqword_logprob, this_word_logprob), dim=-1)
        # 更新seq_mask
        seq_mask = torch.gather(seq_mask, 1, selected_beam.unsqueeze(-1))
        # 更新seq(先换beam，再添加词)
        seq = torch.gather(seq, 1, selected_beam.unsqueeze(-1).expand_as(seq))
        # 根据seq_mask将遮掩的词语赋值'<pad>'
        pad_words = torch.ones_like(selected_words) * pad_idx
        seq_mask_bool = seq_mask.to(torch.bool).squeeze(-1)
        selected_words = torch.where(seq_mask_bool, selected_words, pad_words)
        seq = torch.cat((seq, selected_words.unsqueeze(-1)), dim=-1)
        # 更新下一个时间步使用单词
        cur_seq = seq.reshape(batch_size*beam_size, -1)[:, -1].unsqueeze(-1)
        # 更新model中的buffer
        model.update_state(selected_beam)
        
    # 排序 去掉seq中的初始<pad>符号
    model.recover_state()
    seq_logprob, sort_idxs = torch.sort(seq_logprob, 1, descending=True)
    seq = torch.gather(seq[:,:,1:], 1, sort_idxs.expand(batch_size, beam_size, seq_len))
    seqword_logprob = torch.gather(seqword_logprob, 1, sort_idxs.expand(batch_size, beam_size, seq_len))
    oneseq = seq[:,0,:].contiguous()
    allseq = seq.reshape(-1, seq_len).contiguous()
    seqword_logprob = seqword_logprob.contiguous()
    return oneseq, allseq, seqword_logprob



def predict_captions(model, dataloader, text_field, out_file):
    import itertools
    model.eval()
    gen = {}
    gts = {}
    outs = {}
    with tqdm(desc='Evaluation', unit='it', total=len(dataloader)) as pbar:
        for it, ((img_id, images, pixels), caps_gt) in enumerate(dataloader):
            # print(img_id.data.numpy()[0])
            images = images.to(device)
            # images = torch.zeros((50, 49, 2048)).to(device)
            pixels = pixels.to(device)
            # pixels = torch.zeros((50, 49, 133)).to(device)
            # depths1 = depths1.to(device)
            with torch.no_grad():
                out, _ = model.beam_search(images, pixels, 20, text_field.vocab.stoi['<eos>'], 5, out_size=1)
            caps_gen = text_field.decode(out, join_words=False)
            for i, (gts_i, gen_i) in enumerate(zip(caps_gt, caps_gen)):
                gen_i = ' '.join([k for k, g in itertools.groupby(gen_i)]) # 消除重复元素
                gen['%d_%d' % (it, i)] = [gen_i.strip(), ]
                gts['%d_%d' % (it, i)] = gts_i

                # print('gen:',gen['%d_%d' % (it, i)])
                # print('gts:',gts['%d_%d' % (it, i)])
                # out_file.write('gen:{}'.format(gen['%d_%d' % (it, i)]))
                # out_file.write('gts:{}'.format(gts['%d_%d' % (it, i)]))
                # out_file.write('img_id: {}'.format(str(img_id.data.numpy()[0])))
                # out_file.write('\n')
                # out_file.write('gen:{}'.format(gen['%d_%d' % (it, i)]))
                # out_file.write('\n')
                # out_file.write('gts:{}'.format(gts['%d_%d' % (it, i)]))
                # out_file.write('\n')
                outs[str(img_id.data.numpy()[0])] = {'gen': gen['%d_%d' % (it, i)], 'gts': gts['%d_%d' % (it, i)]}
                # out_file.write(gen['%d_%d' % (it, i)][0])
                # out_file.write('\n')
            pbar.update()
    #         if it > 5:
    #             break
    # print(outs)
    # json.dump(outs, out_file)
    gts = evaluation.PTBTokenizer.tokenize(gts)
    gen = evaluation.PTBTokenizer.tokenize(gen)
    scores, score_list = evaluation.compute_scores(gts, gen)
    # cc = {}
    # out1_file = open(os.path.join(args.out_path, args.exp_name + '_cider.json'), 'w')
    # cc['cider'] = list(score_list['CIDEr'])
    # json.dump(cc, out1_file)
    # out1_file.close()

    return scores


if __name__ == '__main__':
    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

    parser = argparse.ArgumentParser(description='DIFNet')
    parser.add_argument('--exp_name', type=str, default='DIFNet_lrp')
    # parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=48)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--model_path', type=str, default='./')
    parser.add_argument('--out_path', type=str, default='./output/output_lrp')
    parser.add_argument('--features_path', type=str, default='./datasets/coco2014_gridfeats')
    parser.add_argument('--annotation_folder', type=str, default='./datasets/coco2014/annotations')
    parser.add_argument('--pixel_path', type=str, default='./datasets/coco2014_seg')

    parser.add_argument('--embed_size', type=int, default=512,
                        help='dimension of word embedding vectors')
    parser.add_argument('--d_model', type=int, default=512,
                        help='dimension of lstm hidden states')
    parser.add_argument('--mode', type=str, default='difnet_lrp', choices=['base', 'base_lrp', 'difnet', 'difnet_lrp'])
    args = parser.parse_args()

    print('{} Evaluation'.format(args.mode))

    # Pipeline for image regions
    image_field = ImageDetectionsField1(detections_path=args.features_path, max_detections=49, load_in_tmp=False)
    # Pipeline for depth
    pixel_field = PixelField(pixel_path=args.pixel_path, load_in_tmp=False)
    # Pipeline for text
    text_field = TextField(init_token='<bos>', eos_token='<eos>', lower=True, tokenize='spacy',
                           remove_punctuation=True, nopoints=False)

    text_field.vocab = pickle.load(open('vocab.pkl', 'rb'))

    # Create the dataset
    coco_dataset = COCO(image_field, text_field, pixel_field, './datasets/coco2014', args.annotation_folder, args.annotation_folder)
    _, _, dataset_test = coco_dataset.splits



    # Model and dataloaders
    if args.mode == 'base':
        encoder = TransformerEncoder(3, 0, attention_module=ScaledDotProductAttention)
        decoder = TransformerDecoder(len(text_field.vocab), 54, 3, text_field.vocab.stoi['<pad>'])
        model = Transformer(text_field.vocab.stoi['<bos>'], encoder, decoder).to(device)
    if args.mode == 'base_lrp':
        encoder = TransformerEncoder_LRP(3, 0, attention_module=ScaledDotProductAttention_LRP)
        decoder = TransformerDecoder_LRP(len(text_field.vocab), 54, 3, text_field.vocab.stoi['<pad>'])
        model = Transformer_LRP(text_field.vocab.stoi['<bos>'], encoder, decoder).to(device)
    if args.mode == 'difnet':
        encoder = DifnetEncoder(1, 2, 3, 0, attention_module=ScaledDotProductAttention)
        decoder = DifnetDecoder(len(text_field.vocab), 54, 3, text_field.vocab.stoi['<pad>'])
        model = Difnet(text_field.vocab.stoi['<bos>'], encoder, decoder).to(device)
    if args.mode == 'difnet_lrp':
        encoder = DifnetEncoder_LRP(3, 0, attention_module=ScaledDotProductAttention_LRP)
        decoder = TransformerDecoder_LRP(len(text_field.vocab), 54, 3, text_field.vocab.stoi['<pad>'])
        model = Difnet_LRP(text_field.vocab.stoi['<bos>'], encoder, decoder).to(device)

    data = torch.load(os.path.join(args.model_path, args.exp_name + '.pth'), map_location=device)
    # model.load_state_dict(data['state_dict'])


    dict_dataset_test = dataset_test.image_dictionary({'image': image_field, 'text': RawField(), 'pixel': pixel_field})
    dict_dataloader_test = DataLoader(dict_dataset_test, batch_size=args.batch_size, num_workers=args.workers)


    # out_file = open(os.path.join(args.out_path, args.exp_name + '.json'), 'w')
    # out_file = open(os.path.join(args.out_path, args.exp_name + '.txt'), 'w')
    out_file = None
    scores = predict_captions(model, dict_dataloader_test, text_field, out_file)
    # out_file.close()
    print(scores)

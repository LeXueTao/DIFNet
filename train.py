import random
from test_data import ImageDetectionsField, TextField, RawField, PixelField
from test_data import COCO, DataLoader
# from data import ImageDetectionsField, TextField, RawField, PixelField
# from data import COCO, DataLoader
import evaluation
from evaluation import PTBTokenizer, Cider
from models.transformer import TransformerEncoder, TransformerDecoder, ScaledDotProductAttention, Transformer
from models.transformer_lrp import TransformerEncoder_LRP, TransformerDecoder_LRP, ScaledDotProductAttention_LRP, Transformer_LRP
from models.difnet import Difnet, DifnetEncoder, DifnetDecoder
from models.difnet_lrp import Difnet_LRP, DifnetEncoder_LRP

import torch
from torch.optim import Adam
# from torch.optim.lr_scheduler import LambdaLR
from torch.nn import NLLLoss
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import argparse
import os
import pickle
import numpy as np
import itertools
import multiprocessing
from shutil import copyfile

import time

random.seed(1234)
torch.manual_seed(1234)
np.random.seed(1234)

test = False

def evaluate_loss(model, dataloader, loss_fn, text_field):

    # Validation loss
    model.eval()
    running_loss = .0
    with tqdm(desc='Epoch %d - validation' % e, unit='it', total=len(dataloader)) as pbar:
        with torch.no_grad():
            for it, (detections, captions, pixels) in enumerate(dataloader):
                detections, captions, pixels = detections.to(device), captions.to(device), pixels.to(device)
                out = model(detections, captions, pixels)
                captions = captions[:, 1:].contiguous()
                out = out[:, :-1].contiguous()
                loss = loss_fn(out.view(-1, len(text_field.vocab)), captions.view(-1))
                this_loss = loss.item()
                running_loss += this_loss

                pbar.set_postfix(avg_loss=running_loss / (it + 1))
                pbar.update()
                if test:
                    break

    val_loss = running_loss / len(dataloader)
    return val_loss


def evaluate_metrics(model, dataloader, text_field):
    import itertools
    model.eval()
    gen = {}
    gts = {}
    with tqdm(desc='Epoch %d - evaluation' % e, unit='it', total=len(dataloader)) as pbar:
        for it, ((images, pixels), caps_gt) in enumerate(iter(dataloader)):
            images = images.to(device)
            pixels = pixels.to(device)
            with torch.no_grad():
                out, _ = model.beam_search(images, pixels, 20, text_field.vocab.stoi['<eos>'], 5, out_size=1)
            caps_gen = text_field.decode(out, join_words=False)
            for i, (gts_i, gen_i) in enumerate(zip(caps_gt, caps_gen)):
                gen_i = ' '.join([k for k, g in itertools.groupby(gen_i)]) 
                gen['%d_%d' % (it, i)] = [gen_i, ]
                gts['%d_%d' % (it, i)] = gts_i
            pbar.update()
            if test:
                break

    gts = evaluation.PTBTokenizer.tokenize(gts)
    gen = evaluation.PTBTokenizer.tokenize(gen)
    scores, _ = evaluation.compute_scores(gts, gen)
    return scores


def train_xe(model, dataloader, optim, text_field):
    # Training with cross-entropy
    model.train()
    running_loss = .0
    with tqdm(desc='Epoch %d - train' % e, unit='it', total=len(dataloader)) as pbar:
        for it, (detections, captions, pixels) in enumerate(dataloader):
            detections, captions, pixels = detections.to(device), captions.to(device), pixels.to(device)
            time1 = time.time()
            out = model(detections, captions, pixels)
            time2 = time.time()
            # print('tt:{}'.format(time2 - time1))
            start_time = time.time()
            optim.zero_grad()
            captions_gt = captions[:, 1:].contiguous()
            out = out[:, :-1].contiguous()
            loss = loss_fn(out.view(-1, len(text_field.vocab)), captions_gt.view(-1))
            loss.backward()

            optim.step()
            this_loss = loss.item()
            running_loss += this_loss

            pbar.set_postfix(avg_loss=running_loss / (it + 1))
            pbar.update()

            end_time = time.time()
            # print("loss_time:{}".format(end_time-start_time))

            if test:
                break

    loss = running_loss / len(dataloader)
    return loss


def train_scst(model, dataloader, optim, cider, text_field):
    # Training with self-critical
    # 设置进程池，Pool()默认cpu个数
    tokenizer_pool = multiprocessing.Pool()
    running_reward = .0
    running_reward_baseline = .0
    model.train()
    running_loss = .0
    seq_len = 20
    beam_size = 5

    with tqdm(desc='Epoch %d - train' % e, unit='it', total=len(dataloader)) as pbar:
        for it, ((detections, pixels), caps_gt) in enumerate(dataloader):
            detections = detections.to(device)
            pixels = pixels.to(device)
            
            outs, log_probs = model.beam_search(detections, pixels, seq_len, text_field.vocab.stoi['<eos>'],
                                                beam_size, out_size=beam_size)
            optim_rl.zero_grad()
            
            # Rewards
            reward_time1 = time.time()
            caps_gen = text_field.decode(outs.view(-1, seq_len)) #[str1, str2,]
            caps_gt = list(itertools.chain(*([c, ] * beam_size for c in caps_gt))) # [[str1, str2,,,],,]
            caps_gen, caps_gt = tokenizer_pool.map(evaluation.PTBTokenizer.tokenize, [caps_gen, caps_gt])
            beam_time2 = time.time()
            
            reward = cider.compute_score(caps_gt, caps_gen)[1].astype(np.float32)
            
            reward = torch.from_numpy(reward).to(device).view(detections.shape[0], beam_size)
            reward_baseline = torch.mean(reward, -1, keepdim=True)
            loss = -torch.mean(log_probs, -1) * (reward - reward_baseline)
            

            loss = loss.mean()
            loss.backward()
            optim_rl.step()

            running_loss += loss.item()
            running_reward += reward.mean().item()
            running_reward_baseline += reward_baseline.mean().item()
            pbar.set_postfix(loss=running_loss / (it + 1), reward=running_reward / (it + 1),
                             reward_baseline=running_reward_baseline / (it + 1))
            pbar.update()
            if test:
                break

    loss = running_loss / len(dataloader)
    reward = running_reward / len(dataloader)
    reward_baseline = running_reward_baseline / len(dataloader)
    return loss, reward, reward_baseline


if __name__ == '__main__':
    device = torch.device('cuda:2')
    use_rl = True
    parser = argparse.ArgumentParser(description='DIFNet')
    parser.add_argument('--exp_name', type=str, default='DIFNet')

    parser.add_argument('--batch_size', type=int, default=50)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--m', type=int, default=40)
    parser.add_argument('--head', type=int, default=8)
    parser.add_argument('--resume_last', action='store_true')
    parser.add_argument('--resume_best', action='store_true')
    parser.add_argument('--features_path', type=str, default='./datasets/coco2014_gridfeats')
    parser.add_argument('--pixel_path', type=str, default='./datasets/coco2014_seg')
    parser.add_argument('--annotation_folder', type=str, default='./datasets/coco2014/annotations')
    parser.add_argument('--logs_folder', type=str, default='./output/tensorboard_logs')
    parser.add_argument('--model_path', type=str, default='./output/saved_transformer_models')

    parser.add_argument('--mode', type=str, default='base', choices=['base', 'base_lrp', 'difnet', 'difnet_lrp'])
    args = parser.parse_args()
    print(args)

    print('Transformer Training')
    if args.exp_name == 'test':
        test = True
    writer = SummaryWriter(log_dir=os.path.join(args.logs_folder, args.exp_name))

    # Pipeline for image regions
    image_field = ImageDetectionsField(detections_path=args.features_path, max_detections=49, load_in_tmp=False)
    # Pipeline for pixel
    pixel_field = PixelField(pixel_path=args.pixel_path, load_in_tmp=False)
    # Pipeline for text
    text_field = TextField(init_token='<bos>', eos_token='<eos>', lower=True, tokenize='spacy', remove_punctuation=True, nopoints=False)

    # Create the dataset
    dataset = COCO(image_field, text_field, pixel_field, './dataset/coco2014', args.annotation_folder, args.annotation_folder)
    train_dataset, val_dataset, test_dataset = dataset.splits

    if not os.path.isfile('vocab.pkl'):
        print("Building vocabulary")
        text_field.build_vocab(train_dataset, val_dataset, min_freq=5)
        pickle.dump(text_field.vocab, open('vocab.pkl', 'wb'))
    else:
        print('Loading from vocabulary')
        text_field.vocab = pickle.load(open('vocab.pkl', 'rb'))



    dict_dataset_train = train_dataset.image_dictionary({'image': image_field, 'text': RawField(), 'pixel': pixel_field})
    dict_dataset_val = val_dataset.image_dictionary({'image': image_field, 'text': RawField(), 'pixel': pixel_field})
    dict_dataset_test = test_dataset.image_dictionary({'image': image_field, 'text': RawField(), 'pixel': pixel_field})


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



    def lambda_lr(s):
        base_lr = 0.0001
        if s <= 3:
            lr = base_lr * (s+1) /4
        elif s <= 10:
            lr = base_lr
        elif s <= 12:
            lr = base_lr * 0.2
        else:
            lr = base_lr * 0.2 * 0.2
        
        return lr

    def lambda_lr_rl(s):
        base_lr = 5e-6
        if s <= 29:
            lr = base_lr
        elif s <= 31:
            lr = base_lr * 0.2
        else:
            lr = base_lr * 0.2 * 0.2
        return lr

    # 交叉熵训练
    optim = Adam(model.parameters(), lr=1, betas=(0.9, 0.98))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lambda_lr, last_epoch=-1)

    # scst训练
    optim_rl = Adam(model.parameters(), lr=1, betas=(0.9, 0.98))
    scheduler_rl = torch.optim.lr_scheduler.LambdaLR(optim_rl, lambda_lr_rl, last_epoch=-1)

    loss_fn = NLLLoss(ignore_index=text_field.vocab.stoi['<pad>'])
    best_cider = .0
    patience = 0
    start_epoch = 0

    if args.resume_last or args.resume_best:
        if args.resume_last:
            fname = os.path.join(args.model_path, '%s_last.pth' % args.exp_name)
        else:
            fname = os.path.join(args.model_path, '%s_best.pth' % args.exp_name)

        if os.path.exists(fname):
            data = torch.load(fname)
            torch.set_rng_state(data['torch_rng_state'])
            torch.cuda.set_rng_state(data['cuda_rng_state'])
            np.random.set_state(data['numpy_rng_state'])
            random.setstate(data['random_rng_state'])
            model.load_state_dict(data['state_dict'], strict=False)
            optim.load_state_dict(data['optimizer'])
            scheduler.load_state_dict(data['scheduler'])
            scheduler_rl.load_state_dict(data['scheduler_rl'])
            start_epoch = data['epoch'] + 1
            best_cider = data['best_cider']
            patience = data['patience']
            use_rl = data['use_rl']
            print('Resuming from epoch %d, validation loss %f, and best cider %f' % (
                data['epoch'], data['val_loss'], data['best_cider']))
            print('patience:', data['patience'])
            print('num_workers:', args.workers)

    ref_caps_train = list(train_dataset.text)
    cider_train = Cider(PTBTokenizer.tokenize(ref_caps_train))


    print("Training starts")
    for e in range(start_epoch, start_epoch + 100):
        dataloader_train = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,drop_last=True, pin_memory=True)
        dataloader_val = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
        # 这里是针对每张照片，每张照片在coco中有5个caption，所以batch_size默认除以5
        dict_dataloader_train = DataLoader(dict_dataset_train, batch_size=args.batch_size // 5, shuffle=True,num_workers=args.workers)
        dict_dataloader_val = DataLoader(dict_dataset_val, batch_size=args.batch_size // 5, shuffle=True,num_workers=args.workers)
        dict_dataloader_test = DataLoader(dict_dataset_test, batch_size=args.batch_size // 5, shuffle=True,num_workers=args.workers)


        if not use_rl:
            print("epoch:{},\tlr:{}".format(e, scheduler.get_last_lr()))
            train_loss = train_xe(model, dataloader_train, optim, text_field)
            scheduler.step()
            writer.add_scalar('data/train_loss', train_loss, e)
        else:
            print("epoch:{},\tlr:{}".format(e, scheduler_rl.get_last_lr()))
            train_loss, reward, reward_baseline = train_scst(model, dict_dataloader_train, optim, cider_train, text_field)
            scheduler_rl.step()
            writer.add_scalar('data/train_loss', train_loss, e)
            writer.add_scalar('data/reward', reward, e)
            writer.add_scalar('data/reward_baseline', reward_baseline, e)

       

        # Validation loss
        import time
        val_loss_start = time.time()
        val_loss = evaluate_loss(model, dataloader_val, loss_fn, text_field)
        val_loss_end = time.time()
        writer.add_scalar('data/val_loss', val_loss, e)

        # Validation scores
        val_scor_start = time.time()
        scores = evaluate_metrics(model, dict_dataloader_val, text_field)
        val_scor_end = time.time()
        # print("val_loss_time:{}, val_score_time:{}".format(val_loss_end-val_loss_start, val_scor_end-val_scor_start))
        print("Validation scores", scores)
        val_cider = scores['CIDEr']
        writer.add_scalar('data/val_cider', val_cider, e)
        writer.add_scalar('data/val_bleu1', scores['BLEU'][0], e)
        writer.add_scalar('data/val_bleu4', scores['BLEU'][3], e)
        writer.add_scalar('data/val_meteor', scores['METEOR'], e)
        writer.add_scalar('data/val_rouge', scores['ROUGE'], e)


        # Prepare for next epoch
        best = False
        if val_cider >= best_cider:
            best_cider = val_cider
            patience = 0
            best = True
        else:
            patience += 1

        switch_to_rl = False
        exit_train = False
        if patience == 5:
            if not use_rl:
                use_rl = True
                switch_to_rl = True
                patience = 0
                optim = Adam(model.parameters(), lr=5e-6)
                print("Switching to RL")
            else:
                print('patience reached.')
                exit_train = True

        # 大于20epoch必定转向scst
        if not use_rl:
            if e >= 20:
                use_rl = True
                switch_to_rl = True
                patience = 0
                optim = Adam(model.parameters(), lr=5e-6)
                print("Switching to RL")
        # 刚到强化学习阶段，找最好的那个加载
        if switch_to_rl and not best:
            data = torch.load(os.path.join(args.model_path, '%s_best.pth' % args.exp_name))
            torch.set_rng_state(data['torch_rng_state'])
            torch.cuda.set_rng_state(data['cuda_rng_state'])
            np.random.set_state(data['numpy_rng_state'])
            random.setstate(data['random_rng_state'])
            model.load_state_dict(data['state_dict'])
            print('Resuming from epoch %d, validation loss %f, and best cider %f' % (
                data['epoch'], data['val_loss'], data['best_cider']))

        # 每一波都保存
        torch.save({
            'torch_rng_state': torch.get_rng_state(),
            'cuda_rng_state': torch.cuda.get_rng_state(),
            'numpy_rng_state': np.random.get_state(),
            'random_rng_state': random.getstate(),
            'epoch': e,
            'val_loss': val_loss,
            'val_cider': val_cider,
            'state_dict': model.state_dict(),
            'optimizer': optim.state_dict(),
            'scheduler': scheduler.state_dict(),
            'scheduler_rl': scheduler_rl.state_dict(),
            'patience': patience,
            'best_cider': best_cider,
            'use_rl': use_rl,
        }, os.path.join(args.model_path, '%s_last.pth' % args.exp_name))

        # 保存最好的结果
        if best:
            copyfile(os.path.join(args.model_path, '%s_last.pth' % args.exp_name), os.path.join(args.model_path, '%s_best.pth' % args.exp_name))
        if exit_train:
            writer.close()
            break

import random
from data import ImageDetectionsField, TextField, RawField
from data import COCO_KD, COCO, DataLoader
import evaluation
from evaluation import PTBTokenizer, Cider
from DLNA_KDXE import DLNA_KDXE
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR, ReduceLROnPlateau
from torch.nn import NLLLoss
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import argparse, os, pickle
import numpy as np
import itertools
import multiprocessing
from shutil import copyfile
import time
from data import build_image_field
from utils.utils import load_vocab, decode_sequence, clip_chongfu
import warnings
warnings.filterwarnings("ignore")

def evaluate_loss(model, dataloader, loss_fn, text_field):
    model.eval()
    running_loss = .0
    with tqdm(desc='Epoch %d - validation' % e, unit='it', total=len(dataloader)) as pbar:
        with torch.no_grad():
            for it, (region, region_mask, grid, grid_mask, captions) in enumerate(dataloader):
                region, region_mask, grid, grid_mask, captions = region.to(device), region_mask.to(device), grid.to(device), grid_mask.to(device), captions.to(device)
                logit_region, logit_grid = model(region, region_mask, grid, grid_mask)
                loss_region = loss_fn(logit_region.view(-1, logit_region.shape[-1]), captions.view(-1))
                loss_grid = loss_fn(logit_grid.view(-1, logit_grid.shape[-1]), captions.view(-1))
                loss = (loss_region + loss_grid) * 0.5
                this_loss = loss.item()
                running_loss += this_loss

                pbar.set_postfix(loss=running_loss / (it + 1))
                pbar.update()

    val_loss = running_loss / len(dataloader)
    return val_loss

def evaluate_metrics(model, dataloader, text_field):
    import itertools
    model.eval()
    gen = {}
    gts = {} 
    with tqdm(desc='Epoch %d - evaluation' % e, unit='it', total=len(dataloader)) as pbar:
        for it, ((region, region_mask, grid, grid_mask), caps_gt) in enumerate(iter(dataloader)):
            region = region.to(device)
            region_mask = region_mask.to(device)
            grid = grid.to(device)
            grid_mask = grid_mask.to(device)
            with torch.no_grad():
                logit_region, logit_grid = model(region, region_mask, grid, grid_mask)
                out = (logit_region + logit_grid) * 0.5
                _, seq = torch.max(out, -1)
            
            caps_gen = text_field.decode(seq, join_words=False)
            for i, (gts_i, gen_i) in enumerate(zip(caps_gt, caps_gen)):
                gen_i = ' '.join([k for k, g in itertools.groupby(gen_i)])
                gen['%d_%d' % (it, i)] = [gen_i, ]
                gts['%d_%d' % (it, i)] = gts_i
            pbar.update()

    gts = evaluation.PTBTokenizer.tokenize(gts)
    gen = evaluation.PTBTokenizer.tokenize(gen)
    scores, _ = evaluation.compute_scores(gts, gen)
    return scores

def train_xe(model, dataloader, optim, text_field):
    model.train()
    scheduler.step()
    running_loss = .0
    running_loss_region = .0
    running_loss_grid = .0
    with tqdm(desc='Epoch %d - train' % e, unit='it', total=len(dataloader)) as pbar:
        for it, (region, region_mask, grid, grid_mask, captions) in enumerate(dataloader):
            region, region_mask, grid, grid_mask, captions = region.to(device), region_mask.to(device), grid.to(device), grid_mask.to(device), captions.to(device)
            logit_region, logit_grid = model(region, region_mask, grid, grid_mask)
            optim.zero_grad()
            loss_region = loss_fn(logit_region.view(-1, logit_region.shape[-1]), captions.view(-1))
            loss_grid = loss_fn(logit_grid.view(-1, logit_grid.shape[-1]), captions.view(-1))
            loss = (loss_region + loss_grid)*0.5
            
            loss = loss.mean()
            loss.backward()

            optim.step()
            running_loss += loss.item()

            pbar.set_postfix(loss=running_loss / (it + 1))
            pbar.update()

    loss = running_loss / len(dataloader)
    return loss

if __name__ == '__main__':
    device = torch.device('cuda')
    parser = argparse.ArgumentParser(description='DBNAIC')
    parser.add_argument('--exp_name', type=str, default='DBNAIC')
    parser.add_argument('--exp_tag', type=str, default='0520')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--resume_best', action='store_true')
    parser.add_argument('--resume_last', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--resume_path', type=str, default='./saved_models/DBNAIC.pth')
    parser.add_argument('--features_path', type=str, default='./mscoco/')
    parser.add_argument('--annotation_folder', type=str, default='./annotations')
    parser.add_argument('--logs_folder', type=str, default='tensorboard_logs')
    parser.add_argument('--max_len', type=int, default=20)
    parser.add_argument('--d_model', type=int, default=512)
    parser.add_argument('--n_layer', type=int, default=3)
    parser.add_argument('--rl_at', type=int, default=200)
    parser.add_argument('--max_detections', type=int, default=150)
    parser.add_argument('--dim_feats', type=int, default=2048)
    parser.add_argument('--image_field', type=str, default="ImageSwinRegionGridWithMask")
    parser.add_argument('--grid_embed', action='store_true', default=True)
    parser.add_argument('--box_embed', action='store_true', default=True)
    parser.add_argument('--seed', type=int, default=1234)

    args = parser.parse_args()
    print(args)

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    writer = SummaryWriter(log_dir=os.path.join(args.logs_folder, args.exp_name+"_"+args.exp_tag))

    image_field = build_image_field(args.image_field, args.features_path, args.max_detections)
    text_field = TextField(eos_token='<eos>', fix_length=args.max_len, lower=True, tokenize='spacy', remove_punctuation=True, nopoints=False)
    dataset = COCO_KD(image_field, text_field, 'coco/images/', args.annotation_folder)
    train_dataset, val_dataset, test_dataset = dataset.splits

    if not os.path.isfile('vocab_dlna.pkl'):
        print("Building vocabulary")
        text_field.build_vocab(train_dataset, val_dataset, min_freq=1)
        pickle.dump(text_field.vocab, open('vocab_dlna.pkl', 'wb'))
    else:
        text_field.vocab = pickle.load(open('vocab_dlna.pkl', 'rb'))
    print('vocab length: '+str(len(text_field.vocab)))

    dict_dataset_train = train_dataset.image_dictionary({'image': image_field, 'text': RawField()})
    ref_caps_train = list(train_dataset.text)
    cider_train = None
    dict_dataset_val = val_dataset.image_dictionary({'image': image_field, 'text': RawField()})
    dict_dataset_test = test_dataset.image_dictionary({'image': image_field, 'text': RawField()})
    
    def lambda_lr(s):
        base_lr = 0.0001
        if s <= 3:
            lr = base_lr * s / 4
        elif s <= 49:
            lr = base_lr 
        elif s <= 69:
            lr = base_lr * 0.2
        else:
            lr = base_lr * 0.2 * 0.2
        return lr 
    
    best_cider = .0
    patience = 0
    start_epoch = 0
    exit_train = False

    model = DLNA_KDXE(args.n_layer, d_model=args.d_model, max_len=args.max_len, vocab_size=len(text_field.vocab)).to(device)
    
    optim = Adam(model.parameters(), lr=1, betas=(0.9, 0.98))
    scheduler = LambdaLR(optim, lambda_lr)
    loss_fn = NLLLoss(ignore_index=text_field.vocab.stoi['<pad>'])
            
    if args.resume_last or args.resume_best:
        if args.resume_last:
            fname = 'saved_models/%s_last.pth' % (args.exp_name +"_"+ args.exp_tag)
        else:
            fname = 'saved_models/%s_best.pth' % (args.exp_name +"_"+ args.exp_tag)

        if os.path.exists(fname):
            data = torch.load(fname)
            torch.set_rng_state(data['torch_rng_state'])
            torch.cuda.set_rng_state(data['cuda_rng_state'])
            np.random.set_state(data['numpy_rng_state'])
            random.setstate(data['random_rng_state'])
            model.load_state_dict(data['state_dict'], strict=False)
            optim.load_state_dict(data['optimizer'])
            start_epoch = data['epoch'] + 1
            best_cider = data['best_cider']
            patience = data['patience']
            scheduler.load_state_dict(data['scheduler'])
            print('Resuming from epoch %d, validation loss %f, and best cider %f' % (
                data['epoch'], data['val_loss'], data['best_cider']))         

    for e in range(start_epoch, start_epoch + 200):
        dataloader_train = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, drop_last=True)
        dataloader_val = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
        dict_dataloader_val = DataLoader(dict_dataset_val, batch_size=args.batch_size)
        dict_dataloader_test = DataLoader(dict_dataset_test, batch_size=args.batch_size)
        
        train_loss = train_xe(model, dataloader_train, optim, text_field)
        writer.add_scalar('data/train_loss', train_loss, e)

        val_loss = evaluate_loss(model, dataloader_val, loss_fn, text_field)
        writer.add_scalar('data/val_loss', val_loss, e)

        # Validation scores
        scores = evaluate_metrics(model, dict_dataloader_val, text_field)
        print("Validation scores", scores)
        val_cider = scores['CIDEr']
        writer.add_scalar('data/val_cider', val_cider, e)
        writer.add_scalar('data/val_bleu1', scores['BLEU'][0], e)
        writer.add_scalar('data/val_bleu4', scores['BLEU'][3], e)
        writer.add_scalar('data/val_meteor', scores['METEOR'], e)
        writer.add_scalar('data/val_rouge', scores['ROUGE'], e)

        current_lr = optim.state_dict()['param_groups'][0]['lr']
        writer.add_scalar('data/learning_rate', current_lr, e)

        # Test scores
        scores = evaluate_metrics(model, dict_dataloader_test, text_field)
        print("Test scores", scores)
        test_cider = scores['CIDEr']
        writer.add_scalar('data/test_cider', scores['CIDEr'], e)
        writer.add_scalar('data/test_bleu1', scores['BLEU'][0], e)
        writer.add_scalar('data/test_bleu4', scores['BLEU'][3], e)
        writer.add_scalar('data/test_meteor', scores['METEOR'], e)
        writer.add_scalar('data/test_rouge', scores['ROUGE'], e)

        # Prepare for next epoch
        best = False
        if test_cider >= best_cider:
            best_cider = test_cider
            patience = 0
            best = True
        else:
            patience += 1

        if patience >= 10:
            exit_train = True

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
            'patience': patience,
            'best_cider': best_cider
        }, 'saved_models/%s_last.pth' % (args.exp_name +"_"+ args.exp_tag))

        if best:
            copyfile('saved_models/%s_last.pth' % (args.exp_name +"_"+ args.exp_tag), 'saved_models/%s_best.pth' % (args.exp_name +"_"+ args.exp_tag))

        if exit_train:
            writer.close()
            break
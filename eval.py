import random
from data import ImageDetectionsField, TextField, RawField
from data import COCO, COCO_KD, DataLoader
import evaluation
from data import build_image_field
from DLNA_KDXE import DLNA_KDXE
import torch
from tqdm import tqdm
import argparse
import pickle
import numpy as np
import multiprocessing
from evaluation import PTBTokenizer, Cider
import json

random.seed(1234)
torch.manual_seed(1234)
np.random.seed(1234)


def predict_captions(model, dataloader, text_field, cider, args):
    import itertools
    tokenizer_pool = multiprocessing.Pool()
    res = {}
    model.eval()
    gen = {}
    gts = {}
    with tqdm(desc='Evaluation', unit='it', total=len(dataloader)) as pbar:
        for it, ((region, region_mask, grid, grid_mask), caps_gt) in enumerate(iter(dataloader)):
            region = region.to(device)
            region_mask = region_mask.to(device)
            grid = grid.to(device)
            grid_mask = grid_mask.to(device)
            with torch.no_grad():
                logit_region, logit_grid = model(region, region_mask, grid, grid_mask)
                out = (logit_region + logit_grid) *0.5
                _, seq = torch.max(out, -1)
            caps_gen = text_field.decode(seq, join_words=False)
            caps_gen1 = text_field.decode(seq)
            caps_gt1 = list(itertools.chain(*([c, ] * 1 for c in caps_gt)))

            caps_gen1, caps_gt1 = tokenizer_pool.map(evaluation.PTBTokenizer.tokenize, [caps_gen1, caps_gt1])
            reward = cider.compute_score(caps_gt1, caps_gen1)[1].astype(np.float32)

            for i,(gts_i, gen_i) in enumerate(zip(caps_gt1,caps_gen1)):
                res[len(res)] = {
                    'gt':caps_gt1[gts_i],
                    'gen':caps_gen1[gen_i],
                    'cider':reward[i].item(),
                }

            for i, (gts_i, gen_i) in enumerate(zip(caps_gt, caps_gen)):
                gen_i = ' '.join([k for k, g in itertools.groupby(gen_i)])
                gen['%d_%d' % (it, i)] = [gen_i.strip(), ]
                gts['%d_%d' % (it, i)] = gts_i
            pbar.update()

    gts = evaluation.PTBTokenizer.tokenize(gts)
    gen = evaluation.PTBTokenizer.tokenize(gen)
    scores, _ = evaluation.compute_scores(gts, gen)
    if len(args.dump_json)>0:
        json.dump(res,open(args.dump_json,'w'))
    return scores



if __name__ == '__main__':
    device = torch.device('cuda')

    parser = argparse.ArgumentParser(description='DBNAIC')
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--features_path', type=str, default='./mscoco/')
    parser.add_argument('--annotation_folder', type=str, default='./annotations')
    parser.add_argument('--resume_path', type=str, default='./saved_models/DBNAIC.pth')
    parser.add_argument('--image_field', type=str, default="ImageSwinRegionGridWithMask")
    parser.add_argument('--max_detections', type=int, default=150)
    parser.add_argument('--grid_embed', action='store_true', default=True)
    parser.add_argument('--box_embed', action='store_true', default=True)
    parser.add_argument('--max_len', type=int, default=20)
    parser.add_argument('--n_layer', type=int, default=3)
    parser.add_argument('--dim_feats', type=int, default=2048)
    parser.add_argument('--d_model', type=int, default=512)
    parser.add_argument('--spice', action='store_true', default=False)
    parser.add_argument('--dump_json', type=str, default='./DBNAIC_TEST.json')

    args = parser.parse_args()
    image_field = build_image_field(args.image_field, args.features_path, args.max_detections)
    text_field = TextField(eos_token='<eos>', fix_length=args.max_len, lower=True, tokenize='spacy', remove_punctuation=True, nopoints=False)
    dataset = COCO_KD(image_field, text_field, 'coco/images/', args.annotation_folder)
    _, _, test_dataset = dataset.splits
    text_field.vocab = pickle.load(open('vocab_dlna.pkl', 'rb'))

    ref_caps_test = list(test_dataset.text)
    cider_test = Cider(PTBTokenizer.tokenize(ref_caps_test))

    # Model and dataloaders
    model = DLNA_KDXE(args.n_layer, d_model=args.d_model, max_len=args.max_len, vocab_size=len(text_field.vocab)).to(device)
    data = torch.load(args.resume_path)
    model.load_state_dict(data['state_dict'])

    dict_dataset_test = test_dataset.image_dictionary({'image': image_field, 'text': RawField()})
    dict_dataloader_test = DataLoader(dict_dataset_test, batch_size=args.batch_size, num_workers=args.workers)
    scores = predict_captions(model, dict_dataloader_test, text_field, cider_test, args)
    print(scores)
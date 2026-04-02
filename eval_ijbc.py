# coding: utf-8

import argparse
import os
import pickle
import timeit
import warnings
from pathlib import Path

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import prettytable
import skimage.transform
import torch
from menpo.visualize.viewmatplotlib import sample_colours_from_colourmap
from sklearn.metrics import auc, roc_curve
from sklearn.preprocessing import normalize
from torch.utils.data import DataLoader

from backbones import get_model

warnings.filterwarnings("ignore")

SRC = np.array(
    [
        [30.2946, 51.6963],
        [65.5318, 51.5014],
        [48.0252, 71.7366],
        [33.5493, 92.3655],
        [62.7299, 92.2041]]
    , dtype=np.float32)
SRC[:, 0] += 8.0


@torch.no_grad()
class AlignedDataSet(object):
    def __init__(self, root, lines, align=True):
        self.lines = lines
        self.root = root
        self.align = align

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        each_line = self.lines[idx]
        name_lmk_score = each_line.strip().split(' ')
        name = os.path.join(self.root, name_lmk_score[0])
        img = cv2.cvtColor(cv2.imread(name), cv2.COLOR_BGR2RGB)
        landmark5 = np.array([float(x) for x in name_lmk_score[1:-1]], dtype=np.float32).reshape((5, 2))
        st = skimage.transform.SimilarityTransform()
        st.estimate(landmark5, SRC)
        img = cv2.warpAffine(img, st.params[0:2, :], (112, 112), borderValue=0.0)
        img_1 = np.expand_dims(img, 0)
        img_2 = np.expand_dims(np.fliplr(img), 0)
        output = np.concatenate((img_1, img_2), axis=0).astype(np.float32)
        output = np.transpose(output, (0, 3, 1, 2))
        return torch.from_numpy(output)


@torch.no_grad()
def extract(model, dataset, batch_size):
    model.eval()
    def collate_fn(data):
        return torch.cat(data, dim=0)

    data_loader = DataLoader(
        dataset, batch_size=batch_size, drop_last=False, num_workers=4, collate_fn=collate_fn)
    
    feats_list = []
    idx = 0
    for batch in data_loader:
        # Preprocessing as expected by arcface PyTorch models: (x / 255.0 - 0.5) / 0.5
        batch = batch.cuda()
        batch.div_(255).sub_(0.5).div_(0.5)
        
        feat = model(batch)
        feat = feat.reshape(-1, feat.shape[1] * 2) # [bs, 2*feat_dim] (feat and flipped feat)
        feats_list.append(feat.cpu().numpy())
        idx += batch.shape[0] // 2 
        if idx % 1000 == 0 or idx == len(dataset):
            print(f'Extracted: {idx} / {len(dataset)}')
            
    feat_mat = np.concatenate(feats_list, axis=0)
    return feat_mat


def read_template_media_list(path):
    ijb_meta = pd.read_csv(path, sep=' ', header=None).values
    templates = ijb_meta[:, 1].astype(int)
    medias = ijb_meta[:, 2].astype(int)
    return templates, medias


def read_template_pair_list(path):
    pairs = pd.read_csv(path, sep=' ', header=None).values
    t1 = pairs[:, 0].astype(int)
    t2 = pairs[:, 1].astype(int)
    label = pairs[:, 2].astype(int)
    return t1, t2, label


def image2template_feature(img_feats=None, templates=None, medias=None):
    unique_templates = np.unique(templates)
    template_feats = np.zeros((len(unique_templates), img_feats.shape[1]))
    
    for count_template, uqt in enumerate(unique_templates):
        (ind_t,) = np.where(templates == uqt)
        face_norm_feats = img_feats[ind_t]
        face_medias = medias[ind_t]
        unique_medias, unique_media_counts = np.unique(face_medias, return_counts=True)
        media_norm_feats = []
        for u, ct in zip(unique_medias, unique_media_counts):
            (ind_m,) = np.where(face_medias == u)
            if ct == 1:
                media_norm_feats += [face_norm_feats[ind_m]]
            else:
                media_norm_feats += [np.mean(face_norm_feats[ind_m], axis=0, keepdims=True), ]
        media_norm_feats = np.array(media_norm_feats)
        template_feats[count_template] = np.sum(media_norm_feats, axis=0)
        if count_template % 2000 == 0:
            print(f'Finish Calculating {count_template} template features.')
            
    template_norm_feats = normalize(template_feats)
    return template_norm_feats, unique_templates


def verification(template_norm_feats=None, unique_templates=None, p1=None, p2=None):
    template2id = np.zeros((max(unique_templates) + 1, 1), dtype=int)
    for count_template, uqt in enumerate(unique_templates):
        template2id[uqt] = count_template
    
    score = np.zeros((len(p1),))
    total_pairs = np.array(range(len(p1)))
    batchsize = 100000
    sublists = [total_pairs[i: i + batchsize] for i in range(0, len(p1), batchsize)]
    total_sublists = len(sublists)
    
    for c, s in enumerate(sublists):
        feat1 = template_norm_feats[template2id[p1[s]]]
        feat2 = template_norm_feats[template2id[p2[s]]]
        similarity_score = np.sum(feat1 * feat2, -1)
        score[s] = similarity_score.flatten()
        if c % 10 == 0:
            print(f'Finish {c}/{total_sublists} pairs.')
    return score


def main(args):
    use_norm_score = True
    use_detector_score = True
    use_flip_test = True
    assert args.target in ['IJBC', 'IJBB']

    start = timeit.default_timer()
    templates, medias = read_template_media_list(
        os.path.join(args.image_path, 'meta', f'{args.target.lower()}_face_tid_mid.txt'))
    print(f'Time: {timeit.default_timer() - start:.2f} s. ')

    start = timeit.default_timer()
    p1, p2, label = read_template_pair_list(
        os.path.join(args.image_path, 'meta', f'{args.target.lower()}_template_pair_label.txt'))
    print(f'Time: {timeit.default_timer() - start:.2f} s. ')

    start = timeit.default_timer()
    img_path = os.path.join(args.image_path, 'loose_crop')
    img_list_path = os.path.join(args.image_path, 'meta', f'{args.target.lower()}_name_5pts_score.txt')
    with open(img_list_path, 'r') as f:
        files = f.readlines()

    result_dir = args.result_dir
    save_path = os.path.join(result_dir, args.job)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    model_name = os.path.splitext(os.path.basename(args.model_prefix))[0]
    features_path = os.path.join(save_path, f'img_feats_{model_name}.npy')

    if os.path.exists(features_path):
        print(f'Loading features from {features_path}')
        img_feats = np.load(features_path)
    else:
        print('Cache not found. Loading PyTorch model and extracting features...')
        resnet = get_model(args.network, dropout=0, fp16=False).cuda()
        weight = torch.load(args.model_prefix)
        resnet.load_state_dict(weight)
        model = torch.nn.DataParallel(resnet)
        
        dataset = AlignedDataSet(root=img_path, lines=files, align=True)
        img_feats = extract(model, dataset, args.batch_size)
        np.save(features_path, img_feats)

    faceness_scores = np.array([float(line.split()[-1]) for line in files], dtype=np.float32)
    print(f'Time: {timeit.default_timer() - start:.2f} s. ')
    print(f'Feature Shape: ({img_feats.shape[0]} , {img_feats.shape[1]}) .')
    
    start = timeit.default_timer()
    if use_flip_test:
        img_input_feats = img_feats[:, :img_feats.shape[1] // 2] + img_feats[:, img_feats.shape[1] // 2:]
    else:
        img_input_feats = img_feats[:, :img_feats.shape[1] // 2]

    if not use_norm_score:
        img_input_feats = img_input_feats / np.sqrt(np.sum(img_input_feats ** 2, -1, keepdims=True))

    if use_detector_score:
        img_input_feats = img_input_feats * faceness_scores[:, np.newaxis]

    template_norm_feats, unique_templates = image2template_feature(img_input_feats, templates, medias)
    print(f'Time: {timeit.default_timer() - start:.2f} s. ')

    start = timeit.default_timer()
    score = verification(template_norm_feats, unique_templates, p1, p2)
    print(f'Time: {timeit.default_timer() - start:.2f} s. ')
    
    score_save_file = os.path.join(save_path, f"{args.target.lower()}_{model_name}.npy")
    np.save(score_save_file, score)
    
    # Get ROC curves and table
    methods = [os.path.basename(score_save_file)]
    scores = {methods[0]: score}
    colours = dict(zip(methods, sample_colours_from_colourmap(len(methods), 'Set2')))
    x_labels = [10 ** -6, 10 ** -5, 10 ** -4, 10 ** -3, 10 ** -2, 10 ** -1]
    tpr_fpr_table = prettytable.PrettyTable(['Methods'] + [str(x) for x in x_labels])
    
    fig = plt.figure()
    for method in methods:
        fpr, tpr, _ = roc_curve(label, scores[method])
        roc_auc = auc(fpr, tpr)
        fpr = np.flipud(fpr)
        tpr = np.flipud(tpr)
        
        plt.plot(fpr, tpr, color=colours[method], lw=1, 
                 label=f'[{method.split("-")[-1]} (AUC = {roc_auc * 100:0.4f} %)]')
        
        tpr_fpr_row = [f"{method}-{args.target}"]
        for fpr_iter in np.arange(len(x_labels)):
            _, min_index = min(list(zip(abs(fpr - x_labels[fpr_iter]), range(len(fpr)))))
            tpr_fpr_row.append(f'{tpr[min_index] * 100:.2f}')
        tpr_fpr_table.add_row(tpr_fpr_row)
        
    plt.xlim([10 ** -6, 0.1])
    plt.ylim([0.3, 1.0])
    plt.grid(linestyle='--', linewidth=1)
    plt.xticks(x_labels)
    plt.yticks(np.linspace(0.3, 1.0, 8, endpoint=True))
    plt.xscale('log')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC on IJB')
    plt.legend(loc="lower right")
    fig.savefig(os.path.join(save_path, f'{args.target.lower()}_{model_name}.pdf'))
    print(tpr_fpr_table)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='do ijb test')
    parser.add_argument('--model-prefix', default='', help='path to load model.')
    parser.add_argument('--image-path', default='ijb/IJBC', type=str, help='')
    parser.add_argument('--result-dir', default='.', type=str, help='')
    parser.add_argument('--batch-size', default=128, type=int, help='')
    parser.add_argument('--network', default='r50', type=str, help='')
    parser.add_argument('--job', default='insightface', type=str, help='job name')
    parser.add_argument('--target', default='IJBC', type=str, help='target, set to IJBC or IJBB')
    args = parser.parse_args()
    main(args)

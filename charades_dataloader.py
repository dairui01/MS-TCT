import torch
import torch.utils.data as data_utl
from torch.utils.data.dataloader import default_collate
import numpy as np
import json
import os
import os.path
from tqdm import tqdm
import random
from utils import *

def make_dataset(split_file, split, root, num_classes=157):
    gamma = 0.5
    tau = 4
    ku = 1
    dataset = []
    with open(split_file, 'r') as f:
        data = json.load(f)
    print('split!!!!', split)
    i = 0
    for vid in tqdm(data.keys()):
        if data[vid]['subset'] != split:
            continue

        if not os.path.exists(os.path.join(root, vid + '.npy')):
            continue

        if len(data[vid]['actions']) < 1:
            continue

        fts = np.load(os.path.join(root, vid + '.npy'))
        num_feat = fts.shape[0]
        label = np.zeros((num_feat, num_classes), np.float32)
        #
        hmap = np.zeros((num_feat, num_classes), np.float32)
        action_lengths = []
        center_loc = []
        num_action = 0

        fps = num_feat / data[vid]['duration']
        for ann in data[vid]['actions']:
            # 
            if ann[2] < ann[1]:
                continue
            mid_point = (ann[2] + ann[1]) / 2
            for fr in range(0, num_feat, 1):
                if fr / fps > ann[1] and fr / fps < ann[2]:
                    label[fr, ann[0]] = 1  # binary classification

                # G* Ground truth Heat-map
                # if fr / fps + 1 > mid_point and fr / fps < mid_point:
                if (fr+1) / fps > mid_point and fr / fps < mid_point:
                    center = fr + 1
                    class_ = ann[0]
                    action_duration = int((ann[2] - ann[1]) * fps)
                    radius = int(action_duration / gamma)
                    generate_gaussian(hmap[:, class_], center, radius, tau, ku)
                    num_action = num_action + 1
                    center_loc.append([center, class_])
                    action_lengths.append([action_duration])

        dataset.append((vid, label, data[vid]['duration'], [hmap, num_action, np.asarray(center_loc), np.asarray(action_lengths)]))
        i += 1

    return dataset


class Charades(data_utl.Dataset):

    def __init__(self, split_file, split, root, batch_size, classes, num_clips, skip):
        
        self.data = make_dataset(split_file, split, root, classes)
        self.split=split
        self.split_file = split_file
        self.batch_size = batch_size
        self.root = root
        self.in_mem = {}
        self.num_clips = num_clips
        self.skip = skip

    def __getitem__(self, index):
        entry = self.data[index]
        feat = np.load(os.path.join(self.root, entry[0] + '.npy'))
        feat = feat.reshape((feat.shape[0], 1, 1, feat.shape[-1]))
        features = feat.astype(np.float32)

        labels = entry[1]

        hmap, num_action, center_loc, action_lengths = entry[3]
        # print('center_loc',center_loc.shape)
        # center_loc = np.transpose(center_loc, axes=[1, 0])
        num_clips = self.num_clips

        if self.split in ["training", "testing"]:
            if len(features) > num_clips and num_clips > 0:
                if self.split == "testing":
                    random_index = 0
                else:
                    random_index = random.choice(range(0, len(features) - num_clips))
                features = features[random_index: random_index + num_clips: 1]
                labels = labels[random_index: random_index + num_clips: 1]
                hmap = hmap[random_index: random_index + num_clips: 1]
        # center_loc = np.transpose(center_loc, axes=[1, 0])

        return features, labels, hmap, action_lengths, [entry[0], entry[2], num_action]

    def __len__(self):
        return len(self.data)


class collate_fn_unisize():

    def __init__(self,num_clips):
        self.num_clips = num_clips

    def charades_collate_fn_unisize(self, batch):
        max_len = int(self.num_clips)
        max_len1= int(self.num_clips)
        new_batch = []
        for b in batch:
            f = np.zeros((max_len, b[0].shape[1], b[0].shape[2], b[0].shape[3]), np.float32)
            m = np.zeros((max_len), np.float32)
            l = np.zeros((max_len, b[1].shape[1]), np.float32)
            h = np.zeros((max_len, b[2].shape[1]), np.float32)
            f[:b[0].shape[0]] = b[0]
            m[:b[0].shape[0]] = 1
            l[:b[0].shape[0], :] = b[1]
            h[:b[0].shape[0], :] = b[2]

            new_batch.append([video_to_tensor(f), torch.from_numpy(m), torch.from_numpy(l), b[4], torch.from_numpy(h)])

        return default_collate(new_batch)


import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torchvision.datasets.vision import VisionDataset
import sys

import datasets.transforms as transforms

class UBFC(VisionDataset):
    def __init__(self, split, arg_obj):
        super(UBFC, self).__init__(split, arg_obj)

        self.round_robin_index = int(arg_obj.K)
        self.debug           = bool(int(arg_obj.debug))
        self.split           = split.lower()
        self.channels        = arg_obj.channels.lower()
        self.frames_per_clip = int(arg_obj.fpc)
        self.step            = int(arg_obj.step)
        self.aug             = arg_obj.augmentation.lower()
        self.speed_slow      = float(arg_obj.speed_slow)
        self.speed_fast      = float(arg_obj.speed_fast)
        self.fps             = int(arg_obj.fps)

        self.set_augmentations()
        self.load_data()
        self.pad_inputs()
        self.build_samples()

        print(self.split)
        print('Samples: ', self.samples.shape)
        print('Total frames: ', self.samples.shape[0] * self.frames_per_clip)


    def load_data(self):
        if self.fps == 30:
            meta = pd.read_csv('datasets/metadata/UBFC_meta.csv')
        elif self.fps == 90:
            meta = pd.read_csv('datasets/metadata/UBFC_meta_90fps.csv')
        else:
            print('Invalid fps for UBFC loader. Must be in [30,90]. Exiting.')
            sys.exit(-1)

        ids = meta['id']

        # Determine which subject modulus to use
        use_mods = set()
        if self.split == 'train':
            use_mods.add(self.round_robin_index % 5)
            use_mods.add((self.round_robin_index + 1) % 5)
            use_mods.add((self.round_robin_index + 2) % 5)
        elif self.split == 'val':
            use_mods.add((self.round_robin_index + 3) % 5)
        elif self.split == 'test':
            use_mods.add((self.round_robin_index + 4) % 5)
        elif self.split == 'all':
            use_mods = set([0,1,2,3,4])
        else:
            print('Invalid split specified to UBFC dataloader:', self.split, 'Exiting.')
            sys.exit(-1)

        data = []
        self.waves = []
        remove_subjects = set([11, 18, 20, 24]) ## Same samples removed by Gideon et al. 2021
        for idx, row in meta.iterrows():
            subj_id = row['id']
            if (subj_id % 5 in use_mods) and (subj_id not in remove_subjects):
                npz = np.load(row['path'])
                d = {k: npz[k] for k in npz.files}
                d['id'] = subj_id
                d['path'] = row['path']
                self.waves.append(d['wave'])
                data.append(d)
        self.data = data


    def pad_inputs(self):
        ''' Add a step-width pad to both ends so the whole video is processed.
        '''
        if (self.split == 'test') or (self.split == 'all'):
            self.masks = []
            for i in range(len(self.waves)):
                self.data[i]['video'] = self.data[i]['video'][:len(self.waves[i])] #cut to wave if wave is shorter
                pad = self.step - (len(self.waves[i]) % self.step) + 1
                mask = np.ones_like(self.waves[i], dtype=bool)
                if pad > 0:
                    self.waves[i] = np.hstack((self.waves[i], np.repeat(self.waves[i][-1], pad)))
                    back = self.data[i]['video'][[-1]].repeat(pad, 0)
                    self.data[i]['video'] = np.append(self.data[i]['video'], back, axis=0)
                    mask = np.hstack((mask, np.zeros(pad, dtype=bool)))
                self.masks.append(mask)


    def set_augmentations(self):
        raise NotImplementedError


    def build_samples(self):
        start_idcs = self.get_start_idcs()
        ## Want array of size clips with (subj, start_idx) in each element
        samples = []
        for subj in range(len(self.waves)):
            starts = start_idcs[subj]
            subj_rep = np.repeat(subj, len(starts))
            sample = np.vstack((subj_rep, starts))
            samples.append(sample)
        self.samples = np.hstack(samples).T


    def get_start_idcs(self):
        start_idcs = []
        for wave in self.waves:
            slen = len(wave)
            end = slen - self.frames_per_clip
            starts = np.arange(0, end, self.step)
            start_idcs.append(starts)
        start_idcs = np.array(start_idcs, dtype=object)
        return start_idcs


    def get_subj_sizes(self):
        subjects = np.unique(self.samples[:, 0])
        ends = []
        for subj in subjects:
            end = self.samples[self.samples[:, 0]==subj, 1][-1]
            ends.append(end)
        frames_per_subj = np.array(ends) + self.frames_per_clip
        return frames_per_subj


    def apply_transformations(self, clip, subj, idcs, augment=True):
        speed = 1.0
        if augment:
            ## Time resampling
            if self.aug_speed:
                entire_clip = self.data[subj]['video']
                clip, idcs, speed = transforms.augment_speed(entire_clip, idcs, self.frames_per_clip, self.channels, self.speed_slow, self.speed_fast)

            ## Randomly horizontal flip
            if self.aug_flip:
                clip = transforms.augment_horizontal_flip(clip)

            ## Randomly reverse time
            if self.aug_reverse:
                clip = transforms.augment_time_reversal(clip)

            ## Illumination noise
            if self.aug_illum:
                clip = transforms.augment_illumination_noise(clip)

            ## Gaussian noise for every pixel
            if self.aug_gauss:
                clip = transforms.augment_gaussian_noise(clip)

            ## Random resized cropping
            if self.aug_resizedcrop:
                clip = transforms.random_resized_crop(clip)

        clip = np.clip(clip, 0, 255)
        clip = clip / 255
        clip = torch.from_numpy(clip).float()

        return clip, idcs, speed


    def __len__(self):
        return self.samples.shape[0]


    def __getitem__(self, idx):
        raise NotImplementedError

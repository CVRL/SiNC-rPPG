import numpy as np
from datasets.UBFC import UBFC
import datasets.transforms as transforms


class UBFCUnsupervised(UBFC):
    def __init__(self, split, arg_obj):
        super().__init__(split, arg_obj)


    def set_augmentations(self):
        self.aug_flip = False
        self.aug_illum = False
        self.aug_gauss = False
        self.aug_speed = False
        self.aug_resizedcrop = False
        self.aug_reverse = False
        if self.split == 'train':
            self.aug_flip = True if 'f' in self.aug else False
            self.aug_illum = True if 'i' in self.aug else False
            self.aug_gauss = True if 'g' in self.aug else False
            self.aug_speed = True if 's' in self.aug else False
            self.aug_resizedcrop = True if 'c' in self.aug else False
            self.aug_reverse = True if 'r' in self.aug else False


    def __getitem__(self, idx):
        subj, start_idx = self.samples[idx]
        idcs = np.arange(start_idx, start_idx + self.frames_per_clip, dtype=int)
        clip = self.data[subj]['video'][idcs] # [T,H,W,C]
        clip = transforms.prepare_clip(clip, self.channels) # [C,T,H,W]

        if self.split == 'train':
            clip, idcs, speed = self.apply_transformations(clip, subj, idcs)
            return clip, subj, idcs, speed
        else:
            clip, idcs, speed = self.apply_transformations(clip, subj, idcs, augment=False)
            return clip, subj, idcs

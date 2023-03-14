import numpy as np
import torch
import torch.nn.functional as F


def resample_clip(video, length):
    video = np.transpose(video, (3,0,1,2)).astype(float)
    video = interpolate_clip(video, length)
    video = np.transpose(video, (1,2,3,0))
    return video


def arrange_channels(imgs, channels):
    d = {'b':0, 'g':1, 'r':2, 'n':3}
    channel_order = [d[c] for c in channels]
    imgs = imgs[:,:,:,channel_order]
    return imgs


def prepare_clip(clip, channels):
    clip = arrange_channels(clip, channels)
    clip = np.transpose(clip, (3, 0, 1, 2)) # [C,T,H,W]
    clip = clip.astype(np.float64)
    return clip


def augment_speed(clip, idcs, frames_per_clip, channels, speed_slow, speed_fast):
    ''' Interpolates clip to frames_per_clip length given slicing indices, which
        can be floats.
    '''
    vid_len = len(clip)
    within_bounds = False
    while not within_bounds:
        speed_c = np.random.uniform(speed_slow, speed_fast)
        min_idx = idcs[0].astype(int)
        max_idx = np.round(frames_per_clip * speed_c + min_idx).astype(int)
        if max_idx < vid_len:
            within_bounds = True
    speed_c = (max_idx - min_idx) / frames_per_clip # accomodate rounding of end-indices
    clip = clip[min_idx:max_idx]
    clip = prepare_clip(clip, channels)
    interped_clip = interpolate_clip(clip, frames_per_clip)
    interped_idcs = np.linspace(min_idx, max_idx-1, frames_per_clip)
    return interped_clip, interped_idcs, speed_c


def interpolate_clip(clip, length):
    '''
    Input:
        clip: numpy array of shape [C,T,H,W]
        length: number of time points in output interpolated sequence
    Returns:
        Tensor of shape [C,T,H,W]
    '''
    clip = torch.from_numpy(clip[np.newaxis])
    clip = F.interpolate(clip, (length, 64, 64), mode='trilinear', align_corners=True)
    return clip[0].numpy()


def resize_clip(clip, length):
    '''
    Input:
        clip: numpy array of shape [C,T,H,W]
        length: number of time points in output interpolated sequence
    Returns:
        Tensor of shape [C,T,H,W]
    '''
    T = clip.shape[1]
    clip = torch.from_numpy(np.ascontiguousarray(clip[np.newaxis]))
    clip = F.interpolate(clip, (T, length, length), mode='trilinear', align_corners=False)
    return clip[0].numpy()


def random_resized_crop(clip, crop_scale_lims=[0.5, 1]):
    ''' Randomly crop a subregion of the video and resize it back to original size.
    Arguments:
        clip (np.array): expects [C,T,H,W]
    Returns:
        clip (np.array): same dimensions as input
    '''
    C,T,H,W = clip.shape
    crop_scale = np.random.uniform(crop_scale_lims[0], crop_scale_lims[1])
    crop_length = np.round(crop_scale * H).astype(int)
    crop_start_lim = H - (crop_length)
    x1 = np.random.randint(0, crop_start_lim+1)
    y1 = x1
    x2 = x1 + crop_length
    y2 = y1 + crop_length
    cropped_clip = clip[:,:,y1:y2,x1:x2]
    resized_clip = resize_clip(cropped_clip, H)
    return resized_clip


def augment_gaussian_noise(clip):
    clip = clip + np.random.normal(0, 2, clip.shape)
    return clip


def augment_illumination_noise(clip):
    clip = clip + np.random.normal(0, 10)
    return clip


def augment_time_reversal(clip):
    if np.random.rand() > 0.5:
        clip = np.flip(clip, 1)
    return clip


def augment_horizontal_flip(clip):
    if np.random.rand() > 0.5:
        clip = np.flip(clip, 3)
    return clip

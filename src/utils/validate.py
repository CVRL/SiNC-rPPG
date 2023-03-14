import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from tqdm import tqdm
from copy import deepcopy
import os

from utils.postprocess import overlap_add, predict_HR


def partition_by_subject(pred_waves, subjs):
    pred_arrs = []
    unique_subj = np.unique(subjs)
    for i, subj in enumerate(unique_subj):
        subj_idcs = np.where(subjs == subj)[0]
        subj_preds = pred_waves[subj_idcs]
        pred_arrs.append(subj_preds)
    pred_arrs = np.array(pred_arrs, dtype=object)
    return pred_arrs


def overlap_add_all_subjects(pred_arrs, fpc, step, normed=True, hanning=True):
    oadd_arrs = []
    for i, subj_pred in enumerate(pred_arrs):
        oadd_arr = overlap_add(subj_pred, fpc, step, normed, hanning)
        oadd_arrs.append(oadd_arr)
    oadd_arrs = np.array(oadd_arrs, dtype=object)
    return oadd_arrs


def cut_wave_ends(pred_waves, gt_waves):
    for i in range(len(pred_waves)):
        pred_wave = pred_waves[i]
        gt_wave = gt_waves[i]
        if len(gt_wave) >= len(pred_wave):
            gt_waves[i] = gt_wave[:len(pred_wave)]
        else:
            pred_waves[i] = pred_wave[:len(gt_wave)]
    return pred_waves, gt_waves


def predict_all_subjects_HRs(pred_waves, fps=30, window_size=300, stride=1, maf_width=-1, pad_to_input=False):
    HRs = []
    for wave in pred_waves:
        wave = wave.astype(float)
        HR = predict_HR(wave, fps, window_size, stride, maf_width, pad_to_input)
        HRs.append(HR)
    HRs = np.array(HRs, dtype=object)
    return HRs


def infer_over_dataset_training(model, val_set, optimization_step, criterion, device, arg_obj, experiment_dir, epoch):
    loader = torch.utils.data.DataLoader(val_set, batch_size=1, shuffle=False, num_workers=1)
    all_losses = {'total': 0.0}
    loader_iterator = iter(loader)
    iter_length = len(loader)

    plot_psds = bool(int(arg_obj.plot_validation_psds))
    if plot_psds:
        num_plots = int(arg_obj.num_psd_plots)
        plot_idcs = np.random.choice(np.arange(iter_length), num_plots, replace=False)
        plot_dir = os.path.join(experiment_dir, 'psd_plots', str(epoch))
        os.makedirs(plot_dir, exist_ok=True)
        running_psd = None
        plot_i = 0

    for i in range(iter_length):
        try:
            data = next(loader_iterator)
        except StopIteration:
            loader_iterator = iter(loader)
            data = next(loader_iterator)
        with torch.set_grad_enabled(False):
            losses_dict, wave, freq, psd = optimization_step(model, data, criterion, device, val_set.fps, arg_obj, return_pred=True)
            psd = psd[0].cpu().numpy()
            if plot_psds:
                if running_psd is None:
                    running_psd = np.zeros_like(psd, dtype=float)
                running_psd = running_psd + psd
                if i in plot_idcs:
                    wave = wave[0].cpu().numpy()
                    freq = freq.cpu().numpy()
                    fig, axs = plt.subplots(2, 1, figsize=(12,9))
                    axs[0].plot(wave)
                    axs[0].set_xlabel('Frames')
                    axs[0].set_ylabel('Waveform')
                    axs[1].plot(freq, psd)
                    axs[1].set_xticks(np.arange(0, 15, 0.5), fontsize=8)
                    axs[1].set_xlabel('Frequency (Hz)')
                    axs[1].set_ylabel('Power')
                    plt.tight_layout()
                    plt.savefig(os.path.join(plot_dir, f'{plot_i}.jpg'), dpi=250)
                    plt.close()
                    plot_i += 1
            for k in losses_dict.keys():
                if not k in all_losses:
                    all_losses[k] = 0.0
                all_losses[k] = all_losses[k] + losses_dict[k]

    del loader_iterator
    if plot_psds:
        if not isinstance(freq, np.ndarray):
            freq = freq.cpu().numpy()
        running_psd = running_psd / np.sum(running_psd)
        mu = np.sum(running_psd*freq)
        sigma = np.sum((freq - mu)**2 * running_psd)
        plt.plot(freq, running_psd)
        plt.axvline(x=mu, c='k')
        plt.axvline(x=mu-sigma, c='k', linestyle='--')
        plt.axvline(x=mu+sigma, c='k', linestyle='--')
        plt.xticks(np.arange(0, 15, 0.5), rotation=90, fontsize=8)
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Power')
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, f'sum.jpg'), dpi=250)
        plt.close()
    for k in all_losses.keys():
        all_losses[k] = all_losses[k].cpu().numpy() / iter_length
    return all_losses


def infer_over_dataset_testing(model, val_set, criterion, device, arg_obj, normed=True):
    loader = torch.utils.data.DataLoader(val_set, batch_size=1, shuffle=False, num_workers=1)
    subjs = []
    wave_preds = []
    masks = []
    all_losses = 0
    loader_iterator = iter(loader)
    iter_length = len(loader)
    pbar = tqdm(total=iter_length)

    for i in range(iter_length):
        try:
            data = next(loader_iterator)
        except StopIteration:
            loader_iterator = iter(loader)
            data = next(loader_iterator)
        frames, wave, mask, subj = (data[0].to(device), data[1].to(device), data[2], data[3])
        with torch.set_grad_enabled(False):
            outputs = model(frames)
            loss = criterion(outputs, wave)
            all_losses += loss.item()
            wave_pred_copy = deepcopy(outputs.cpu().numpy())
            wave_preds.append(wave_pred_copy)
            subj_copy = deepcopy(subj.cpu().numpy())
            subjs.append(subj_copy)
            mask = deepcopy(mask.cpu().numpy())
            masks.append(mask)
            del subj
            del mask
            del loss
            del outputs
        pbar.update(1)

    pbar.close()
    del loader_iterator
    ave_loss = all_losses / iter_length
    pred_waves = np.vstack(wave_preds)
    masks = np.vstack(masks)
    subjs = np.hstack(subjs)
    orig_gt_waves = val_set.waves.copy()
    orig_masks = val_set.masks.copy()

    orig_lens = []
    for i in range(len(orig_masks)):
        orig_lens.append(orig_masks[i].sum())
    pred_waves = partition_by_subject(pred_waves, subjs)
    oadd_waves = overlap_add_all_subjects(pred_waves, arg_obj.fpc, arg_obj.step, normed=normed, hanning=True)
    masks = partition_by_subject(masks, subjs)
    masks = overlap_add_all_subjects(masks, arg_obj.fpc, arg_obj.step, normed=False, hanning=False)
    gt_waves = val_set.waves.copy()
    oadd_waves, gt_waves = cut_wave_ends(oadd_waves, gt_waves)

    for i in range(len(gt_waves)):
        masks[i] = masks[i] > 0.5
        gt_waves[i] = gt_waves[i][masks[i]]
        oadd_waves[i] = oadd_waves[i][masks[i]]
    window_size = int(np.round(arg_obj.window_size * arg_obj.fps))
    pred_HRs = predict_all_subjects_HRs(oadd_waves, fps=arg_obj.fps, window_size=window_size, maf_width=-1)
    gt_HRs = predict_all_subjects_HRs(gt_waves, fps=arg_obj.fps, window_size=window_size, maf_width=-1)

    return ave_loss, oadd_waves, pred_HRs, gt_waves, gt_HRs


def evaluate_predictions(pred_waves, pred_HRs, gt_waves, gt_HRs):
    flat_pred_waves = np.hstack((pred_waves))
    flat_pred_HRs = np.hstack((pred_HRs))
    flat_gt_waves = np.hstack((gt_waves))
    flat_gt_HRs = np.hstack((gt_HRs))
    ME_HR = np.mean(flat_gt_HRs - flat_pred_HRs)
    MAE_HR = np.mean(np.abs(flat_gt_HRs - flat_pred_HRs))
    RMSE_HR = np.sqrt(np.mean(np.square(flat_gt_HRs - flat_pred_HRs)))
    r_HR, p_HR = pearsonr(flat_gt_HRs, flat_pred_HRs)
    r_wave, p_wave = pearsonr(flat_gt_waves, flat_pred_waves)
    return ME_HR, MAE_HR, RMSE_HR, r_HR, r_wave


if __name__ == '__main__':
    main()


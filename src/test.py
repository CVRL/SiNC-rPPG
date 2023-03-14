import os
import torch
import torch.nn as nn
import numpy as np
import pickle
from scipy.stats import pearsonr
import argparse
import sys

from utils.model_selector import select_model
from utils import validate as validate_utils
from utils.losses import select_loss, select_validation_loss
from utils.optimization import select_optimization_step, optimization_loop, select_validation_step
from datasets.utils import get_dataset
import args


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def main():
    arg_obj = args.get_input()
    load_pickle = False

    experiment_root = arg_obj.experiment_root
    print(experiment_root)

    output_path = os.path.join('../predictions', experiment_root.split('/')[-1]+'.pkl')
    log_path = os.path.join('../results', experiment_root.split('/')[-1]+'.txt')

    testing_datasets = ['ubfc_testing', 'pure_testing', 'ddpm_testing']
    fold_dirs = os.listdir(experiment_root)

    if load_pickle:
        with open(output_path, 'rb') as infile:
            dataset_exper = pickle.load(infile)
    else:
        dataset_exper = {}
        for testing_dataset in testing_datasets:
            print('Using test set:', testing_dataset)
            print()
            exper = {}
            testing_dataset_name = testing_dataset.split('_')[0]
            for fold_dir in fold_dirs:
                print()
                str_splits = fold_dir.split('_')
                fold = str_splits[0][4:]
                seed = str_splits[1][4:]
                print(fold_dir, fold, seed)

                if seed not in exper:
                    exper[seed] = {}
                if fold not in exper[seed]:
                    exper[seed][fold] = {}

                experiment_dir = os.path.join(experiment_root, fold_dir)
                config_dict = parse_log(experiment_dir)

                ## Use the held-out test split if within-dataset testing, else all of dataset (except DDPM)
                training_dataset_name = config_dict['dataset'].split('_')[0]
                print('training, testing datasets:', training_dataset_name, testing_dataset_name)
                testing_split_idx = 2 if training_dataset_name == testing_dataset_name else 3
                test_split = ['train','val','test','all'][testing_split_idx]
                print('testing split:', test_split)
                arg_obj.dataset = testing_dataset
                arg_obj.K = int(config_dict['K'])
                arg_obj.fps = float(config_dict['fps'])
                arg_obj.fpc = int(config_dict['fpc'])
                arg_obj.step = int(config_dict['step'])
                test_set = get_dataset(test_split, arg_obj)

                arg_obj.model_type = config_dict['model_type']
                model = select_model(arg_obj)

                dummy_criterion = nn.MSELoss()

                ## Testing best model
                model_dir = os.path.join(experiment_dir, 'best_saved_models')
                model_tag = os.listdir(model_dir)[0]
                model_path = os.path.join(model_dir, model_tag)
                save_tag = model_tag + f'_{test_split}'
                print('best_model_path, save_tag:', model_path, save_tag)

                checkpoint = torch.load(model_path, map_location=device)
                model.load_state_dict(checkpoint['model_state_dict'])
                model = model.float().to(device)
                model.eval()

                ave_loss, pred_waves, pred_HRs, gt_waves, gt_HRs = validate_utils.infer_over_dataset_testing(model, test_set, dummy_criterion, device, arg_obj)
                exper[seed][fold]['pred_waves'] = pred_waves
                exper[seed][fold]['pred_HRs'] = pred_HRs
                exper[seed][fold]['gt_waves'] = gt_waves
                exper[seed][fold]['gt_HRs'] = gt_HRs

                ME_HR, MAE_HR, RMSE_HR, r_HR, r_wave = validate_utils.evaluate_predictions(pred_waves, pred_HRs, gt_waves, gt_HRs)
                print('ME, MAE, RMSE, r')
                print(f'{ME_HR:.3f} & {MAE_HR:.3f} & {RMSE_HR:.3f} & {r_HR:.3f}')
                print()

                exper[seed][fold]['ME'] = ME_HR
                exper[seed][fold]['MAE'] = MAE_HR
                exper[seed][fold]['RMSE'] = RMSE_HR
                exper[seed][fold]['r'] = r_HR

            dataset_exper[testing_dataset] = exper

        with open(output_path, 'wb') as outfile:
            pickle.dump(dataset_exper, outfile)

    print()
    print('Whole Dataset Values:')
    for testing_dataset in dataset_exper.keys():
        print('Testing dataset:', testing_dataset)
        exper = dataset_exper[testing_dataset]
        exper_errors = {'ME': [], 'MAE': [], 'RMSE': [], 'r': []}
        for seed in exper.keys():
            single_exper = exper[seed]
            pred_waves = []
            gt_waves = []
            pred_HRs = []
            gt_HRs = []
            for fold in single_exper.keys():
                pred_waves.append(single_exper[fold]['pred_waves'])
                gt_waves.append(single_exper[fold]['gt_waves'])
                pred_HRs.append(single_exper[fold]['pred_HRs'])
                gt_HRs.append(single_exper[fold]['gt_HRs'])
            pred_waves = np.hstack(pred_waves)
            gt_waves = np.hstack(gt_waves)
            pred_HRs = np.hstack(pred_HRs)
            gt_HRs = np.hstack(gt_HRs)
            ME_HR, MAE_HR, RMSE_HR, r_HR, r_wave = validate_utils.evaluate_predictions(pred_waves, pred_HRs, gt_waves, gt_HRs)
            print('ME, MAE, RMSE, r')
            print(f'{ME_HR:.3f} & {MAE_HR:.3f} & {RMSE_HR:.3f} & {r_HR:.3f}')
            exper_errors['ME'].append(ME_HR)
            exper_errors['MAE'].append(MAE_HR)
            exper_errors['RMSE'].append(RMSE_HR)
            exper_errors['r'].append(r_HR)
        print()
        ME_mu = np.mean(exper_errors['ME'])
        MAE_mu = np.mean(exper_errors['MAE'])
        RMSE_mu = np.mean(exper_errors['RMSE'])
        r_mu = np.mean(exper_errors['r'])
        ME_std = np.std(exper_errors['ME'])
        MAE_std = np.std(exper_errors['MAE'])
        RMSE_std = np.std(exper_errors['RMSE'])
        r_std = np.std(exper_errors['r'])

        with open(log_path, 'a+') as outfile:
            outfile.write(f'{testing_dataset}\n')
            outfile.write('ME, MAE, RMSE, r\n')
            outfile.write(f'{ME_mu:.2f} $\pm$ {ME_std:.2f} & {MAE_mu:.2f} $\pm$ {MAE_std:.2f} & {RMSE_mu:.2f} $\pm$ {RMSE_std:.2f} & {r_mu:.2f} $\pm$ {r_std:.2f}\n')
            outfile.write('\n')

        print(testing_dataset)
        print('ME, MAE, RMSE, r')
        print(f'{ME_mu:.2f} $\pm$ {ME_std:.2f} & {MAE_mu:.2f} $\pm$ {MAE_std:.2f} & {RMSE_mu:.2f} $\pm$ {RMSE_std:.2f} & {r_mu:.2f} $\pm$ {r_std:.2f}')
        print()

    print('Done.')
    return


def parse_log(experiment_dir):
    config_path = os.path.join(experiment_dir, 'arg_obj.txt')
    d = {}
    with open(config_path, 'r') as infile:
        for line in infile:
            clean_line = line.rstrip()
            splits = clean_line.split(' ')
            d[splits[0]] = splits[-1]
    return d


def get_errors(gt_HRs, pred_HRs):
    gt_HRs = np.hstack(gt_HRs.copy())
    pred_HRs = np.hstack(pred_HRs.copy())
    ME_HR = np.mean(gt_HRs - pred_HRs)
    MAE_HR = np.mean(np.abs(gt_HRs - pred_HRs))
    RMSE_HR = np.sqrt(np.mean(np.square(gt_HRs - pred_HRs)))
    r_HR, p_HR = pearsonr(gt_HRs, pred_HRs)
    return ME_HR, MAE_HR, RMSE_HR, r_HR


if __name__ == "__main__":
    main()

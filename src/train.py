import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import time
import os
import sys

from utils.model_selector import select_model
from utils.losses import select_loss, select_validation_loss
from utils.optimization import select_optimization_step, optimization_loop, select_validation_step
from utils import validate as validate_utils
from utils import model_utils
from utils.train_logger import TrainLogger
from datasets.utils import get_dataset
import args

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def main():
    arg_obj = args.get_input()
    args.print_args(arg_obj)

    seed = int(arg_obj.K / 5)
    torch.manual_seed(seed)
    np.random.seed(seed)

    if arg_obj.experiment_root is not None:
        experiment_root = arg_obj.experiment_root
        experiment_dir = os.path.join(experiment_root, f'fold{arg_obj.K}_seed{seed}')
        if not os.path.isdir(experiment_dir):
            os.makedirs(experiment_dir)
        else:
            if not bool(arg_obj.continue_training):
                print('Directory already exists:', experiment_dir, 'Exiting.')
                sys.exit(-1)
    else:
        experiment_root = '../experiments'
        experiment_dir = get_experiment_dir(experiment_root)


    print('Saving experiment to: ', experiment_dir)
    save_root = os.path.join(experiment_dir, 'saved_models')
    best_save_root = os.path.join(experiment_dir, 'best_saved_models')
    os.makedirs(save_root, exist_ok=True)
    os.makedirs(best_save_root, exist_ok=True)

    args.log_args(arg_obj, os.path.join(experiment_dir, 'arg_obj.txt'))
    logger = TrainLogger(experiment_dir, arg_obj, print_iter=1)

    model = select_model(arg_obj)
    model = model.float().to(device)

    train_set = get_dataset('train', arg_obj)
    val_set = get_dataset('val', arg_obj)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=arg_obj.batch_size,
                                               shuffle=True, num_workers=arg_obj.num_workers)

    if bool(arg_obj.scheduler):
        optimizer = optim.SGD(model.parameters(), lr=arg_obj.lr, momentum=0.9)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.5)
    else:
        if arg_obj.optimizer == 'adam':
            optimizer = optim.Adam(model.parameters(), lr=arg_obj.lr)
        elif arg_obj.optimizer == 'sgd':
            optimizer = optim.SGD(model.parameters(), lr=arg_obj.lr, momentum=0.9)
        else:
            optimizer = optim.AdamW(model.parameters(), lr=arg_obj.lr)

    train_criterions = select_loss(arg_obj)
    optimization_step = select_optimization_step(arg_obj)
    validation_criterion = select_validation_loss(arg_obj)
    validation_step = select_validation_step(arg_obj)

    best_loss = np.inf
    if bool(arg_obj.continue_training):
        checkpoint_path, last_epoch = model_utils.get_last_checkpoint(save_root)
        if checkpoint_path is not None:
            start_epoch = last_epoch + 1
            best_loss = model_utils.get_best_loss(best_save_root)
            print(f'Continuing model training from {checkpoint_path} with best_loss of {best_loss}.')
            checkpoint = torch.load(checkpoint_path)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            val_loss = checkpoint['loss']
            print('start_epoch:', start_epoch)
            print('val_loss:', val_loss)
            print('best_loss:', best_loss)

    val_losses = []
    model_paths = []
    start = time.time()

    global_i = 0
    for epoch in range(arg_obj.epochs):
        model.train()
        model, optimizer, logger, global_i = optimization_loop(model, train_loader, optimizer, optimization_step, train_criterions, logger, global_i, epoch, device, arg_obj)

        model.eval()
        val_loss = validate_utils.infer_over_dataset_training(model, val_set, validation_step, validation_criterion, device, arg_obj, experiment_dir, epoch)
        val_losses.append(val_loss['total'])

        print('Validation Loss: %.6f' % (val_loss['total']))
        print('Took %.3f seconds.' % (time.time() - start))
        print('************************')
        print('')

        logger.log_validation(val_loss, epoch)
        ########################################################################

        ## Save model for later
        save_path = create_save_path(save_root, epoch, arg_obj)
        torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': val_loss,
                    }, save_path)
        model_paths.append(save_path)

        if val_loss['total'] < best_loss:
            best_loss = val_loss['total']
            save_path = create_best_save_path(best_save_root, arg_obj)
            torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': val_loss,
                        }, save_path)

        if bool(arg_obj.scheduler):
            scheduler.step()

    print('Finished Training.')
    best_epoch = np.argmin(val_losses)
    best_path = model_paths[best_epoch]
    best_loss = val_losses[best_epoch]
    print(f'The best model was epoch {best_epoch} saved to {best_path}.')
    print(f'validation loss for best model ({arg_obj.validation_loss}):', round(best_loss, 4))
    print('')
    print('Took %.3f seconds total.' % (time.time() - start))
    logger.symlink_logfile()
    logger.close()


def create_save_path(root, epoch, arg_obj):
    file_name = f'{arg_obj.model_type}_e{epoch}'
    path = os.path.join(root, file_name)
    return path


def create_best_save_path(root, arg_obj):
    path = os.path.join(root, arg_obj.model_type)
    return path


def get_experiment_dir(root):
    dirs = sorted(os.listdir(root))
    if len(dirs) > 0:
        last_number = int(dirs[-1].split('_')[-1])
        last_number += 1
    else:
        last_number = 0
    experiment_dir = os.path.join(root, 'exper_%04d' % last_number)
    os.makedirs(experiment_dir, exist_ok=False)
    return experiment_dir


if __name__ == '__main__':
    main()


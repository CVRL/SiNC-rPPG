import argparse

def get_input():
    parser = argparse.ArgumentParser(description='Train and evaluate SiNC.')

    ########################## Experiment Details ############################
    parser.add_argument('--model_type',
                        default='physnet',
                        type=str,
                        help='Model architecture to use (physnet, rpnet). [physnet]')
    parser.add_argument('--experiment_root',
                        default=None,
                        type=str,
                        help='Root directory of the experimental results.')
    parser.add_argument('--K',
                        default=0,
                        type=int,
                        help='Fold to use when using K folds. [0]')
    parser.add_argument('--log_path',
                        default=None,
                        type=str,
                        help='Path to output log, which may be copied to the experiment directory at the end of the job.')
    parser.add_argument('--debug',
                        default=0,
                        type=int,
                        help='Whether to enter debugging mode, which decreases dataset size. [0]')
    parser.add_argument('--num_workers',
                        default=4,
                        type=int,
                        help='Number of cpus dedicated to preprocessing and feeding data. [4]')
    parser.add_argument('--continue_training',
                        default=0,
                        type=int,
                        help='Whether to continue training a model that has already been trained. [0]')
    parser.add_argument('--plot_validation_psds',
                        default=0,
                        type=int,
                        help='Whether to plot the PSDs from the validation set. [0]')
    parser.add_argument('--num_psd_plots',
                        default=10,
                        type=int,
                        help='Number of PSD plots. [10]')

    ################### Losses ########################
    parser.add_argument('--losses',
                        default='bsv',
                        type=str,
                        help='Loss functions used during training (b=bandwidth, s=sparsity, v=variance). [bsv]')
    parser.add_argument('--validation_loss',
                        default='bs',
                        type=str,
                        help='Loss functions used for selecting the best model. Similar to --losses. [bs]')
    parser.add_argument('--bandwidth_loss',
                        default='ipr',
                        type=str,
                        help='Loss function for enforcing spectral coverage. [ipr]')
    parser.add_argument('--sparsity_loss',
                        default='snr',
                        type=str,
                        help='Loss function for enforcing spectral sparsity. [snr]')
    parser.add_argument('--variance_loss',
                        default='emd',
                        type=str,
                        help='Loss function for enforcing variance over the batch. [emd]')
    parser.add_argument('--supervised_loss',
                        default='normnp',
                        type=str,
                        help='Loss function for supervised training. [normnp]')
    parser.add_argument('--supervised_scalar',
                        default=1.0,
                        type=float,
                        help='Scalar for supervised loss. [1.0]')
    parser.add_argument('--bandwidth_scalar',
                        default=1.0,
                        type=float,
                        help='Scalar for bandwidth loss. [1.0]')
    parser.add_argument('--sparsity_scalar',
                        default=1.0,
                        type=float,
                        help='Scalar for sparsity loss. [1.0]')
    parser.add_argument('--variance_scalar',
                        default=1.0,
                        type=float,
                        help='Scalar for variance loss. [1.0]')

    ################### Training Hyperparameters ########################
    parser.add_argument('--optimization_step',
                        default='unsupervised',
                        type=str,
                        help='Optimization step (supervised, unsupervised). [unsupervised]')
    parser.add_argument('--validation_step',
                        default='unsupervised',
                        type=str,
                        help='Validation step (e.g. what inputs and criterion are needed for the loss). [unsupervised]')
    parser.add_argument('--optimizer',
                        default='adamw',
                        type=str,
                        help='Optimizer (sgd, adam, adamw). [adamw]')
    parser.add_argument('--scheduler',
                        default=0,
                        type=int,
                        help='Whether to use a scheduler during training and SGD rather than AdamW. [0]')
    parser.add_argument('--lr',
                        default=0.0001,
                        type=float,
                        help='Learning rate. [0.0001]')
    parser.add_argument('--augmentation',
                        default='figscr',
                        type=str,
                        help='Augmentation during training. f=flipping, i=illumination \
                              changes, g=gaussian noise, s=speed, c=resizecropped, r=reverse. [figscr]')
    parser.add_argument('--speed_slow',
                        default=0.6,
                        type=float,
                        help='Speed for augmenting a video by slowing it down via interpolation. [0.6]')
    parser.add_argument('--speed_fast',
                        default=1.4,
                        type=float,
                        help='Speed for augmenting a video by speeding it up via interpolation. [1.4]')
    parser.add_argument('--dropout',
                        default=0.5,
                        type=float,
                        help='Dropout used in model. [0.5]')
    parser.add_argument('--batch_size',
                        default=20,
                        type=int,
                        help='Batch size for training. [20]')
    parser.add_argument('--fpc',
                        default=120,
                        type=int,
                        help='Frames per clip input to the model. [120]')
    parser.add_argument('--step',
                        default=60,
                        type=int,
                        help='Step between clips when training. [60]')
    parser.add_argument('--epochs',
                        default=40,
                        type=int,
                        help='Number of epochs to train. [40]')

    ################### Dataset Hyperparameters ########################
    parser.add_argument('--dataset',
                        default='pure_unsupervised',
                        type=str,
                        help='Dataset: {pure,ubfc,ddpm,hkbu,celebv}_{unsupervised,supervised,testing}. [pure_unsupervised]')
    parser.add_argument('--fps',
                        default=30,
                        type=int,
                        help='The framerate of the videos. Linear interpolation is used if necessary (30 or 90). [30]')
    parser.add_argument('--frame_width',
                        default=64,
                        type=int,
                        help='Width of input frames. [64]')
    parser.add_argument('--frame_height',
                        default=64,
                        type=int,
                        help='Height of input frames. [64]')
    parser.add_argument('--channels',
                        default='rgb',
                        type=str,
                        help='Input channels (any combo of {r,g,b} where order matters). [rgb]')

    ################### Frequency Analysis Hyperparameters ########################
    parser.add_argument('--low_hz',
                        default=0.66666667,
                        type=float,
                        help='Lower cutoff frequency.')
    parser.add_argument('--high_hz',
                        default=3.0,
                        type=float,
                        help='Upper cutoff frequency.')
    parser.add_argument('--window_size',
                        default=10,
                        type=int,
                        help='Size of each window in seconds for STFT. [10]')

    arg_obj = parser.parse_args()

    return arg_obj


def print_args(args):
    print('')
    for arg in sorted(vars(args)):
        val = getattr(args, arg)
        if val is not None:
            print('{0:<21} {1:<}'.format(arg, val))
        else:
            print('{0:<21} None'.format(arg))
    print('')


def log_args(args, file_path):
    with open(file_path, 'w') as outfile:
        for arg in sorted(vars(args)):
            val = getattr(args, arg)
            if val is not None:
                outfile.write('{0:<21} {1:<}\n'.format(arg, val))
            else:
                outfile.write('{0:<21} None\n'.format(arg))


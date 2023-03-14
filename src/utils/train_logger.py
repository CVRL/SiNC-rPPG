from torch.utils.tensorboard import SummaryWriter
import os

class TrainLogger:
    def __init__(self, experiment_dir, arg_obj, print_iter=200):
        self.log_dir = os.path.join(experiment_dir, 'runs')
        self.experiment_dir = experiment_dir
        self.arg_obj = arg_obj
        self.print_iter = print_iter
        self.running_losses = {}
        self.running_losses['total'] = 0.0
        self.writer = SummaryWriter(log_dir=self.log_dir, comment=f'{arg_obj.model_type}')

    def log(self, epoch, global_step, step, current_losses):
        for criterion in current_losses:
            if criterion not in self.running_losses:
                self.running_losses[criterion] = 0.0
            self.running_losses[criterion] += current_losses[criterion]
        if global_step % self.print_iter == (self.print_iter - 1):
            for criterion in current_losses:
                print(f'[{epoch}, {global_step:5d}] Train loss ({criterion}): {current_losses[criterion]:.6f}')
                self.writer.add_scalar(f'Training loss ({criterion})', self.running_losses[criterion] / self.print_iter, step)
                self.running_losses[criterion] = 0.0

    def log_validation(self, val_loss, epoch):
        for k in val_loss.keys():
            self.writer.add_scalar(f'Validation loss ({k})', val_loss[k], epoch)

    def close(self):
        self.writer.close()

    def symlink_logfile(self):
        log_path = self.arg_obj.log_path
        if log_path is not None:
            if os.path.exists(log_path):
                head, tail = os.path.split(log_path)
                log_dir = os.path.join(self.experiment_dir, 'logs')
                os.makedirs(log_dir, exist_ok=True)
                log_symlink = os.path.join(log_dir, tail)
                os.symlink(log_path, log_symlink)
            else:
                print('WARNING: Invalid log_path parameter, file does not exist.')


import sys
from utils.losses import torch_power_spectral_density, normalize_psd


def select_optimization_step(arg_obj):
    optim_step = arg_obj.optimization_step
    if optim_step == 'unsupervised':
        return unsupervised_train_step
    elif optim_step == 'supervised':
        return supervised_train_step
    else:
        print('Unknown optimization_step: {optim_step}, exiting.')
        sys.exit(-1)


def select_validation_step(arg_obj):
    validation_step = arg_obj.validation_step
    if validation_step == 'unsupervised':
        return unsupervised_validation_step
    elif validation_step == 'supervised':
        return supervised_validation_step
    else:
        print('Unknown validation_step: {validation_step}, exiting.')
        sys.exit(-1)


def optimization_loop(model, train_loader, optimizer, optimization_step, criterions, logger, global_i, epoch, device, arg_obj):
    model.train()
    for i, data in enumerate(train_loader, 0):
        optimizer.zero_grad()
        losses_dict = optimization_step(model, data, criterions, device, arg_obj)
        losses_dict['total'].backward()
        optimizer.step()
        global_i += 1
        logger.log(epoch, global_i, i, losses_dict)
    return model, optimizer, logger, global_i


def unsupervised_validation_step(model, data, criterions, device, fps, arg_obj, return_pred=False):
    frames = data[0].to(device)
    outputs = model(frames)
    freqs, psd = torch_power_spectral_density(outputs, fps=fps, normalize=False, bandpass=False)
    criterions_str = arg_obj.validation_loss
    losses_dict = accumulate_validation_losses(freqs, psd, criterions, device, arg_obj)
    if return_pred:
        psd = normalize_psd(psd)
        return losses_dict, outputs, freqs, psd
    return losses_dict


def supervised_train_step(model, data, criterions, device, arg_obj):
    frames, wave = (data[0].to(device), data[1].to(device))
    outputs = model(frames)
    loss = criterions['supervised'](outputs, wave)
    losses_dict = {'total': loss}
    return losses_dict


def supervised_validation_step(model, data, criterions, device, fps, arg_obj, return_pred=False):
    frames, wave = (data[0].to(device), data[1].to(device))
    outputs = model(frames)
    loss = criterions['supervised'](outputs, wave)
    losses_dict = {'total': loss}
    if return_pred:
        freqs, psd = torch_power_spectral_density(outputs, fps=fps, normalize=False, bandpass=False)
        psd = normalize_psd(psd)
        return losses_dict, outputs, freqs, psd
    return losses_dict


def unsupervised_train_step(model, data, criterions, device, arg_obj):
    fps = float(arg_obj.fps)
    low_hz = float(arg_obj.low_hz)
    high_hz = float(arg_obj.high_hz)
    frames, speed = (data[0].to(device), data[3])
    predictions = model(frames)
    predictions = add_noise_to_constants(predictions)
    freqs, psd = torch_power_spectral_density(predictions, fps=fps, low_hz=low_hz, high_hz=high_hz, normalize=False, bandpass=False)
    losses_dict = accumulate_unsupervised_losses(freqs, psd, speed, criterions, device, arg_obj)
    return losses_dict


def accumulate_unsupervised_losses(freqs, psd, speed, criterions, device, arg_obj):
    criterions_str = arg_obj.losses
    low_hz = float(arg_obj.low_hz)
    high_hz = float(arg_obj.high_hz)
    total_loss = 0.0
    losses_dict = {}
    if 'b' in criterions_str:
        bandwidth_loss = criterions['bandwidth'](freqs, psd, speed=speed, low_hz=low_hz, high_hz=high_hz, device=device)
        total_loss += (arg_obj.bandwidth_scalar*bandwidth_loss)
        losses_dict['bandwidth'] = bandwidth_loss
    if 's' in criterions_str:
        sparsity_loss = criterions['sparsity'](freqs, psd, speed=speed, low_hz=low_hz, high_hz=high_hz, device=device)
        total_loss += (arg_obj.sparsity_scalar*sparsity_loss)
        losses_dict['sparsity'] = sparsity_loss
    if 'v' in criterions_str:
        variance_loss = criterions['variance'](freqs, psd, speed=speed, low_hz=low_hz, high_hz=high_hz, device=device)
        total_loss += (arg_obj.variance_scalar*variance_loss)
        losses_dict['variance'] = variance_loss
    losses_dict['total'] = total_loss
    return losses_dict


def accumulate_validation_losses(freqs, psd, criterions, device, arg_obj):
    criterions_str = arg_obj.validation_loss
    total_loss = 0.0
    losses_dict = {}
    if 'b' in criterions_str:
        bandwidth_loss = criterions['bandwidth'](freqs, psd, device=device)
        total_loss = total_loss + (arg_obj.bandwidth_scalar*bandwidth_loss)
        losses_dict['bandwidth'] = bandwidth_loss
    if 's' in criterions_str:
        sparsity_loss = criterions['sparsity'](freqs, psd, device=device)
        total_loss = total_loss + (arg_obj.sparsity_scalar*sparsity_loss)
        losses_dict['sparsity'] = sparsity_loss
    losses_dict['total'] = total_loss
    return losses_dict


def add_noise_to_constants(predictions):
    B,T = predictions.shape
    for b in range(B):
        if torch.allclose(predictions[b][0], predictions[b]): # constant volume
            predictions[b] = torch.rand(T) - 0.5
    return predictions

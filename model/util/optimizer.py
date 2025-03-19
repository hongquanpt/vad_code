import torch.optim as optim


# https://github.com/vt-le/astnet
def get_optimizer(cfg, model):
    optimizer = None
    if cfg.TRAIN.optimizer.name == 'SGD':
        optimizer = optim.SGD(
            # filter(lambda p: p.requires_grad, model.parameters()),
            model.parameters(),
            lr=float(cfg.TRAIN.optimizer.lr),
            momentum=cfg.TRAIN.optimizer.momentum,
            weight_decay=cfg.TRAIN.optimizer.weight_decay,
            nesterov=cfg.TRAIN.optimizer.nesterov
        )
    elif cfg.TRAIN.optimizer.name == 'Adam':
        # print(f'float(cfg.TRAIN.optimizer.lr) = {float(cfg.TRAIN.optimizer.lr)}')
        # print(f'float(cfg.TRAIN.optimizer.eps) = {float(cfg.TRAIN.optimizer.eps)}')
        optimizer = optim.Adam(
            # filter(lambda p: p.requires_grad, model.parameters()),
            model.parameters(),
            lr=float(cfg.TRAIN.optimizer.lr),
            betas=cfg.TRAIN.optimizer.betas,
            eps=float(cfg.TRAIN.optimizer.eps),
            weight_decay=cfg.TRAIN.optimizer.weight_decay
        )
    elif cfg.TRAIN.optimizer.name == 'RMSprop':
        optimizer = optim.RMSprop(
            # filter(lambda p: p.requires_grad, model.parameters()),
            model.parameters(),
            lr=float(cfg.TRAIN.optimizer.lr),
            momentum=cfg.TRAIN.optimizer.momentum,
            weight_decay=cfg.TRAIN.optimizer.weight_decay,
            alpha=cfg.TRAIN.optimizer.rmsprop_alpha,
            centered=cfg.TRAIN.optimizer.rmsprop_centered
        )

    return optimizer


def get_optimizer_lr(cfg, model, lr):
    optimizer = None
    if cfg.TRAIN.optimizer.name == 'SGD':
        optimizer = optim.SGD(
            # filter(lambda p: p.requires_grad, model.parameters()),
            model.parameters(),
            lr=lr,
            momentum=cfg.TRAIN.optimizer.momentum,
            weight_decay=cfg.TRAIN.optimizer.weight_decay,
            nesterov=cfg.TRAIN.optimizer.nesterov
        )
    elif cfg.TRAIN.optimizer.name == 'Adam':
        optimizer = optim.Adam(
            # filter(lambda p: p.requires_grad, model.parameters()),
            model.parameters(),
            lr=lr,
            betas=cfg.TRAIN.optimizer.betas,
            eps=cfg.TRAIN.optimizer.eps,
            weight_decay=cfg.TRAIN.optimizer.weight_decay
        )
    elif cfg.TRAIN.optimizer.name == 'RMSprop':
        optimizer = optim.RMSprop(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=lr,
            momentum=cfg.TRAIN.optimizer.momentum,
            weight_decay=cfg.TRAIN.optimizer.weight_decay,
            alpha=cfg.TRAIN.optimizer.rmsprop_alpha,
            centered=cfg.TRAIN.optimizer.rmsprop_centered
        )

    return optimizer


def get_scheduler(cfg, optimizer):
    if cfg.TRAIN.lr_scheduler.name == 'LinearLR':
        scheduler = optim.lr_scheduler.LinearLR(optimizer, start_factor=0.5, total_iters=4)
    if cfg.TRAIN.lr_scheduler.name == 'LambdaLR':
        # https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.LambdaLR.html
        # https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.LinearLR.html
        epoch_count = 0  # the starting epoch count, eg: 0
        n_epochs = cfg.TRAIN.lr_scheduler.steps[0]  # eg: 99
        n_epochs_decay = cfg.TRAIN.lr_scheduler.steps[1]  # eg: 100

        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + epoch_count - n_epochs) / float(n_epochs_decay + 1)
            return lr_l

        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif cfg.TRAIN.lr_scheduler.name == 'StepLR':
        # https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.StepLR.html
        scheduler = optim.lr_scheduler.StepLR(optimizer,
                                              step_size=cfg.TRAIN.lr_scheduler.step_size,
                                              gamma=cfg.TRAIN.lr_scheduler.gamma)
    elif cfg.TRAIN.lr_scheduler.name == 'MultiStepLR':
        # https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.MultiStepLR.html
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
                                                   cfg.TRAIN.lr_scheduler.milestones,
                                                   cfg.TRAIN.lr_scheduler.gamma)
    elif cfg.TRAIN.lr_scheduler.name == 'CosineAnnealingLR':
        # https://discuss.pytorch.org/t/how-to-implement-torch-optim-lr-scheduler-cosineannealinglr/28797/6
        # https://www.tutorialexample.com/understand-torch-optim-lr_scheduler-cosineannealinglr-with-examples-pytorch-tutorial/
        # https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.CosineAnnealingLR.html
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                         T_max=cfg.TRAIN.end_epoch,  # Maximum number of iterations.
                                                         eta_min=cfg.TRAIN.lr_scheduler.eta_min
                                                         # Minimum learning rate.
                                                         )
    else:
        return NotImplementedError('Learning rate policy [%s] is not implemented', cfg.TRAIN.lr_scheduler.name)

    return scheduler


def update_learning_rate(cfg, optimizer):
    """Update learning rates for all the networks; called at the end of every epoch"""
    old_lr = optimizer.param_groups[0]['lr']
    scheduler = get_scheduler(cfg, optimizer)
    scheduler.step()

    lr = optimizer.param_groups[0]['lr']
    print('learning rate %.7f -> %.7f' % (old_lr, lr))


def update_learning_rate_linear(cfg, optimizer, epoch):
    epoch_count = 0  # the starting epoch count, eg: 0
    n_epochs = cfg.TRAIN.lr_scheduler.steps[0]  # eg: 99
    n_epochs_decay = cfg.TRAIN.lr_scheduler.steps[1]  # eg: 100
    base_lr = cfg.TRAIN.optimizer.learning_rate

    rule = 1.0 - max(0, epoch + epoch_count - n_epochs) / float(n_epochs_decay)
    lr = base_lr * rule
    optimizer.param_groups[0]['lr'] = lr

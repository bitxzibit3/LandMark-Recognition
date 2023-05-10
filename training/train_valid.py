import numpy as np
import time
import torch

from tqdm import tqdm


def train_valid(model, train_dl, valid_dl,
                opt_cls, opt_params, loss_fn,
                metric_fn, max_epochs: int,
                device,
                scheduler_cls=None, scheduler_params=None):
    """
    Train and validation cycle.

    train_dl - DataLoader with train data
    valid_dl - DataLoader with valid data
    opt - optimizer
    loss_fn - loss function
    metric_fn - metric function to evaluate model
    max_epochs - epochs to training and validation
    scheduler_cls - class of scheduler
    scheduler_params - parameters for scheduler
    """
    train_losses = []
    valid_losses = []
    train_metrics = []
    valid_metrics = []

    def print_loss_metric_info(train_loss=train_losses,
                               valid_loss=valid_losses,
                               train_metric=train_metrics,
                               valid_metric=valid_metrics):
        """
        Logger function
        """
        template = '\n'.join(['',
                              'Losses on train: {}',
                              'Losses on valid: {}',
                              'Metric on train: {}',
                              'Metric on valid: {}'])
        print(template.format(train_loss,
                              valid_loss,
                              train_metric,
                              valid_metric))

    # Optimizer initialization
    opt = opt_cls(params=model.parameters(),
                  **opt_params)

    # Scheduler initialization
    if scheduler_cls:
        scheduler = scheduler_cls(optimizer=opt,
                                  **scheduler_params)
    else:
        scheduler = None
    model = model.to(device)
    train_time = 0
    valid_time = 0
    for epoch in tqdm(range(max_epochs), desc='Epoch'):
        # Training cycle
        model.train()
        train_losses_epoch = []
        train_metric_epoch = []
        print_loss_metric_info()
        begin_time = time.time()
        for x, y in tqdm(train_dl):
            opt.zero_grad()
            x, y = x.to(device), y.to(device)
            output = model(x)
            y_pred = torch.argmax(output, dim=-1)

            loss = loss_fn(output, y)
            loss.backward()
            opt.step()
            metric_value = metric_fn(y.to('cpu'), y_pred.to('cpu'), average='macro')
            train_metric_epoch.append(metric_value)
            train_losses_epoch.append(loss.item())
        train_time += (time.time() - begin_time)
        train_losses.append(np.mean(train_losses_epoch))
        train_metrics.append(np.mean(train_metric_epoch))

        # Valid cycle
        model.eval()
        valid_losses_epoch = []
        valid_metric_epoch = []
        print_loss_metric_info()
        begin_time = time.time()
        with torch.no_grad():
            for x, y in tqdm(valid_dl):
                x, y = x.to(device), y.to(device)
                output = model(x)
                y_pred = torch.argmax(output, dim=-1)

                loss = loss_fn(output, y)

                metric_value = metric_fn(y.to('cpu'), y_pred.to('cpu'), average='macro')
                valid_losses_epoch.append(loss.item())
                valid_metric_epoch.append(metric_value)

        valid_metrics.append(np.mean(valid_metric_epoch))
        valid_losses.append(np.mean(valid_losses_epoch))
        valid_time += (time.time() - begin_time)

        if scheduler:
            scheduler.step()
    return train_losses, valid_losses, train_metrics, valid_metrics, train_time, valid_time

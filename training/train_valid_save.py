import os
import torch
import wandb
import train_valid

from utils import make_loaders


def train_valid_save(model, artifact_config,
                     preprocess_config,
                     train_config):
    art = wandb.Artifact(**artifact_config)
    exp_name = artifact_config['name']
    device = train_config['device']
    print('Making dataloaders...')
    train_dl, valid_dl = make_loaders(**preprocess_config)
    os.system('clear')
    train_losses, valid_losses, train_metric, valid_metric, train_time, valid_time = train_valid.train_valid(
        model=model,
        train_dl=train_dl,
        valid_dl=valid_dl,
        **train_config
    )

    # Saving model
    torch.save(model.state_dict(),
               './models/models/' + exp_name + '.pth')

    epochs = train_config['max_epochs']
    for_table = list(zip(range(1, epochs + 1),
                         train_losses,
                         valid_losses,
                         train_metric,
                         valid_metric))

    tabled_cfg = wandb.Table(
        columns=['Epoch', 'Train losses', 'Valid losses', 'Train score', 'Valid score'],
        data=for_table
    )

    # Model state dict
    art.add_file('./models/models/' + exp_name + '.pth',
                 name='state_dict.pth')

    # Losses and metrics
    art.add(tabled_cfg, 'Losses and scores table')

    # Add result description
    result_config = {'Train time': train_time,
                     'Valid time': valid_time,
                     'Device': device}

    # Add configuration
    common_config = {'Preprocess': preprocess_config,
                     'Training': train_config,
                     'Resulting': result_config}

    art.metadata = common_config

    x = next(model.modules())
    with open('./models/desc/' + exp_name + '.txt', 'w') as f:
        f.write(str(x))

    art.add_file('./models/desc/' + exp_name + '.txt',
                 name='desc.txt')

    wandb.log_artifact(art)

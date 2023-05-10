from torch.utils.data import DataLoader


def make_loaders(ds_cls, train_size, train_bs, valid_bs, ds_params):
    """
    Return two DataLoaders: train loader and valid loader
    ds_cls - class of using dataset
    train_size - param for splitting dataset to train and valid parts
    train_bs - train loader`s batch size
    valid_bs - valid loader`s batch_size
    ds_params - parameters to create dataset
    """

    ds = ds_cls(**ds_params)
    train_ds, valid_ds = ds.train_valid_split(train_size=train_size)

    train_dl = DataLoader(train_ds, batch_size=train_bs,
                          shuffle=True, num_workers=1)

    valid_dl = DataLoader(valid_ds, batch_size=valid_bs,
                          shuffle=False, num_workers=1)

    return train_dl, valid_dl

import PIL

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset
from torchvision import transforms as tf


class CustomDataset(Dataset):
    def __init__(self, files, mode='train',
                 transform=None,
                 image_shape=(200, 200)):

        """
        mode - train/valid/test
        files - list/set with filepaths
        transform - processing of file
        image_shape - shape of result tensor
        """

        self.mode = mode
        self.transform = transform
        self.image_shape = image_shape
        self.files = files

        self.check_mode = self.mode in ('train', 'valid')

        labels = list(set([self.get_label(filename) for filename in files]))
        self.le = LabelEncoder()
        self.le.fit(labels)

    def __len__(self):
        return len(self.files)

    def default_transform(self, img):
        """
        Make image resizing, and converting to tensor
        """
        transform = tf.Compose([
            tf.Resize(self.image_shape),
            tf.PILToTensor()
        ])
        return transform(img)

    def __getitem__(self, idx):
        path = self.files[idx]
        with PIL.Image.open(path) as img:
            if self.transform:
                tensor = self.transform(img)
            else:
                tensor = self.default_transform(img)

        tensor = tensor / 255
        tensor = tensor.float()

        if self.check_mode:
            label = self.get_label(path)
            return tensor, self.le.transform([label])[0]
        else:
            return tensor

    def get_label(self, path):
        assert self.check_mode, \
            'It is not possible to get label'
        return path.split('/')[2]

    def decode(self, num_label):
        return self.le.inverse_transform([num_label])[0]

    def train_valid_split(self, train_size=0.9):
        """
        Uniform split of files.

        Returns two datasets: train_dataset and valid_dataset (augmentations = [None])
        """

        def handle_one_class(label):
            file_list = get_class_samples(label)
            train_set, valid_set = train_test_split(tuple(file_list),
                                                    train_size=train_size)
            return train_set, valid_set

        def get_class_samples(label):
            return set([filename
                        for filename in self.files if label in filename.split('/')])

        train_list = []
        valid_list = []
        labels = self.le.classes_

        for label in labels:
            cur_train_list, cur_valid_list = handle_one_class(label)
            train_list.extend(cur_train_list)
            valid_list.extend(cur_valid_list)

        train_ds = CustomDataset(mode='train',
                                 image_shape=self.image_shape,
                                 files=train_list)

        valid_ds = CustomDataset(mode='valid',
                                 image_shape=self.image_shape,
                                 files=valid_list)
        return train_ds, valid_ds

import matplotlib.pyplot as plt
import PIL

from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torchvision import transforms as tf


class AugmentedCustomDataset(Dataset):
    def __init__(self, files, mode='train',
                 transform=None,
                 image_shape=(200, 200), augmentations=None):
        """
        mode - train/valid/test
        files - list/set with filepaths
        labels - list with all possible namelabels
        transform - processing of file
        image_shape - shape of result tensor
        """
        self.mode = mode
        self.transform = transform
        self.image_shape = image_shape

        self.check_mode = self.mode in ('train', 'valid')

        labels = list(set([self.get_label(filename) for filename in files]))
        self.le = LabelEncoder()
        self.le.fit(labels)

        # Initialize augmentation options
        if augmentations:
            self.augmentations = augmentations
        else:
            self.augmentations = [
                None,
                tf.ColorJitter(brightness=0.3,
                               contrast=0.3,
                               saturation=0.3),
                tf.RandomPosterize(bits=2, p=1),
                tf.RandomAdjustSharpness(sharpness_factor=2,
                                         p=1),
                tf.RandomEqualize(p=1),
                tf.RandomRotation(degrees=(-20, 20)),
                tf.RandomHorizontalFlip(p=1)
            ]
        self.augmentations_amount = len(self.augmentations)
        if self.augmentations == [None]:
            self.files = files

        else:
            self.files = []
            for filename in files:
                augmented_filenames = [(filename, i)
                                       for i in range(self.augmentations_amount)]
                self.files.extend(augmented_filenames)

        self._len = len(self.files)

    def __len__(self):
        return self._len

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
        # Find path to file depending on idx
        filename, augment_idx = self.files[idx]
        augment = self.augmentations[augment_idx]
        with PIL.Image.open(filename) as img:
            if self.transform:
                tensor = self.transform(img)
            else:
                tensor = self.default_transform(img)

            if augment:
                tensor = augment(tensor)

            tensor = tensor / 255

        if self.check_mode:
            label = self.get_label(filename)
            return tensor, self.encode(label)
        else:
            return tensor

    def get_label(self, path):
        assert self.check_mode, \
            'It is not possible to get label'
        return path.split('/')[-2]

    def encode(self, str_label):
        return self.le.transform([str_label])[0]

    def decode(self, num_label):
        return self.le.inverse_transform([num_label])[0]

    def get_augmented_samples(self, idx):
        begin_idx = idx * self.augmentations_amount
        return [self[begin_idx + i][0] for i in range(self.augmentations_amount)]

    def draw_augmented_samples(self, idx):
        samples = self.get_augmented_samples(idx)
        plt.figure(figsize=(20, 20))
        for i, sample in enumerate(samples):
            plt.subplot(1, len(samples), i + 1)
            plt.imshow(sample.permute(1, 2, 0))

    def train_valid_split(self, train_size=0.9):
        '''
        Unfirom split of files.

        Returns two datasets: train_dataset and valid_dataset (augmentations = [None])
        '''

        def handle_one_class(label):
            file_list = get_class_samples(label)
            train_set, valid_set = train_test_split(tuple(file_list),
                                                    train_size=train_size)
            return train_set, valid_set

        def get_class_samples(label):
            return set([filename
                        for filename in self.files if label in filename[0].split('/')])

        train_list = []
        valid_list = []
        labels = self.le.classes_

        for label in labels:
            cur_train_list, cur_valid_list = handle_one_class(label)
            train_list.extend(cur_train_list)
            valid_list.extend(cur_valid_list)

        train_ds = AugmentedCustomDataset(mode='train',
                                          image_shape=self.image_shape,
                                          files=train_list,
                                          augmentations=[None])
        train_ds.augmentations = self.augmentations

        valid_ds = AugmentedCustomDataset(mode='valid',
                                          image_shape=self.image_shape,
                                          files=valid_list,
                                          augmentations=[None])
        valid_ds.augmentations = self.augmentations
        return train_ds, valid_ds

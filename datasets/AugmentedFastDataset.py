import PIL

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset
from torchvision import transforms as tf


class AugmentedFastDataset(Dataset):
    def __init__(self, files, mode='train',
                 transform=None,
                 image_shape=(200, 200)):
        """
        files - list with paths to files
        mode - train/valid/test
        transform - processing of file
        image_shape - shape of result tensor
        """

        self.mode = mode
        self.image_shape = image_shape
        self.transform = transform if transform \
            else tf.Compose([tf.Resize(image_shape), tf.PILToTensor()])

        self.x = []
        self.y = []

        self.check_mode = self.mode in ('train', 'valid')
        self._len = len(files)

        labels = list(set([self.get_label(filename) for filename in files]))
        self.le = LabelEncoder()
        self.le.fit(labels)

        self.augmentations = (
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
        )

        self.augmentations_amount = len(self.augmentations)

        # Saving tensors from PIL.Image
        for path in files:
            label = path.split('/')[-2]
            tensor = self.get_sample(path)
            augmentations = self.get_augmented_samples(tensor)
            self.x.extend(augmentations)
            self.y.extend([label] * self.augmentations_amount)

    def get_sample(self, filepath):
        with PIL.Image.open(filepath) as image:
            image = PIL.Image.open(filepath)
            tensor = self.transform(image)
        return tensor

    def get_label(self, path):
        assert self.check_mode, \
            'It is not possible to get label'
        return path.split('/')[-2]

    def get_augmented_samples(self, tensor):
        answer = [tensor / 255]
        answer.extend(
            [augmentation(tensor) / 255
             for augmentation in self.augmentations if augmentation]
        )
        return answer

    def __len__(self):
        return self._len * self.augmentations_amount

    def __getitem__(self, idx):
        """
        Returns Tensor, str (optional)
        """
        if self.check_mode:
            y = self.le.transform([self.y[idx]])
            return self.x[idx], y[0]
        else:
            return self.x[idx]

    def decode(self, num_label):
        return self.le.inverse_transform([num_label])[0]

    def train_valid_split(self, train_size=0.9):
        """
        Unfirom split of files.

        Returns two datasets: train_dataset and valid_dataset (augmentations = [None])
        """

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

        train_ds = AugmentedFastDataset(mode='train',
                                        image_shape=self.image_shape,
                                        files=train_list)
        train_ds.augmentations = self.augmentations

        valid_ds = AugmentedFastDataset(mode='valid',
                                        image_shape=self.image_shape,
                                        files=valid_list)
        valid_ds.augmentations = self.augmentations
        return train_ds, valid_ds

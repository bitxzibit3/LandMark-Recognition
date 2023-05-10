import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import matplotlib.pyplot as plt
import numpy as np

from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


class AdvancedCustomDataset(Dataset):
    def __init__(self, augmentate, files, ex_amount=1000, mode='train',
                 transform=None, image_shape=(200, 200), augmentations=None):
        """
        augmentate - do augmentation or not
        ex_amount - number of photo per class
        mode - train/valid/test
        files - list/set with filepaths
        transform - processing of file
        image_shape - shape of result tensor
        """
        self.mode = mode
        self.transform = transform \
            if transform \
            else A.Resize(*image_shape)
        self.image_shape = image_shape
        self.ex_amount = ex_amount
        self.check_mode = self.mode in ('train', 'valid')
        # Labels initialization
        labels = list(set([self.get_label(filename) for filename in files]))
        self.le = LabelEncoder()
        self.le.fit(labels)

        # Initialize augmentation options
        if augmentations:
            self.augmentations = augmentations
        else:
            self.augmentations = (
                A.ColorJitter(brightness=0.3,
                              contrast=0.3,
                              saturation=0.3),
                A.Posterize(num_bits=2, p=1),
                A.Sharpen(alpha=(0.9, 1.0)),
                A.Equalize(p=1),
                A.Rotate(limit=(-20, 20), p=1),
                A.HorizontalFlip(p=1)
            )
        self.augmentations_amount = len(self.augmentations)
        self.files = files
        if augmentate:
            self.files = self.augmentate()

        self._len = len(self.files)

    def augmentate(self):
        labels = self.le.classes_
        new_files = []
        for label in labels:
            new_files_for_label = self.augmentate_one_class(label)
            new_files.extend(new_files_for_label)
        return new_files

    def augmentate_one_class(self, label):
        ex_amount = self.ex_amount
        files = self.get_class_samples(label)
        new_files = []
        while len(new_files) < ex_amount:
            filename = np.random.choice(files, size=1)[0]
            augmentations_amount = np.random.randint(low=0,
                                                     high=self.augmentations_amount)
            if augmentations_amount:
                augmentations = np.random.choice(a=self.augmentations,
                                                 size=augmentations_amount,
                                                 replace=False)
                augmentations = A.Compose(augmentations)
                new_files.append((filename, augmentations))
            else:
                new_files.append((filename, None))
        return new_files

    def get_class_samples(self, label):
        return [filename
                for filename in self.files if label in filename.split('/')]

    def __len__(self):
        return self._len

    def __getitem__(self, idx):
        # Find path to file depending on idx
        filename, augmentations = self.files[idx]
        img = cv2.imread(filename)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        tensor = self.transform(image=img)['image']
        if augmentations:
            tensor = augmentations(image=tensor)['image']

        tensor = tensor / 255
        tensor = ToTensorV2()(image=tensor)['image'].float()

        if self.check_mode:
            label = self.get_label(filename)
            return tensor, self.encode(label)
        else:
            return tensor

    def get_augmented_samples(self, idx):
        """
        Method to get all augmentations with the same image
        idx - index in files
        """
        filename = self.files[idx][0]
        answer = [item for item in self.files
                  if filename == item[0]]
        return answer

    def draw_augmented_samples(self, idx):
        files = self.get_augmented_samples(idx)
        columns = 5
        number = len(files)
        if number % columns:
            lines = int(number / columns) + 1
        else:
            lines = int(number / columns)
        print(f'{number}: {lines}:{columns}')
        plt.figure(figsize=(20, 20))
        for idx, item in enumerate(files):
            filename, augmentation = item
            img = cv2.imread(filename)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if augmentation:
                img = augmentation(image=img)['image']
            plt.subplot(lines, columns, idx + 1)
            plt.imshow(img)

    def get_label(self, path):
        assert self.check_mode, \
            'It is not possible to get label'
        return path.split('/')[-2]

    def encode(self, str_label):
        return self.le.transform([str_label])[0]

    def decode(self, num_label):
        return self.le.inverse_transform([num_label])[0]

    def train_valid_split(self, train_size=0.9):
        """
        Uniform split of files.

        Returns two datasets: train_dataset and valid_dataset
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

        train_ds = AdvancedCustomDataset(augmentate=False, mode='train',
                                         image_shape=self.image_shape,
                                         files=train_list)
        train_ds.augmentations = self.augmentations

        valid_ds = AdvancedCustomDataset(augmentate=False, mode='valid',
                                         image_shape=self.image_shape,
                                         files=valid_list)
        valid_ds.augmentations = self.augmentations
        return train_ds, valid_ds

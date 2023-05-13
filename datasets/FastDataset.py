import PIL

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset
from torchvision import transforms as tf


class FastDataset(Dataset):
    def __init__(self, files, mode='train',
                 transform=None,
                 image_shape=(200, 200)):

        """
        mode - train/valid/test
        files - list with paths to files
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
        get_label = lambda x: x.split('/')[-2]
        labels = list(set([get_label(filename) for filename in files]))
        self.le = LabelEncoder()
        self.le.fit(labels)

        # Saving tensors from PIL.Image
        for path in files:
            label = path.split('/')[-2]
            tensor = self.get_sample(path)
            self.x.append(tensor / 255)
            self.y.append(label)

        self._len = len(self.x)

    def get_sample(self, filepath):
        with PIL.Image.open(filepath) as image:
            image = PIL.Image.open(filepath)
            tensor = self.transform(image)
        return tensor

    def __len__(self):
        return self._len

    def __getitem__(self, idx):
        """
        Returns Tensor, str (optional)
        """
        if self.check_mode:
            y = self.le.transform([self.y[idx]])
            return self.x[idx].float(), y[0]
        else:
            return self.x[idx].float()

    def decode(self, num_label):
        return self.le.inverse_transform([num_label])[0]

    def train_valid_split(self, train_size=0.9): # REJECTED
        """
        Uniform split of files.

        Returns two datasets: train_dataset and valid_dataset
        """

        assert self.check_mode, 'Test can not be splitted'
        dictionary = {}
        for x, y in self:
            if y not in dictionary:
                dictionary[y] = [x]
            else:
                dictionary[y].append(x)

        def handle_one_class(objects):
            train_set, valid_set = train_test_split(tuple(objects),
                                                    train_size=train_size)
            return train_set, valid_set

        train_list = []
        valid_list = []
        labels = self.le.classes_

        for label in labels:
            cur_train_list, cur_valid_list = handle_one_class(label)
            train_list.extend(cur_train_list)
            valid_list.extend(cur_valid_list)

        train_ds = FastDataset(mode='train',
                               image_shape=self.image_shape,
                               files=train_list)

        valid_ds = FastDataset(mode='valid',
                               image_shape=self.image_shape,
                               files=valid_list)
        return train_ds, valid_ds

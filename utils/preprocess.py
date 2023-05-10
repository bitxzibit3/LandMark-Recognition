import os
from PIL import Image
import torch

INIT_PATH = './data/landmarks/'


def preprocess(init_path=INIT_PATH):
    # Get all filepaths
    all_files = set()
    labels = []
    for path, dirs, files in os.walk(init_path):
        if not dirs:
            for file_ in files:
                filepath = '/'.join([path, file_])
                all_files.add(filepath)
        else:
            labels.extend(dirs)
    # Filtering
    supported_types = 'RGB'

    incorrect_files = set()
    for filename in all_files:
        img = Image.open(filename)
        if img.mode not in supported_types:
            incorrect_files.add(filename)
        del img

    correct_files = list(all_files - incorrect_files)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    return correct_files, incorrect_files, labels, device


if __name__ == '__main__':
    correct_files, incorrect_files, labels, device = preprocess()
    print('\n'.join([f'Corrected files: {len(correct_files)}',
                     f'Deleted files: {len(incorrect_files)}',
                     f'Labels amount: {len(labels)}',
                     f'Device: {device}']))

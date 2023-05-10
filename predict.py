import albumentations as A
import cv2
import os
import requests
import torch
import torch.nn as nn

from albumentations.pytorch import ToTensorV2
from argparse import ArgumentParser
from torchvision import models


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('image_path', type=str,
                        help='Path to image to predict.')
    parser.add_argument('net', type=str,
                        help='Model to use to predict. Available models: mynet, vgg13')

    return parser.parse_args()


def build_model(net: str):
    """
    Build a model and returns preprocess transforms.
    """
    if net == 'mynet':
        from models.my_model import Net
        net = Net()
        if not os.path.exists('./models/my_cnn.pth'):
            print('Mynet not on computer, downloading state dict...')
            url = 'https://api.wandb.ai/artifactsV2/gcp-us/ml_landmarks/QXJ0aWZhY3Q6NDMxMjI1NDA1' \
                  '/e145661cd08a805dc454469d20af08f6/state_dict.pth'
            path = './models/my_cnn.pth'
            r = requests.get(url)
            with open(path, 'wb') as f:
                f.write(r.content)
        net.load_state_dict(torch.load('./models/my_cnn.pth'))
        tf = A.Resize(100, 100)
    elif net == 'vgg13':
        net = models.vgg13_bn()
        net.classifier[-4] = nn.Linear(in_features=4096,
                                       out_features=1024)
        net.classifier[-1] = nn.Linear(in_features=1024,
                                       out_features=210)
        print('VGG13 not on computer, downloading state dict...')
        url = 'https://api.wandb.ai/artifactsV2/gcp-us/ml_landmarks/QXJ0aWZhY3Q6NDQ2NzIyODM3' \
              '/088752866cd6ebfca343024efd297925/state_dict.pth'
        path = './models/vgg13.pth'
        r = requests.get(url)
        with open(path, 'wb') as f:
            f.write(r.content)
        net.load_state_dict(torch.load('./models/vgg13.pth'))
        tf = A.Resize(224, 224)
    else:
        raise AssertionError('Wrong net')

    return net, tf


def open_image(path, tf):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    tensor = tf(image=img)['image']
    tensor = tensor / 255
    width = tf.width
    height = tf.height
    tensor = ToTensorV2()(image=tensor)['image'].float()
    return tensor.view(1, 3, width, height)


def decode(answer):
    idx = torch.argmax(answer)
    with open('./data/classes.txt', 'r') as f:
        labels = f.read()
    labels = labels.split('\n')
    print(labels[idx])


def main():
    args = parse_args()
    path, model = args.image_path, args.net
    model, tf = build_model(model)
    tensor = open_image(path, tf)
    model.eval()
    answer = model(tensor)
    decode(answer)


if __name__ == '__main__':
    main()
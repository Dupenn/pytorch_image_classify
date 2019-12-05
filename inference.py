# coding:utf8

import torch
from torch.autograd import Variable
from torchvision import transforms, models
from PIL import Image
import sys
import torch.nn.functional as F

from net.simple import simpleconv3

data_transforms = transforms.Compose([
    transforms.Resize(48),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

net = models.resnet18(num_classes=2)
model_path = sys.argv[1]
net.load_state_dict(torch.load(model_path, map_location=lambda storage, loc: storage))

image_path = sys.argv[2]
print("image_path=", image_path)
image = Image.open(image_path)
img_blob = data_transforms(image).unsqueeze(0)
img_blob = Variable(img_blob)

torch.no_grad()

predict = F.softmax(net(img_blob), dim=1)
print(predict)

import torch
from torch.autograd import Variable
from torchvision import transforms
from PIL import Image
import sys
import torch.nn.functional as F

from net.simple import simpleconv3

data_transforms = transforms.Compose([
    transforms.Resize(48),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

net = simpleconv3()
modelpath = sys.argv[1]
net.load_state_dict(torch.load(modelpath, map_location=lambda storage, loc: storage))

imagepath = sys.argv[2]
print("image_path=", imagepath)
image = Image.open(imagepath)
imgblob = data_transforms(image).unsqueeze(0)
imgblob = Variable(imgblob)

torch.no_grad()

predict = F.softmax(net(imgblob), dim=1)
print(predict)

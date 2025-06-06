import torch
import torch.nn as nn
import numpy as np

from darknet import *
from torch.ao.quantization import QuantStub, DeQuantStub
class YoloV2Net(nn.Module):
    def __init__(self, num_anchors=5, num_classes=80):
        super(YoloV2Net, self).__init__()
        self.num_classes = num_classes
        self.anchors = [0.57273, 0.677385, 1.87446, 2.06253, 3.33843,
                        5.47434, 7.88882, 3.52778, 9.77052, 9.16888]
        self.num_anchors = num_anchors
        self.darknet = Darknet()
     
        self.conv1 = nn.Sequential(
             Conv2D(1024, 1024, 3),
             Conv2D(1024, 1024, 3))
     
        self.conv2 = nn.Sequential(
             Conv2D(512, 64, 1))
     
        self.conv  = nn.Sequential(
             Conv2D(1280, 1024, 3),
             nn.Conv2d(1024, self.num_anchors * (self.num_classes + 5), 1))
        self.quant   = QuantStub()
        self.dequant = DeQuantStub()
     
    def forward(self, x):
        x  = self.quant(x)
        x1, x2 = self.darknet(x)
        x1 = self.conv1(x1)
        x2 = self.conv2(x2)
        x2 = Reorg(x2)
        x = torch.cat([x2, x1], 1)
        x = self.conv(x)
        x  = self.dequant(x)
        return x
    
    # 🔑 ①—Fuse 專用函式
    def fuse_model(self):
        # darknet 內部 Conv+BN+ReLU
        for blk in [self.darknet, self.conv1, self.conv2, self.conv]:
            for m in blk.modules():
                if isinstance(m, Conv2D):
                    # ⭐ 只 fuse ["conv", "bn"]，不要帶 LeakyReLU
                    torch.ao.quantization.fuse_modules(
                        m, ["conv", "bn"], inplace=True)


def load_weights(model, wt_file):
    """ load weights from .weights file """
    buf = np.fromfile(wt_file, dtype=np.float32)
    start = 4   
    for i in range(13):
        bn   = model.darknet.main1[i].bn
        conv = model.darknet.main1[i].conv
        start = load_conv_bn(buf, start, conv, bn)
    for i in range(1, 6):
        bn   = model.darknet.main2[i].bn
        conv = model.darknet.main2[i].conv
        start = load_conv_bn(buf, start, conv, bn)
    for i in range(2):
        start = load_conv_bn(buf, start, model.conv1[i].conv, model.conv1[i].bn)
 
    start = load_conv_bn(buf, start, model.conv2[0].conv, model.conv2[0].bn)
    start = load_conv_bn(buf, start, model.conv[0].conv, model.conv[0].bn)
    start = load_conv(buf, start, model.conv[1])


def load_model(weights, device):
    """ load model and it's weights """
    model = YoloV2Net()
    if weights:
        load_weights(model, weights)
        #model.load_state_dict(torch.load(weights))
    return model.to(device)


def load_Qmodel(model, weights, qconfig=None, fuse_modules=False):
    if qconfig:
        model.qconfig = qconfig
        torch.quantization.prepare(model, inplace=True)
        # 執行 calibration 過程後
        torch.quantization.convert(model, inplace=True)
    
    model.load_state_dict(torch.load(weights, map_location='cpu'))
    model.eval()
    return model

if __name__ == "__main__":
    # Example usage
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model('weights/yolov2.weights', device)
    print(model)
    # Save the model to a file
    torch.save(model.state_dict(), "./yolov2_from_darknet_RELU.pth")
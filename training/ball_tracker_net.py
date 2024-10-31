import cv2
import torch.nn as nn
import numpy as np
import torch
import time

from torch.optim.lr_scheduler import ReduceLROnPlateau

import matplotlib.pyplot as plt
from torch.optim.adadelta import Adadelta

import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, pad, bias=True, bn=True):
        super().__init__()
        if bn:
            self.block = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, padding=pad, bias=bias),
                nn.ReLU(),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.block = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, padding=pad, bias=bias),
                nn.ReLU()
            )

    def forward(self, x):
        return self.block(x)


class BallTrackerNet(nn.Module):
    """
    Deep network for ball detection
    """
    def __init__(self, out_channels=256, bn=True):
        super().__init__()
        self.out_channels = out_channels

        # Encoder layers
        layer_1 = ConvBlock(in_channels=9, out_channels=64, kernel_size=3, pad=1, bias=True, bn=bn)
        layer_2 = ConvBlock(in_channels=64, out_channels=64, kernel_size=3, pad=1, bias=True, bn=bn)
        layer_3 = nn.MaxPool2d(kernel_size=2, stride=2)
        layer_4 = ConvBlock(in_channels=64, out_channels=128, kernel_size=3, pad=1, bias=True, bn=bn)
        layer_5 = ConvBlock(in_channels=128, out_channels=128, kernel_size=3, pad=1, bias=True, bn=bn)
        layer_6 = nn.MaxPool2d(kernel_size=2, stride=2)
        layer_7 = ConvBlock(in_channels=128, out_channels=256, kernel_size=3, pad=1, bias=True, bn=bn)
        layer_8 = ConvBlock(in_channels=256, out_channels=256, kernel_size=3, pad=1, bias=True, bn=bn)
        layer_9 = ConvBlock(in_channels=256, out_channels=256, kernel_size=3, pad=1, bias=True, bn=bn)
        layer_10 = nn.MaxPool2d(kernel_size=2, stride=2)
        layer_11 = ConvBlock(in_channels=256, out_channels=512, kernel_size=3, pad=1, bias=True, bn=bn)
        layer_12 = ConvBlock(in_channels=512, out_channels=512, kernel_size=3, pad=1, bias=True, bn=bn)
        layer_13 = ConvBlock(in_channels=512, out_channels=512, kernel_size=3, pad=1, bias=True, bn=bn)

        self.encoder = nn.Sequential(layer_1, layer_2, layer_3, layer_4, layer_5, layer_6, layer_7, layer_8, layer_9,
                                     layer_10, layer_11, layer_12, layer_13)

        # Decoder layers
        layer_14 = nn.Upsample(scale_factor=2)
        layer_15 = ConvBlock(in_channels=512, out_channels=256, kernel_size=3, pad=1, bias=True, bn=bn)
        layer_16 = ConvBlock(in_channels=256, out_channels=256, kernel_size=3, pad=1, bias=True, bn=bn)
        layer_17 = ConvBlock(in_channels=256, out_channels=256, kernel_size=3, pad=1, bias=True, bn=bn)
        layer_18 = nn.Upsample(scale_factor=2)
        layer_19 = ConvBlock(in_channels=256, out_channels=128, kernel_size=3, pad=1, bias=True, bn=bn)
        layer_20 = ConvBlock(in_channels=128, out_channels=128, kernel_size=3, pad=1, bias=True, bn=bn)
        layer_21 = nn.Upsample(scale_factor=2)
        layer_22 = ConvBlock(in_channels=128, out_channels=64, kernel_size=3, pad=1, bias=True, bn=bn)
        layer_23 = ConvBlock(in_channels=64, out_channels=64, kernel_size=3, pad=1, bias=True, bn=bn)
        layer_24 = ConvBlock(in_channels=64, out_channels=self.out_channels, kernel_size=3, pad=1, bias=True, bn=bn)

        self.decoder = nn.Sequential(layer_14, layer_15, layer_16, layer_17, layer_18, layer_19, layer_20, layer_21,
                                     layer_22, layer_23, layer_24)

        self.softmax = nn.Softmax(dim=1)
        self._init_weights()

    def forward(self, x, testing=False):
        batch_size = x.size(0)
        features = self.encoder(x)
        scores_map = self.decoder(features)
        output = scores_map.reshape(batch_size, self.out_channels, -1)
        # output = output.permute(0, 2, 1)
        if testing:
            output = self.softmax(output)
        return output

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.uniform_(module.weight, -0.05, 0.05)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def inference(self, frames: torch.Tensor):
        self.eval()
        with torch.no_grad():
            if len(frames.shape) == 3:
                frames = frames.unsqueeze(0)
            if next(self.parameters()).is_cuda:
                frames.cuda()
            # Forward pass
            output = self(frames, True)
            output = output.argmax(dim=1).detach().cpu().numpy()
            if self.out_channels == 2:
                output *= 255
            x, y = self.get_center_ball(output)
        return x, y

    def get_center_ball(self, output):
        """
        Detect the center of the ball using Hough circle transform
        :param output: output of the network
        :return: indices of the ball`s center
        """
        output = output.reshape((360, 640))

        # cv2 image must be numpy.uint8, convert numpy.int64 to numpy.uint8
        output = output.astype(np.uint8)

        # reshape the image size as original input image
        heatmap = cv2.resize(output, (640, 360))

        # heatmap is converted into a binary image by threshold method.
        ret, heatmap = cv2.threshold(heatmap, 127, 255, cv2.THRESH_BINARY)

        # find the circle in image with 2<=radius<=7
        circles = cv2.HoughCircles(heatmap, cv2.HOUGH_GRADIENT, dp=1, minDist=1, param1=50, param2=2, minRadius=2,
                                   maxRadius=7)
        # check if there have any tennis be detected
        if circles is not None:
            # if only one tennis be detected
            if len(circles) == 1:
                x = int(circles[0][0][0])
                y = int(circles[0][0][1])

                return x, y
        return None, None


def accuracy(y_pred, y_true):
    """
    Calculate accuracy of prediction
    """
    # Number of correct predictions
    correct = (y_pred == y_true).sum()
    # Predictions accuracy
    acc = correct / (len(y_pred[0]) * y_pred.shape[0]) * 100
    # Accuracy of non zero pixels predictions
    non_zero = (y_true > 0).sum()
    non_zero_correct = (y_pred[y_true > 0] == y_true[y_true > 0]).sum()
    if non_zero == 0:
        if non_zero_correct == 0:
            non_zero_acc = 100.0
        else:
            non_zero_acc = 0.0
    else:

        non_zero_acc = non_zero_correct / non_zero * 100
    return acc, non_zero_acc, non_zero_correct


def show_result(inputs, labels, outputs):
    """
    Display network`s predictions results
    """
    num_classes = outputs.size(1)
    outputs = outputs.argmax(dim=1).detach().cuda().numpy()
    if num_classes == 2:
        outputs *= 255
    mask = outputs[0].reshape((360, 640))
    fig, ax = plt.subplots(1, 2, figsize=(20, 1 * 5))
    ax[0].imshow(inputs[0, :3, :, ].detach().cuda().numpy().transpose((1, 2, 0)))
    ax[0].set_title('Image')
    ax[1].imshow(labels[0].detach().cuda().numpy().reshape((360, 640)), cmap='gray')
    ax[1].set_title('gt')
    plt.show()
    plt.figure()
    plt.imshow(mask, cmap='gray')
    plt.title('Pred')
    plt.show()


def get_center_ball_dist(output, x_true, y_true, num_classes=256):
    """
    Calculate distance of predicted center from the real center
    Success if distance is less than 5 pixels, fail otherwise
    """
    max_dist = 5
    success, fail = 0, 0
    dists = []
    Rx = 640 / 1280
    Ry = 360 / 720

    for i in range(len(x_true)):
        x, y = -1, -1
        # Reshape output
        cur_output = output[i].reshape((360, 640))

        # cv2 image must be numpy.uint8, convert numpy.int64 to numpy.uint8
        cur_output = cur_output.astype(np.uint8)

        # reshape the image size as original input image
        heatmap = cv2.resize(cur_output, (640, 360))

        # heatmap is converted into a binary image by threshold method.
        if num_classes == 256:
            ret, heatmap = cv2.threshold(heatmap, 127, 255, cv2.THRESH_BINARY)
        else:
            heatmap *= 255

        # find the circle in image with 2<=radius<=7
        circles = cv2.HoughCircles(heatmap, cv2.HOUGH_GRADIENT, dp=1, minDist=1, param1=50, param2=2, minRadius=2,
                                   maxRadius=7)
        # check if there have any tennis be detected
        if circles is not None:
            # if only one tennis be detected
            if len(circles) == 1:

                x = int(circles[0][0][0])
                y = int(circles[0][0][1])

        if x_true[i] < 0:
            if x < 0:
                success += 1
            else:
                fail += 1
            dists.append(-2)
        else:
            if x < 0:
                fail += 1
                dists.append(-1)
            else:
                dist = np.linalg.norm(((x_true[i] * Rx) - x, (y_true[i] * Ry) - y))
                dists.append(dist)
                if dist < max_dist:
                    success += 1
                else:
                    fail += 1

    return dists, success, fail





if __name__ == "__main__":
    '''state = torch.load("saved states/tracknet_weights_lr_1.0_epochs_125_256_classes.pth")
    plot_graph(state['train_loss'][5:], state['valid_loss'][5:], 'loss', '../report/tracknet_losses_150_epochs_256.png')
    plot_graph(state['train_acc'], state['valid_acc'], 'acc', '../report/tracknet_acc_150_epochs_256.png')
    plot_graph(np.array(state['train_success']) * 100 / (np.array(state['train_success']) + np.array(state['train_fail'])),
               np.array(state['valid_success']) * 100 / (np.array(state['valid_success']) + np.array(state['valid_fail'])), 'success acc', '../report/tracknet_success_acc_150_epochs_256.png')
    '''
    start = time.time()
    for lr in [0.05]:
        s = time.time()
        print(f'Start training with LR = {lr}')
        print(f'End training with LR = {lr}, Time = {time.time() - s}')
    print(f'Finished all runs, Time = {time.time() - start}')

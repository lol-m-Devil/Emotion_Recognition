import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from tqdm import tqdm



def get_inplanes():
    return [64, 128, 256, 512]

def conv3x3x3(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes,
                     out_planes,
                     kernel_size=3,
                     stride=stride,
                     padding=1,
                     bias=False)


def conv1x1x1(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes,
                     out_planes,
                     kernel_size=1,
                     stride=stride,
                     bias=False)

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()

        self.conv1 = conv1x1x1(in_planes, planes)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = conv3x3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = conv1x1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self,
                 block,
                 layers,
                 block_inplanes,
                 conv1_t_size=7,
                 conv1_t_stride=1,
                 no_max_pool = False,
                 shortcut_type='B',
                 widen_factor=1.0,
                 n_classes=400):
        super().__init__()

        block_inplanes = [int(x * widen_factor) for x in block_inplanes]

        self.in_planes = block_inplanes[0]
        self.no_max_pool = no_max_pool

        self.conv1 = nn.Conv3d(3,
                               self.in_planes,
                               kernel_size=(conv1_t_size, 7, 7),
                               stride=(conv1_t_stride, 2, 2),
                               padding=(conv1_t_size // 2, 3, 3),
                               bias=False)
        self.bn1 = nn.BatchNorm3d(self.in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, block_inplanes[0], layers[0],
                                       shortcut_type)
        self.layer2 = self._make_layer(block,
                                       block_inplanes[1],
                                       layers[1],
                                       shortcut_type,
                                       stride=2)
        self.layer3 = self._make_layer(block,
                                       block_inplanes[2],
                                       layers[2],
                                       shortcut_type,
                                       stride=2)
        self.layer4 = self._make_layer(block,
                                       block_inplanes[3],
                                       layers[3],
                                       shortcut_type,
                                       stride=2)

        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(block_inplanes[3] * block.expansion, n_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _downsample_basic_block(self, x, planes, stride):
        out = F.avg_pool3d(x, kernel_size=1, stride=stride)
        zero_pads = torch.zeros(out.size(0), planes - out.size(1), out.size(2),
                                out.size(3), out.size(4))
        if isinstance(out.data, torch.cuda.FloatTensor):
            zero_pads = zero_pads.cuda()

        out = torch.cat([out.data, zero_pads], dim=1)

        return out

    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1):
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(self._downsample_basic_block,
                                     planes=planes * block.expansion,
                                     stride=stride)
            else:
                downsample = nn.Sequential(
                    conv1x1x1(self.in_planes, planes * block.expansion, stride),
                    nn.BatchNorm3d(planes * block.expansion))

        layers = []
        layers.append(
            block(in_planes=self.in_planes,
                  planes=planes,
                  stride=stride,
                  downsample=downsample))
        self.in_planes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_planes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if not self.no_max_pool:
            x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x


class video_preprocessor():
    def __init__(self, video_folder, imageSize) -> None:
        self.model_path = 'resnet_101_kinetics.pth'
        self.video_folder = video_folder
        self.image_size = imageSize
        self.labels = []
        self.output = self.process()
        self.output = self.output.squeeze(3)
        self.output = self.output.reshape(self.output.shape[0], self.output.shape[1], self.output.shape[2], -1)
    
    @staticmethod
    def processVideo(videoPath, imageSize, num_snippets = 6, frames_per_snippet = 16, augmentation = True):
        cap = cv2.VideoCapture(videoPath)

        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps

        # Initialize an array to store keyframes
        snippets = []
        for i in range(num_snippets):
            # Calculate the start and end time for the snippet
            start_time = duration * (i / num_snippets)
            end_time = duration * ((i + 1) / num_snippets)

            # Uniformly sample time points within the snippet interval
            keyframe_time = np.linspace(start_time, end_time, frames_per_snippet)

            keyframes = []
            for j in range(frames_per_snippet):
                # Read the frame at the selected time
                frame_index = min(int(keyframe_time[j]*fps), total_frames - 2)
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                _, frame = cap.read()
                # Perform data augmentation if specified
                if augmentation:
                    # Apply random cropping
                    h, w, _ = frame.shape
                    crop_start_x = np.random.randint(0, w // 4)
                    crop_start_y = np.random.randint(0, h // 4)
                    frame = frame[crop_start_y:crop_start_y + 3 * h // 4, crop_start_x:crop_start_x + 3 * w // 4, :]

                    # Apply horizontal flipping
                    if np.random.rand() > 0.5:
                        frame = cv2.flip(frame, 1)

                    # Adjust brightness
                    alpha = 1.0 + np.random.uniform(-0.2, 0.2)
                    frame = np.clip(alpha * frame, 0, 255).astype(np.uint8)

                # Append the keyframe to the list
                frame = cv2.resize(frame, imageSize)
                frameTensor = torch.tensor(frame, dtype=torch.float32).view(3, *imageSize)
                keyframes.append(frameTensor)
            snippets.append(torch.stack(keyframes))

        # Release the video capture object
        cap.release()
        return torch.stack(snippets)
    
    @staticmethod
    def evaluateResnet3D(ResNet101, snippets):
        output = ResNet101(snippets)
        return output
    
    def process(self):
        checkpoint = torch.load(self.model_path, map_location=torch.device('cpu'))
        ResNet101 = ResNet(Bottleneck, [3, 4, 23, 3], get_inplanes())
        if 'module.' in list(checkpoint['state_dict'].keys())[0]:
            # Remove the 'module.' prefix from keys
            new_state_dict = {k.replace('module.', ''): v for k, v in checkpoint['state_dict'].items()}
            # Update the model's state dictionary
            ResNet101.load_state_dict(new_state_dict, strict = False)
        else:
            # If the 'module.' prefix is not present, load the state dictionary directly
            ResNet101.load_state_dict(checkpoint['state_dict'])

        folderPath = self.video_folder
        imageSize = self.image_size
        tensorList = []
        for filename in tqdm(os.listdir(folderPath), desc="Video Processing in Resnet101-3D", ncols=100):
            videoPath = os.path.join(folderPath, filename)
            label = filename.split('-')[2]
            self.labels.append(int(label))
            snippets = video_preprocessor.processVideo(videoPath, imageSize)
            snippets = snippets.transpose(1, 2)
            output = video_preprocessor.evaluateResnet3D(ResNet101, snippets)
            tensorList.append(output)
        
        return torch.stack(tensorList)

    



        
        
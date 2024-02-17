from moviepy.editor import VideoFileClip
import os
import sys

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

from typing import Any, Callable, List, Optional, Type, Union

import torch
import torch.nn as nn
from torch import Tensor

import shutil
import torchvision.transforms as transforms
from PIL import Image

from tqdm import tqdm

def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class SpatialAveragePooling(nn.Module):
    def __init__(self):
        super(SpatialAveragePooling, self).__init__()

    def forward(self, x):
        #dimension of x --> N x m x s
        pooled = torch.mean(x, dim=(2,3))
        return pooled
    
class ResNet(nn.Module):
    def __init__(
        self,
        block: Type[BasicBlock],
        layers: List[int],
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                f"or a 3-element tuple, got {replace_stride_with_dilation}"
            )
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, BasicBlock) and m.bn2.weight is not None:
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(
        self,
        block: Type[Union[BasicBlock]],
        planes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False,
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        x = self.conv1(x)
       
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
              

        x = self.layer1(x)
        
        x = self.layer2(x)
        
        x = self.layer3(x)
        
        x = self.layer4(x)
    
        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


def _resnet(
    block: Type[BasicBlock],
    layers: List[int],
    **kwargs: Any,
) -> ResNet:
    
    model = ResNet(block, layers, **kwargs)

    return model

class audio_preprocesser:
    def __init__(self, video_folder) -> None:
        self.video_folder = video_folder
        self.audio_folder = "Extracted_Audio"
        self.spectograms = "Extracted_Spectograms"
        self.output_values = []
        self.extract_audio_from_folder()
        
        model_path = 'resnet18-f37072fd.pth'
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
        self.Resnet18 = _resnet(BasicBlock, [2, 2, 2, 2])
        self.Resnet18.load_state_dict(checkpoint)
        self.pooling_layer = SpatialAveragePooling()
        self.save_normalized_spectrogram_images_from_folder()
        self.output_values = torch.stack(self.output_values)
        if os.path.exists(self.audio_folder):
            # Delete the folder and all its contents recursively
            shutil.rmtree(self.audio_folder)
        if os.path.exists(self.spectograms):
            # Delete the folder and all its contents recursively
            shutil.rmtree(self.spectograms)
    
    @staticmethod
    def extract_audio(input_video_path, output_audio_path):
        video_clip = VideoFileClip(input_video_path)
        audio_clip = video_clip.audio

        original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')
    
        audio_clip.write_audiofile(output_audio_path, codec='pcm_s16le', fps=audio_clip.fps)

        sys.stdout = original_stdout
    
        video_clip.close()
    
        
    def extract_audio_from_folder(self):
        # Create the output folder if it doesn't exist
        os.makedirs(self.audio_folder, exist_ok=True)

        # Process each video file in the input folder
        for video_file in tqdm(os.listdir(self.video_folder), desc="Audio Extraction", ncols=100):
            if video_file.endswith('.mp4'):
                input_video_path = os.path.join(self.video_folder, video_file)
                output_audio_path = os.path.join(self.audio_folder, f'{os.path.splitext(video_file)[0]}.wav')

                audio_preprocesser.extract_audio(input_video_path, output_audio_path)
    
    @staticmethod
    def generate_spectrogram(audio_file, fft_size=256, hop_size=10, window_size=32, num_parts=6):
        # Load audio file
        y, sr = librosa.load(audio_file, sr=None, duration=4)

        # Calculate the required padding for the spectrogram
        n_fft = fft_size
        hop_length = int(sr * hop_size / 1000)  # Convert hop_size from ms to samples
        win_length = int(sr * window_size / 1000)  # Convert window_size from ms to samples

        # Adjust the n_fft to be at least the length of the signal
        n_fft = max(n_fft, len(y))

        # Compute spectrogram
        spectrogram = librosa.feature.melspectrogram(
            y=y,
            sr=sr,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window="hamming",
            n_mels=256  # Number of frequency components
        )

        # Convert to decibels
        spectrogram_db = librosa.power_to_db(spectrogram, ref=np.max)

        # Split spectrogram into N shorter parts
        part_size = spectrogram_db.shape[1] // num_parts
        spectrogram_parts = [spectrogram_db[:, i * part_size:(i + 1) * part_size] for i in range(num_parts)]

        return spectrogram_parts

    @staticmethod
    def normalize_sequences(sequences):
        # Flatten the sequences to compute mean and variance
        flat_sequences = np.concatenate(sequences, axis=1)

        # Compute mean and variance
        mean = np.mean(flat_sequences, axis=1, keepdims=True)
        std = np.std(flat_sequences, axis=1, keepdims=True)

        # Normalize sequences
        normalized_sequences = [(seq - mean) / std for seq in sequences]

        return normalized_sequences
        
    def save_normalized_spectrogram_images(self, audio_file_path, audio_file, normalized_parts):
        os.makedirs(self.spectograms, exist_ok=True)
        _, sr = librosa.load(audio_file_path, sr=None, duration=4)
        hop_length = int(sr * 10 / 1000)
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x[:3, :, :]),  # Remove alpha channel if present
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        All_audio_outputs = []
        for i, part in enumerate(normalized_parts):
            # Plot the normalized spectrogram without labels
            plt.figure(figsize=(6, 4))
            librosa.display.specshow(part, sr=sr, hop_length=hop_length, x_axis=None, y_axis=None)
            plt.axis('off')

            # Save the image
            image_path = os.path.join(self.spectograms, f'{os.path.splitext(audio_file)[0]}-0{i+1}.png')
            plt.savefig(image_path, bbox_inches='tight', pad_inches=0)
            plt.close()
            image = Image.open(image_path)
            input_image = transform(image)
            input_image = input_image.unsqueeze(0)
            with torch.no_grad():
                audio_output = self.Resnet18(input_image)
                audio_output = self.pooling_layer(audio_output)
            All_audio_outputs.append(audio_output)
            if os.path.exists(image_path):
                # Delete the file
                os.remove(image_path)

        All_audio_outputs = torch.stack(All_audio_outputs)
        All_audio_outputs = All_audio_outputs.squeeze(1)
        return All_audio_outputs

    def save_normalized_spectrogram_images_from_folder(self):
        os.makedirs(self.spectograms, exist_ok=True)

        # Process each audio file in the input folder
        for audio_file in tqdm(os.listdir(self.audio_folder), desc="Audio Processing in Resnet18-2D", ncols=100):
            if audio_file.endswith('.wav'):
                audio_file_path = os.path.join(self.audio_folder, audio_file)
                spectrogram_parts = audio_preprocesser.generate_spectrogram(audio_file_path)
                normalized_parts = audio_preprocesser.normalize_sequences(spectrogram_parts)
                self.output_values.append(self.save_normalized_spectrogram_images(audio_file_path, audio_file, normalized_parts))










import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class video_processor():
    def process_video(videoPath, num_snippets = 6, frames_per_snippet = 16, augmentation = True):
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
                ret, frame = cap.read()

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
                keyframes.append(frame)
            snippets.append(keyframes)

        # Release the video capture object
        cap.release()

        # Convert the list of keyframes to a NumPy array
        snippets = np.array(snippets)

        return snippets

class Visual_SpatialMultiHeadAttention(nn.Module):
    def __init__(self, m: int, H: int) -> None:
        super().__init__()
        self.m = m
        self.H = H
        
        #making sure m is divisible by  H, otherwise problem!
        assert m % H == 0, "dimensions of model are divisible by number of heads"
        
        self.dim_head = m // H
        self.w_q = nn.Linear(m, m, bias = False)
        self.w_k = nn.Linear(m, m, bias = False)
        self.w_v = nn.Linear(m, m, bias = False)
        self.w_o = nn.Linear(m, m, bias = False)
        
    
    @staticmethod
    def attention(query, key, value):
        dim_head = query.shape[-1]
        # Just apply the formula from the paper
        # (batch, h, seq_len, d_k) --> (batch, h, seq_len, seq_len)
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(dim_head)
        attention_scores = attention_scores.softmax(dim=-1) # (batch, h, seq_len, seq_len) # Apply softmax
        
        # (batch, h, seq_len, seq_len) --> (batch, h, seq_len, d_k)
        # return attention scores which can be used for visualization
        return (attention_scores @ value), attention_scores

    def forward(self, q):
        query = self.w_q(q) # (batch, seq_len, d_model) --> (batch, seq_len, d_model)
        key = self.w_k(q) # (batch, seq_len, d_model) --> (batch, seq_len, d_model)
        value = self.w_v(q) # (batch, seq_len, d_model) --> (batch, seq_len, d_model)

        # (batch, seq_len, d_model) --> (batch, seq_len, h, d_k) --> (batch, h, seq_len, d_k)
        query = query.view(query.shape[0], query.shape[1], self.H, self.dim_head).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.H, self.dim_head).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.H, self.dim_head).transpose(1, 2)

        # Calculate attention
        x, self.attention_scores = Visual_SpatialMultiHeadAttention.attention(query, key, value)
        
        # Combine all the heads together
        # (batch, h, seq_len, d_k) --> (batch, seq_len, h, d_k) --> (batch, seq_len, d_model)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.H * self.dim_head)

        # Multiply by Wo
        # (batch, seq_len, d_model) --> (batch, seq_len, d_model)  
        return self.w_o(x)


class Visual_ChannelMultiHeadAttention(nn.Module):
    def __init__(self, s: int, H: int) -> None:
        super().__init__()
        self.s = s
        self.H = H

        #making sure s is divisible by  H, otherwise problem!
        assert s%H == 0, "dimensions of model are divisble by number of heads"

        self.dim_head = s // H
        self.w_q = nn.Linear(s, s, bias = False)
        self.w_k = nn.Linear(s, s, bias = False)
        self.w_v = nn.Linear(s, s, bias = False)
        self.w_o = nn.Linear(s, s, bias = False)

    @staticmethod
    def attention(query, key, value):
        dim_head = query.shape[-1]
        # Just apply the formula from the paper
        # (batch, h, seq_len, d_k) --> (batch, h, seq_len, seq_len)
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(dim_head)
        attention_scores = attention_scores.softmax(dim=-1) # (batch, h, seq_len, seq_len) # Apply softmax
        
        # (batch, h, seq_len, seq_len) --> (batch, h, seq_len, d_k)
        # return attention scores which can be used for visualization
        return (attention_scores @ value), attention_scores

    def forward(self, q):
        query = self.w_q(q) # (batch, seq_len, d_model) --> (batch, seq_len, d_model)
        key = self.w_k(q) # (batch, seq_len, d_model) --> (batch, seq_len, d_model)
        value = self.w_v(q) # (batch, seq_len, d_model) --> (batch, seq_len, d_model)

        # (batch, seq_len, d_model) --> (batch, seq_len, h, d_k) --> (batch, h, seq_len, d_k)
        query = query.view(query.shape[0], query.shape[1], self.H, self.dim_head).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.H, self.dim_head).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.H, self.dim_head).transpose(1, 2)

        # Calculate attention
        x, self.attention_scores = Visual_ChannelMultiHeadAttention.attention(query, key, value)
        
        # Combine all the heads together
        # (batch, h, seq_len, d_k) --> (batch, seq_len, h, d_k) --> (batch, seq_len, d_model)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.H * self.dim_head)

        # Multiply by Wo
        # (batch, seq_len, d_model) --> (batch, seq_len, d_model)  
        return self.w_o(x)


class Visual_TemporalMultiHeadAttention(nn.Module):
    def __init__(self, m: int, H: int) -> None:
        super().__init__()
        self.H = H
        self.m = m

        #making sure s is divisible by  H, otherwise problem!
        assert m%H == 0, "dimensions of model are divisble by number of heads"

        self.dim_head = m // H
        self.w_q = nn.Linear(m, m, bias = False)
        self.w_k = nn.Linear(m, m, bias = False)
        self.w_v = nn.Linear(m, m, bias = False)
        self.w_o = nn.Linear(m, self.dim_head*2, bias = False)

    @staticmethod
    def attention(query, key, value):
        dim_head = query.shape[-1]
        # Just apply the formula from the paper
        # (batch, h, seq_len, d_k) --> (batch, h, seq_len, seq_len)
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(dim_head)
        attention_scores = attention_scores.softmax(dim=-1) # (batch, h, seq_len, seq_len) # Apply softmax
        
        # (batch, h, seq_len, seq_len) --> (batch, h, seq_len, d_k)
        # return attention scores which can be used for visualization
        return (attention_scores @ value), attention_scores

    def forward(self, q):
        query = self.w_q(q) # (batch, seq_len, d_model) --> (batch, seq_len, d_model)
        key = self.w_k(q) # (batch, seq_len, d_model) --> (batch, seq_len, d_model)
        value = self.w_v(q) # (batch, seq_len, d_model) --> (batch, seq_len, d_model)

        # (batch, seq_len, d_model) --> (batch, seq_len, h, d_k) --> (batch, h, seq_len, d_k)
        query = query.view(query.shape[0], query.shape[1], self.H, self.dim_head).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.H, self.dim_head).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.H, self.dim_head).transpose(1, 2)

        # Calculate attention
        x, self.attention_scores = Visual_TemporalMultiHeadAttention.attention(query, key, value)
        
        # Combine all the heads together
        # (batch, h, seq_len, d_k) --> (batch, seq_len, h, d_k) --> (batch, seq_len, d_model)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.H * self.dim_head)

        # Multiply by Wo
        # (batch, seq_len, d_model) --> (batch, seq_len, d_model)  
        return self.w_o(x)
        

class SpatialAveragePooling(nn.Module):
    def __init__(self):
        super(SpatialAveragePooling, self).__init__()

    def forward(self, x):
        #dimension of x --> N x m x s
        pooled = torch.mean(x, dim=2)
        return pooled

        
        
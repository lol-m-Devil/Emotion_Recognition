import torch
import torch.nn as nn
import torch.nn.functional as F
import math

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

class Audio_TemporalMultiHeadAttention(nn.Module):
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
        x, self.attention_scores = Audio_TemporalMultiHeadAttention.attention(query, key, value)
        
        # Combine all the heads together
        # (batch, h, seq_len, d_k) --> (batch, seq_len, h, d_k) --> (batch, seq_len, d_model)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.H * self.dim_head)

        # Multiply by Wo
        # (batch, seq_len, d_model) --> (batch, seq_len, d_model)  
        return self.w_o(x)
        
class AV_CrossAttention(nn.Module):
    def __init__(self, m: int) -> None:
        super().__init__()
        self.m = m
        self.w = nn.Linear(m, m, bias = False)

    def forward(self, audio_data, video_data):
        Corr =  self.w(audio_data) @ video_data.transpose(0,1) # N*m * m*m * m*N -> N*N
        
        w_audio = F.softmax(Corr, dim =0)
        w_video = F.softmax(Corr.transpose(0,1), dim = 0)
        Dvideo = w_video @ video_data
        Daudio = w_audio @ audio_data
        
        DCorrVideo = torch.tanh(Dvideo + video_data)
        DCorrAudio = torch.tanh(Daudio + audio_data)
       
        return torch.cat((DCorrVideo,DCorrAudio), dim = 1)
        
               
        
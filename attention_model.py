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
        
        # (batch, snippets, h, s, m/h)  --> (batch, snippets, h, s, s)
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(dim_head)
        attention_scores = attention_scores.softmax(dim=-1) # (batch, snippets, h, s, s) # Apply softmax
        
        # (batch, snippets, h, s, s)  --> (batch, snippets, h, s, m/h)
        return (attention_scores @ value)

    def forward(self, q): # q -> (batch, snippets, m, s)
        q = q.transpose(2,3) #(batch, snippets, m, s) -> (batch, snippets, s, m)
        query = self.w_q(q) #(batch, snippets, s, m) -> (batch, snippets, s, m)
        key = self.w_k(q) #(batch, snippets, s, m) -> (batch, snippets, s, m)
        value = self.w_v(q) #(batch, snippets, s, m) -> (batch, snippets, s, m)

        #(batch, snippets, s, m) -> (batch, snippets, s, h, m/h) -> (batch, snippets, h, s, m/h) 
        query = query.view(query.shape[0], query.shape[1], query.shape[2], self.H, self.dim_head).transpose(2, 3)
        key = key.view(key.shape[0], key.shape[1], key.shape[2], self.H, self.dim_head).transpose(2, 3)
        value = value.view(value.shape[0], value.shape[1], value.shape[2], self.H, self.dim_head).transpose(2, 3)

        # Calculate attention
        x = Visual_SpatialMultiHeadAttention.attention(query, key, value)
        
        # Combine all the heads together
        # (batch, snippets, h, s, m/h) --> (batch, snippets, s, h, m/h) -> (batch, snippets, s, m)
        x = x.transpose(2, 3).contiguous().view(x.shape[0], x.shape[1], -1, self.H * self.dim_head)

        # Multiply by Wo
        #(batch, snippets, s, m) -> (batch, snippets, s, m)
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
        
        # (batch, snippets, h, s, m/h)  --> (batch, snippets, h, s, s)
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(dim_head)
        attention_scores = attention_scores.softmax(dim=-1) # (batch, snippets, h, s, s) # Apply softmax
        
        # (batch, snippets, h, s, s)  --> (batch, snippets, h, s, m/h)
        return (attention_scores @ value)
    
    def forward(self, q): # q -> (batch, snippets, s, m)
        q = q.transpose(2,3) #(batch, snippets, s, m) -> (batch, snippets, m, s)
        query = self.w_q(q) #(batch, snippets, m, s) -> (batch, snippets, m, s)
        key = self.w_k(q) #(batch, snippets, m, s) -> (batch, snippets, m, s)
        value = self.w_v(q) #(batch, snippets, m, s) -> (batch, snippets, m, s)

        #(batch, snippets, m, s) -> (batch, snippets, m, h, s/h) -> (batch, snippets, h, m, s/h) 
        query = query.view(query.shape[0], query.shape[1], query.shape[2], self.H, self.dim_head).transpose(2, 3)
        key = key.view(key.shape[0], key.shape[1], key.shape[2], self.H, self.dim_head).transpose(2, 3)
        value = value.view(value.shape[0], value.shape[1], value.shape[2], self.H, self.dim_head).transpose(2, 3)

        # Calculate attention
        x = Visual_SpatialMultiHeadAttention.attention(query, key, value)
        
        # Combine all the heads together
        # (batch, snippets, h, m, s/h) --> (batch, snippets, m, h, s/h) -> (batch, snippets, m, s)
        x = x.transpose(2, 3).contiguous().view(x.shape[0], x.shape[1], -1, self.H * self.dim_head)

        # Multiply by Wo
        #(batch, snippets, m, s) -> (batch, snippets, m, s)
        return self.w_o(x)

# class SpatialAveragePooling(nn.Module):
#     def __init__(self):
#         super(SpatialAveragePooling, self).__init__()

#     def forward(self, x):
#         #dimension of x --> Batch x N x m x s --> Batch x N x m x 1
#         pooled = torch.mean(x, dim=3)
#         pooled = pooled.unsqueeze(-1) #Batch x N x m x 1 --> Batch x N x m
#         return pooled
            
        
class Visual_TemporalMultiHeadAttention(nn.Module):
    def __init__(self, m: int, a_m: int, H: int) -> None:
        super().__init__()
        self.H = H
        self.m = m

        #making sure s is divisible by  H, otherwise problem!
        assert m%H == 0, "dimensions of model are divisble by number of heads"

        self.dim_head = m // H
        self.w_q = nn.Linear(m, m, bias = False)
        self.w_k = nn.Linear(m, m, bias = False)
        self.w_v = nn.Linear(m, m, bias = False)
        self.w_o = nn.Linear(m, a_m, bias = False) #To make sure last variable looks like 512

    @staticmethod
    def attention(query, key, value):
        dim_head = query.shape[-1]
        
        # (Batch x h x N x m/h) --> (Batch x h x N x N)
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(dim_head)
        attention_scores = attention_scores.softmax(dim=-1) # (Batch x h x N x N) # Apply softmax
        
        # (Batch x h x N x N)  --> (Batch x h x N x m/h)
        return (attention_scores @ value)

    def forward(self, q): #q -> (Batch x N x m)
        query = self.w_q(q) # (Batch x N x m) --> (Batch x N x m)
        key = self.w_k(q) # (Batch x N x m) --> (Batch x N x m)
        value = self.w_v(q) # (Batch x N x m) --> (Batch x N x m)

        # (Batch x N x m)--> (Batch x N x h x m/h) --> (Batch x h x N x m/h)
        query = query.view(query.shape[0], query.shape[1], self.H, self.dim_head).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.H, self.dim_head).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.H, self.dim_head).transpose(1, 2)

        # Calculate attention
        x = Visual_TemporalMultiHeadAttention.attention(query, key, value)
        
        # Combine all the heads together
        # (Batch x h x N x m/h) --> (Batch x N x h x m/h) -> (Batch x N x m)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.H * self.dim_head)

        # Multiply by Wo
        # (Batch x N x m) --> (Batch x N x a_m)  
        return self.w_o(x)      


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
        
        # (Batch x h x N x m/h) --> (Batch x h x N x N)
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(dim_head)
        attention_scores = attention_scores.softmax(dim=-1) # (Batch x h x N x N) # Apply softmax
        
        # (Batch x h x N x N)  --> (Batch x h x N x m/h)
        return (attention_scores @ value)
    
    def forward(self, q): #q -> (batch, N, m) 
        query = self.w_q(q) # (Batch x N x m) --> (Batch x N x m)
        key = self.w_k(q) # (Batch x N x m) --> (Batch x N x m)
        value = self.w_v(q) # (Batch x N x m) --> (Batch x N x m)

        # (Batch x N x m)--> (Batch x N x h x m/h) --> (Batch x h x N x m/h)
        query = query.view(query.shape[0], query.shape[1], self.H, self.dim_head).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.H, self.dim_head).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.H, self.dim_head).transpose(1, 2)

        # Calculate attention
        x = Audio_TemporalMultiHeadAttention.attention(query, key, value)
        
        # Combine all the heads together
        # (Batch x h x N x m/h) --> (Batch x N x h x m/h) -> (Batch x N x m)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.H * self.dim_head)

        # Multiply by Wo
        # (Batch x N x m) --> (Batch x N x m)  
        return self.w_o(x)
        
class AV_CrossAttention(nn.Module):
    def __init__(self, m: int) -> None:
        super().__init__()
        self.m = m
        self.w = nn.Linear(m, m, bias = False)

    def forward(self, audio_data, video_data): # both of form: (batch, N, m')
        # (batch, N, m') -> (batch, N, m') * (batch, m', N)-> (batch, N, N) 
        Corr =  self.w(audio_data) @ video_data.transpose(1,2) 
        
        w_audio = F.softmax(Corr, dim = 1) # (batch, N, N)
        w_video = F.softmax(Corr.transpose(1,2), dim = 1) # (batch, N, N)
        
        #(batch, N, N) * (batch, N, m')
        Dvideo = w_video @ video_data # (batch, N, m')
        Daudio = w_audio @ audio_data # (batch, N, m')
        
        #(batch, N, m') -> (batch, N, m')
        DCorrVideo = torch.tanh(Dvideo + video_data)
        DCorrAudio = torch.tanh(Daudio + audio_data)
       
        #(batch, N, 2*m')
        return torch.cat((DCorrVideo,DCorrAudio), dim = -1)
    
    
    
class Architecture(nn.Module):
    def __init__(self, N: int, a_m:int, v_m: int, s: int, H:int, out_classes: int):  #N =6, a_m = 512, v_m = 2048, s = 64, H = 8, out_classes= 8
        # self.pooling_layer = SpatialAveragePooling()
        super().__init__()
        self.Visual_spatial = Visual_SpatialMultiHeadAttention(v_m, H)
        self.Visual_channel = Visual_ChannelMultiHeadAttention(s, H)
        self.Visual_temporal = Visual_TemporalMultiHeadAttention(v_m, a_m, H)
        self.Audio_temporal = Audio_TemporalMultiHeadAttention(a_m, H)
        self.cross = AV_CrossAttention(a_m) 
        self.fc = nn.Linear(2*N*a_m, out_classes) # Number of features = 2*N*a_m
    
    def visual_task(self, vData): # (batch, N, m, s)
        vData = self.Visual_spatial(vData)
        vData = self.Visual_channel(vData)
        # vData = self.pooling_layer(vData)
        vData = torch.mean(vData, dim=3)
        vData = vData.squeeze(-1) 
        vData = self.Visual_temporal(vData)
        return vData
    
    def audio_task(self, aData): # (batch, N, m')
        aData = self.Audio_temporal(aData)
        return aData
    
    def forward(self, data): # data =  list of (vdata, adata, label)
        vData = []
        aData = []
        for d in data:
            v, a, _ = d
            vData.append(v)
            aData.append(a)
        vData = torch.stack(vData)
        aData = torch.stack(aData)    
        vData = self.visual_task(vData)
        aData = self.audio_task(aData)
         
        output = self.cross(aData, vData)
        output = output.view(output.shape[0], -1) # batch x f
        
        output = self.fc(output) # batch x 8
        output = F.softmax(output, dim = -1)
        output = torch.argmax(output, dim = -1) + 1
        output = output.tolist()
        
        return output
         
         
               
        
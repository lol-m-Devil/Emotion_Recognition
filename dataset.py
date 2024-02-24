import torch
import os
from torch.utils.data import Dataset
import random

class TripletDataset(Dataset):
    def __init__(self, config):
        self.data_path = os.path.join(config["tensor_data_path"], config["training_data_folder"])
        self.dict ={}
        idx = 0
        for folder in os.listdir(self.data_path):
            label_folder = os.path.join(self.data_path, folder)
            label = int(folder)
            for filename in os.listdir(label_folder):
                if filename.startswith("tensor_v_"):
                    i = int(filename.split("_")[2].split(".")[0])
                    tensorV = torch.load(os.path.join(label_folder, f"tensor_v_{i}.pt"))
                    tensorA = torch.load(os.path.join(label_folder, f"tensor_a_{i}.pt"))
                    self.dict[idx] = [tensorV, tensorA, label]
                    idx += 1
        
    def __len__(self):
        return len(self.dict)

    def __getitem__(self, idx):
        #anchor point : dict[idx]
        anchor = self.dict[idx]
        label = anchor[2]
        
        #positive point
        label_folder = os.path.join(self.data_path, f"{label}")  
        file_list = os.listdir(label_folder)
        random_file = random.choice(file_list)
        i = int(random_file.split("_")[2].split(".")[0])
        tensorV = torch.load(os.path.join(label_folder, f"tensor_v_{i}.pt"))
        tensorA = torch.load(os.path.join(label_folder, f"tensor_a_{i}.pt"))
        positive = [tensorV, tensorA, label] 
        
        #negative point
        list = [i for i in range(1,9) if i != label]
        neg_label = random.choice(list)
        neg_label_folder = os.path.join(self.data_path, f"{neg_label}")  
        neg_file_list = os.listdir(neg_label_folder)
        neg_random_file = random.choice(neg_file_list)
        neg_i = int(neg_random_file.split("_")[2].split(".")[0])
        neg_tensorV = torch.load(os.path.join(neg_label_folder, f"tensor_v_{neg_i}.pt"))
        neg_tensorA = torch.load(os.path.join(neg_label_folder, f"tensor_a_{neg_i}.pt")) 
        negative = [neg_tensorV, neg_tensorA, neg_label]

        return {
            "anchor_a": anchor[1],
            "anchor_v": anchor[0],
            "positive_a": positive[1],
            "positive_v": positive[0],
            "negative_a": negative[1],
            "negative_v": negative[0],
            "label_p": label,
            "label_n": negative[2]
        }        
        
class SingularDataset(Dataset):
    def __init__(self, config):
        self.data_path = os.path.join(config["tensor_data_path"], config["validation_data_folder"])
        self.dict ={}
        idx = 0
        for folder in os.listdir(self.data_path):
            label_folder = os.path.join(self.data_path, folder)
            label = int(folder)
            for filename in os.listdir(label_folder):
                if filename.startswith("tensor_v_"):
                    i = int(filename.split("_")[2].split(".")[0])
                    tensorV = torch.load(os.path.join(label_folder, f"tensor_v_{i}.pt"))
                    tensorA = torch.load(os.path.join(label_folder, f"tensor_a_{i}.pt"))
                    self.dict[idx] = [tensorV, tensorA, label]
                    idx += 1
        
    def __len__(self):
        return len(self.dict)

    def __getitem__(self, idx):
        #anchor point : dict[idx]
        anchor = self.dict[idx]
        
        return {
            "anchor_a": anchor[1],
            "anchor_v": anchor[0],
            "label": anchor[2]
        }
# def load_data(folderPath):
#     data = []
#     files = os.listdir(folderPath)
#     num_files = len(files)
#     for i in range(num_files//3):
#         tensor1 = torch.load(os.path.join(folderPath, f"tensor1_{i}.pt"))
#         tensor2 = torch.load(os.path.join(folderPath, f"tensor2_{i}.pt"))
        
#         number_file = os.path.join(folderPath, f"number_{i}.txt")
        
#         with open(number_file, "r") as file:
#             number = int(file.read().strip())
        
#         data.append((tensor1, tensor2, number))
    
#     return data



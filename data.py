import torch
import os

def load_data(folderPath):
    data = []
    files = os.listdir(folderPath)
    num_files = len(files)
    for i in range(num_files//3):
        tensor1 = torch.load(os.path.join(folderPath, f"tensor1_{i}.pt"))
        tensor2 = torch.load(os.path.join(folderPath, f"tensor2_{i}.pt"))
        
        number_file = os.path.join(folderPath, f"number_{i}.txt")
        
        with open(number_file, "r") as file:
            number = int(file.read().strip())
        
        data.append((tensor1, tensor2, number))
    
    return data



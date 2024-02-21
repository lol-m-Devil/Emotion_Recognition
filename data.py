import audio_model
import video_model
import torch
import os


def save_data(incomingFolderPath, imageSize, outgoingFolderPath): 
    audioProcessor = audio_model.audio_preprocesser(incomingFolderPath)
    videoProcessor = video_model.video_preprocessor(incomingFolderPath, imageSize)

    videoData = videoProcessor.output
    audioData = audioProcessor.output_values
    labels = videoProcessor.labels

    data = []
    for v,a,l in zip(videoData, audioData, labels):
        data.append((v,a,l))
    #v --> (6, 2048, 64)  #a --> (6, 512)
    os.makedirs(outgoingFolderPath, exist_ok=True)

    for i, (tensor1, tensor2, number) in enumerate(data):
        
        torch.save(tensor1, os.path.join(outgoingFolderPath, f"tensor1_{i}.pt"))
        torch.save(tensor2, os.path.join(outgoingFolderPath, f"tensor2_{i}.pt"))
        with open(os.path.join(outgoingFolderPath, f"number_{i}.txt"), "w") as file:
            file.write(str(number))
    return data
    
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



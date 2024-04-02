from pathlib import Path
import os

def get_config():
    return {
        "batch_size": 4,
        "momentum": 0.9,
        "num_epochs": 1,
        "lr": 10**-4,
        "N": 6,
        "a_m": 512,
        "v_m": 2048,
        "s": 64,
        "H": 8,
        "out_classes": 8,
        "train_test_split": 0.9,
        "model_folder": "weights",
        "model_basename": "EmoDet_",
        "experiment_name": "experiment_1",
        "preload": "latest",
        "input_data_path": "Actors", 
        "tensor_data_path": "tensorData",
        "training_data_folder": "training",
        "validation_data_folder": "validation",
        "resnet-18_path": "resnet/resnet18-f37072fd.pth",
        "resnet-101_path": "resnet/resnet_101_kinetics.pth",
        "image_size": (256, 256),
        "eps": 1e-9,
        "margin": 1.0,
    }
    
def get_weights(config, epoch: str):
    model_folder = config['model_folder']
    os.makedirs(model_folder, exists_ok = True)
    model_filename = f"{config['model_basename']}{epoch}.pt"
    return str(Path('.') / model_folder / model_filename)


def latest_weights(config):
    model_folder = config["model_folder"]
    os.makedirs(model_folder, exists_ok = True)
    model_filename = f"{config['model_basename']}*"
    weights_files = list(Path(model_folder).glob(model_filename))
    if len(weights_files) == 0:
        return None
    weights_files.sort()
    return str(weights_files[-1])

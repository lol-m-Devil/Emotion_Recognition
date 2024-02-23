from pathlib import Path

def get_config():
    return {
        "batch_size": 16,
        "momentum": 0.9,
        "num_epochs": 20,
        "lr": 10**-4,
        "N": 6,
        "a_m": 512,
        "v_m": 2048,
        "s": 64,
        "H": 8,
        "out_classes": 8,
        "train_test_split": 0.9,
        # "model_folder": "weights",
        # "model_basename": "tmodel_",
        "preload": "latest",
        "input_data_path": "Single_Actor_01", 
        "tensor_data_path": "tensorData",
        "resnet-18_path": "resnet18-f37072fd.pth",
        "resnet-101_path": "resnet_101_kinetics.pth",
        "image_size": (256, 256),
    }
    
def get_weights_file_path(config, epoch: str):
    model_folder = f"{config['datasource']}_{config['model_folder']}"
    model_filename = f"{config['model_basename']}{epoch}.pt"
    return str(Path('.') / model_folder / model_filename)


def latest_weights_file_path(config):
    model_folder = f"{config['datasource']}_{config['model_folder']}"
    model_filename = f"{config['model_basename']}*"
    weights_files = list(Path(model_folder).glob(model_filename))
    if len(weights_files) == 0:
        return None
    weights_files.sort()
    return str(weights_files[-1])

import data
import os
import random
import attention_model
import torch
import configuration


def get_full_data(config):
    data_preloaded = os.path.exists(config["tensor_data_path"]) and os.listdir(config["tensor_data_path"])
    #datapreloaded = True if the tensor_data_path exists and is non-empty
    if data_preloaded:
        return data.load_data(config["tensor_data_path"])
    else:
        data = data.save_data(config["video_data_path"], config["image_size"], config["tensor_data_path"])
        return data


def split_data_uniformly(config):
    data = get_full_data(config)
    grouped_data = {}
    for item in data:
        label = item[-1]  
        if label not in grouped_data:
            grouped_data[label] = []
        grouped_data[label].append(item)

    for label_data in grouped_data.values():
        random.shuffle(label_data)

    train_data = []
    validation_data = []
    
    train_counts = {label: int(len(label_data) * config["train_test_split"]) for label, label_data in grouped_data.items()}

    # Iterate over each group of data and distribute the samples into the two splits
    for label, label_data in grouped_data.items():
        train_data.extend(label_data[:train_counts[label]])
        validation_data.extend(label_data[train_counts[label]:])

    return train_data, validation_data
    
def get_model(config):
    model = attention_model.Architecture(config["N"], config["a_m"], config["v_m"], config["s"], config["H"], config["out_classes"])
    return model

def train_model(config):
    trainData, validationData = split_data_uniformly(config)
    model = get_model(config)
    optimizer = torch.optim.Adam(model.parameters(), lr = config['lr'], eps = 1e-9)

    initial_epoch = 0
    global_step = 0
    preload = config['preload']
    modelFilename = None
    if preload == 'latest':
        modelFilename = configuration.latest_weights(config)
    elif preload:
        modelFilename = configuration.get_weights(config, preload)
    
    if modelFilename:
        print(f'Preloaded the weights of the model {modelFilename}')
        state = torch.load(modelFilename)
        initial_epoch = state['epoch'] + 1
        global_step = state['global_step']
        optimizer.load_state_dict(state['optmizer_state_dictionary'])
        model.load_state_dict(state['model_state_dictionary'])
    else:
        print('No model to preload, starting from scratch')

    
    
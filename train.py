import os
import shutil
import random
import attention_model
import torch
import configuration
import torch.nn as nn

def split_data(config):
    data_dir = config["tensor_data_path"]
    train_data_folder = config["training_data_folder"]
    validation_data_folder = config["validation_data_folder"]
    train_split = config["train_test_split"]
    n_samples = len(os.listdir(data_dir)) // 3
    
    os.makedirs(os.path.join(data_dir, train_data_folder), exist_ok=True)
    os.makedirs(os.path.join(data_dir, validation_data_folder), exist_ok=True)

    sample_indices = [i for i in range(n_samples)]
    random.shuffle(sample_indices)

    num_train_samples = int(train_split * n_samples)
    
    train_indices = sample_indices[:num_train_samples]
    val_indices = sample_indices[num_train_samples:]

    # Move the data files to the appropriate directories
    for i in train_indices:
        shutil.move(os.path.join(data_dir, f'tensor_v_{i}.pt'), os.path.join(data_dir, train_data_folder))
        shutil.move(os.path.join(data_dir, f'tensor_a_{i}.pt'), os.path.join(data_dir, train_data_folder))
        shutil.move(os.path.join(data_dir, f'number_{i}.txt'), os.path.join(data_dir, train_data_folder))

    for i in val_indices:
        shutil.move(os.path.join(data_dir, f'tensor_v_{i}.pt'), os.path.join(data_dir, validation_data_folder))
        shutil.move(os.path.join(data_dir, f'tensor_a_{i}.pt'), os.path.join(data_dir, validation_data_folder))
        shutil.move(os.path.join(data_dir, f'number_{i}.txt'), os.path.join(data_dir, validation_data_folder))
        

def get_model(config):
    model = attention_model.Architecture(config["N"], config["a_m"], config["v_m"], config["s"], config["H"], config["out_classes"])
    return model

def train_model(config):
    
    split_data(config)
    
    #load the data using data loader
    #triplet loss
    #training loop
    #save the model weights and stuff
    #run the validation after each epoch
    #torch.device on cuda/cpu/gpu
    #implement a writer!
    
    
    model = get_model(config)
    optimizer = torch.optim.Adam(model.parameters(), lr = config['lr'], eps = config['eps'])

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

    loss_fn = nn.CrossEntropyLoss()
    
    
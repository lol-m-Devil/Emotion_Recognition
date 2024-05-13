import os
import shutil
import random
import attention_model
import torch
import configuration
import torch.nn as nn
from torch.utils.data import DataLoader
import dataset
import audio_model
import video_model
import torchmetrics
from tqdm import tqdm

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from torch.utils.tensorboard import SummaryWriter
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)



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
        
def classify_in_folders(config):
    training_path = os.path.join(config["tensor_data_path"], config["training_data_folder"])
    validation_path = os.path.join(config["tensor_data_path"], config["validation_data_folder"])
    
    os.makedirs(os.path.join(training_path, "1" ), exist_ok=True)
    os.makedirs(os.path.join(training_path, "2" ), exist_ok=True)
    os.makedirs(os.path.join(training_path, "3" ), exist_ok=True)
    os.makedirs(os.path.join(training_path, "4" ), exist_ok=True)
    os.makedirs(os.path.join(training_path, "5" ), exist_ok=True)
    os.makedirs(os.path.join(training_path, "6" ), exist_ok=True)
    os.makedirs(os.path.join(training_path, "7" ), exist_ok=True)
    os.makedirs(os.path.join(training_path, "8" ), exist_ok=True)
    
    os.makedirs(os.path.join(validation_path, "1" ), exist_ok=True)
    os.makedirs(os.path.join(validation_path, "2" ), exist_ok=True)
    os.makedirs(os.path.join(validation_path, "3" ), exist_ok=True)
    os.makedirs(os.path.join(validation_path, "4" ), exist_ok=True)
    os.makedirs(os.path.join(validation_path, "5" ), exist_ok=True)
    os.makedirs(os.path.join(validation_path, "6" ), exist_ok=True)
    os.makedirs(os.path.join(validation_path, "7" ), exist_ok=True)
    os.makedirs(os.path.join(validation_path, "8" ), exist_ok=True)
    
    for filename in os.listdir(training_path):
        if filename.endswith(".txt") and filename.startswith("number_"):
            file_path = os.path.join(training_path, filename)
            i = int(filename.split("_")[1].split(".")[0])
            with open(file_path, 'r') as file:
                label = file.read()
            shutil.move(os.path.join(training_path, f'tensor_v_{i}.pt'), os.path.join(training_path, label))
            shutil.move(os.path.join(training_path, f'tensor_a_{i}.pt'), os.path.join(training_path, label))
        
    for filename in os.listdir(validation_path):
        if filename.endswith(".txt") and filename.startswith("number_"):
            file_path = os.path.join(validation_path, filename)
            i = int(filename.split("_")[1].split(".")[0])
            with open(file_path, 'r') as file:
                label = file.read()
            shutil.move(os.path.join(validation_path, f'tensor_v_{i}.pt'), os.path.join(validation_path, label))
            shutil.move(os.path.join(validation_path, f'tensor_a_{i}.pt'), os.path.join(validation_path, label))
    
    
    for filename in os.listdir(training_path):
        if filename.endswith(".txt"):
            file_path = os.path.join(training_path, filename)
            os.remove(file_path)        

    for filename in os.listdir(validation_path):
        if filename.endswith(".txt"):
            file_path = os.path.join(validation_path, filename)
            os.remove(file_path)


def validation(model, validation_ds, device, global_step, writer, config):
    model.eval()

    expected = []
    predicted = []
    with torch.no_grad():
        for batch in validation_ds:
            aData = batch["anchor_a"].to(device)
            vData = batch["anchor_v"].to(device)
            label = batch["label"]
            output = model(vData, aData)
            predicted_label = torch.argmax(output, dim = -1) + 1
            expected.append(label)
            predicted.append(predicted_label)
            
    if writer:
        metric = torchmetrics.classification.Accuracy(task = "multiclass", num_classes = config["out_classes"]).to(device)
        predicted_tensor = torch.tensor(predicted)
        expected_tensor = torch.tensor(expected)
        acc = metric(predicted_tensor, expected_tensor)
        writer.add_scalar('validation accuracy', acc, global_step)
        writer.flush()



def get_ds(config):
    folder_path = config["tensor_data_path"]
    if not os.path.exists(folder_path):
        _ = audio_model.audio_preprocesser(config)
        _ = video_model.video_preprocessor(config)
        split_data(config)
        classify_in_folders(config)
        
    training_ds = dataset.TripletDataset(config)
    validation_ds = dataset.SingularDataset(config)
    
    training_dataloader = DataLoader(training_ds, batch_size = config["batch_size"], shuffle = True)
    validation_dataloader = DataLoader(validation_ds, batch_size = 1, shuffle = True)
    
    return training_dataloader, validation_dataloader
            
def get_model(config):
    model = attention_model.Architecture(config["N"], config["a_m"], config["v_m"], config["s"], config["H"], config["out_classes"])
    return model

def train_model(config):
    
     # Define the device
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.has_mps or torch.backends.mps.is_available() else "cpu"
    print("Using device:", device)
    if (device == 'cuda'):
        print(f"Device name: {torch.cuda.get_device_name(device.index)}")
        print(f"Device memory: {torch.cuda.get_device_properties(device.index).total_memory / 1024 ** 3} GB")
    elif (device == 'mps'):
        print(f"Device name: <mps>")
    else:
        print("Device name: CPU")    
    device = torch.device(device)   
    
    
    training_dl, validation_dl = get_ds(config)   
    
    model = get_model(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr = config['lr'], eps = config['eps'])

    writer = SummaryWriter(config['experiment_name'])

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
        optimizer.load_state_dict(state['optimizer_state_dictionary'])
        model.load_state_dict(state['model_state_dictionary'])
    else:
        print('No model to preload, starting from scratch')

    loss_fn = nn.TripletMarginLoss(margin=config["margin"], p = 2).to(device)
    for epoch in range(initial_epoch, config["num_epochs"]):
        model.train()
        batch_iterator = tqdm(training_dl, desc = f"Processing Epoch: {epoch:02d}")
        for batch in batch_iterator:
            
            anchor_a = batch["anchor_a"].to(device)
            anchor_v = batch["anchor_v"].to(device)
            positive_a = batch["positive_a"].to(device)
            positive_v = batch["positive_v"].to(device)
            negative_a = batch["negative_a"].to(device)
            negative_v = batch["negative_v"].to(device)    
            
            anchor_output = model(anchor_v, anchor_a)
            positive_output = model(positive_v, positive_a)
            negative_output = model(negative_v, negative_a)
            print("Forward Completed")
            loss = loss_fn(anchor_output, positive_output, negative_output)    
            batch_iterator.set_postfix({"loss": f"{loss.item():6.3f}"})
            print(f"Loss Calculated{loss.item():6.3f}")
            
            # Log the loss
            writer.add_scalar('train loss', loss.item(), global_step)
            writer.flush()
            
            loss.backward()
            print("Backprop Completed")
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            print("Updation Step Completed")
            global_step += 1
        
        #save your model
        model_filename = configuration.get_weights(config, f"{epoch:02d}")
        torch.save({
            'epoch': epoch,
            'model_state_dictionary': model.state_dict(),
            'optimizer_state_dictionary': optimizer.state_dict(),
            'global_step': global_step
        }, model_filename)

        #run the validation and log the details using tensorboard!
        validation(model, validation_dl, device, epoch, writer, config)
    
    writer.close()
        
        

if __name__ == '__main__':
    config = configuration.get_config()
    train_model(config)        

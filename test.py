import train
import configuration
import os
from tqdm import tqdm

d = train.get_ds(configuration.get_config())
print(type(d))
print(len(d))
batch_iterator = tqdm(d, desc=f"Processing Epoch")
t = True
for batch in batch_iterator:
    
    # print(type(batch))
    # print(len(batch))
    # print(batch.keys())
    
    if t:
        for key in batch.keys():
            print(key)
            print(batch[key].shape)
        t = False
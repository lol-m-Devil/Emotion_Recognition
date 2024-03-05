#Emotion Recognition:
Worked on implementation of the paper titled: "Multimodal emotion recognition using cross modal audio-video fusion with attention and deep metric learning" by Bogdan Mocanu, Ruxandra Tapu, Titus Zaharia.

#File Formats:

1. Add a resnet folder containing two resnets - Resnet 18 and Resnet 101. 
   - These can be found at the following accessable link: https://drive.google.com/drive/folders/1YTcXOuctoVbX_PHnPoIqdsDGoE6FKHT5?usp=sharing
   - The folder should explicity be named "resnet" and files name should be same as in google drive. 
2. RAVDESS data download link: https://zenodo.org/records/1188976#.YFZuJ0j7SL8
   - Store all the videos in a folder named "Actors". 
   - The code has assumed each video file is named using the following convention: 
    - The filename consists of a 7-part numerical identifier (e.g., 02-01-06-01-02-01-12.mp4). These identifiers define the stimulus characteristics: 
    - Filename identifiers 
        Modality (01 = full-AV). # Only considered a full av file. 
        Vocal channel (01 = speech, 02 = song).
        Emotion (01 = neutral, 02 = calm, 03 = happy, 04 = sad, 05 = angry, 06 = fearful, 07 = disgust, 08 = surprised).
        Emotional intensity (01 = normal, 02 = strong). NOTE: There is no strong intensity for the 'neutral' emotion.
        Statement (01 = "Kids are talking by the door", 02 = "Dogs are sitting by the door").
        Repetition (01 = 1st repetition, 02 = 2nd repetition).
        Actor (01 to 24. Odd numbered actors are male, even numbered actors are female).
    - Filename example: 01-01-06-01-02-01-12.mp4 
        Audio-Video (01)
        Speech (01)
        Fearful (06)
        Normal intensity (01)
        Statement "dogs" (02)
        1st Repetition (01)
        12th Actor (12)
        Female, as the actor ID number is even.  
3. If any folder path changes or hyperparameter changes are required, refer to the configuration.py file
4. All jupyter notebooks are only for testing purpose and do not serve any other purpose (*.ipynb files)


# To run this code: Run the train.py file which will run the code and run all processes in order to train the model.

# Important Files:

1. video_model.py
    - contains architecture of 3D Resnet 101
    - Video_preprocessor class does the following:
        - Preprocess each video (create 6 snippets each of 16 frames)
        - processes it through loaded resent_101 to generate tensors and saves the tensors
2. audio_model.py
    - Implementation of Audio_preprocessor, Resnet2D-18, Spatial Average Pooling Layer
    - Audio_preprocessor class does the following:
        - Extracts audio from video
        - Loads the resnet 18 using pretrained model
        - For each audio file, 
            -Generates normalized spectogram. 
            -Passes it through loaded resnet to generate a tensor.
            -Tensor is passed through SpatialAvgPoolingLayer.
            -Saves this tensor to an output folder "tensorData" as "tensor_a_{i}.pt", where i is for ith file. 

3. attention_model.py
    - Implementation of multiple classes for Audio and Video Self attention, cross attention and an Architecture.
    - In order of use for videos,
        -Implemented Visual_SpatialMultiHeadAttention
        -Implemented Visual_ChannelMultiHeadAttention
        -Implemented Visual_TemporalMultiHeadAttention
    - For Audio, we implement Audio Temporal Attention Module.    
    - Finally, we use cross attention and pass this through a fully connected layer which returns softmax values as a list for number of output classes. 
    
4. dataset.py
    - for training, we need TripletDataset
    - for validation, we need SingularDataset
5. configuration.py
    - contains file paths and hyperparameter values
6. train.py
    - sets the device to 'GPU', if available.
    - splits data into training and validation.
    - gets model using get_model function.
    - uses traditional TripletMarginLoss.
    - saves model weights after each epoch.
    - Main file that trains the model. 

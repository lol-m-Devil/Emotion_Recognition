import os
import video_model

folderPath = 'Single_Actor_01'
imageSize = (224, 224)
videoProcessor = video_model.video_preprocessor(folderPath, imageSize)
print(videoProcessor.labels)
print(videoProcessor.output.shape)


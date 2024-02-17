import audio_model
import video_model


folderPath = 'Single_Actor_01'
imageSize = (224, 224) #set for safety only!


audioProcessor = audio_model.audio_preprocesser(folderPath)
videoProcessor = video_model.video_preprocessor(folderPath, imageSize)

videoData = videoProcessor.output
audioData = audioProcessor.output_values
labels = videoProcessor.labels

data = []

for v,a,l in zip(videoData, audioData, labels):
    data.append((v,a,l))

# print(len(data))
# print("Data 1")
# print(data[0][0].shape)
# print(data[0][1].shape)
# print(data[0][2])



import audio_model
import video_model
import attention_model


folderPath = 'Single_Actor_01'
imageSize = (256, 256) #set for safety only!


audioProcessor = audio_model.audio_preprocesser(folderPath)
videoProcessor = video_model.video_preprocessor(folderPath, imageSize)

videoData = videoProcessor.output
audioData = audioProcessor.output_values
labels = videoProcessor.labels

data = []

for v,a,l in zip(videoData, audioData, labels):
    data.append((v,a,l))

arch = attention_model.Architecture(6,512,2048,64,8,8)
output = arch(data)
print(output)





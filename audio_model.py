from moviepy.editor import VideoFileClip
import os

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np


class audio_preprocesser:
    def __init__(self, video_folder, audio_folder, spectogram_folder ) -> None:
        self.video_folder = video_folder
        self.audio_folder = audio_folder
        self.spectograms = spectogram_folder
    
    @staticmethod
    def extract_audio(input_video_path, output_audio_path):
        video_clip = VideoFileClip(input_video_path)
        audio_clip = video_clip.audio

        audio_clip.write_audiofile(output_audio_path, codec='pcm_s16le', fps=audio_clip.fps)

        video_clip.close()
    
        
    def extract_audio_from_folder(self):
        # Create the output folder if it doesn't exist
        os.makedirs(self.audio_folder, exist_ok=True)

        # Process each video file in the input folder
        for video_file in os.listdir(self.video_folder):
            if video_file.endswith('.mp4'):
                input_video_path = os.path.join(self.video_folder, video_file)
                output_audio_path = os.path.join(self.audio_folder, f'{os.path.splitext(video_file)[0]}.wav')

                self.extract_audio(input_video_path, output_audio_path)
    

    def generate_spectrogram(audio_file, fft_size=256, hop_size=10, window_size=32, num_parts=6):
        # Load audio file
        y, sr = librosa.load(audio_file, sr=None, duration=4)

        # Calculate the required padding for the spectrogram
        n_fft = fft_size
        hop_length = int(sr * hop_size / 1000)  # Convert hop_size from ms to samples
        win_length = int(sr * window_size / 1000)  # Convert window_size from ms to samples

        # Adjust the n_fft to be at least the length of the signal
        n_fft = max(n_fft, len(y))

        # Compute spectrogram
        spectrogram = librosa.feature.melspectrogram(
            y=y,
            sr=sr,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window="hamming",
            n_mels=256  # Number of frequency components
        )

        # Convert to decibels
        spectrogram_db = librosa.power_to_db(spectrogram, ref=np.max)

        # Split spectrogram into N shorter parts
        part_size = spectrogram_db.shape[1] // num_parts
        spectrogram_parts = [spectrogram_db[:, i * part_size:(i + 1) * part_size] for i in range(num_parts)]

        return spectrogram_parts

    def normalize_sequences(sequences):
        # Flatten the sequences to compute mean and variance
        flat_sequences = np.concatenate(sequences, axis=1)

        # Compute mean and variance
        mean = np.mean(flat_sequences, axis=1, keepdims=True)
        std = np.std(flat_sequences, axis=1, keepdims=True)

        # Normalize sequences
        normalized_sequences = [(seq - mean) / std for seq in sequences]

        return normalized_sequences
        
    def save_normalized_spectrogram_images(self, audio_file_path, audio_file, normalized_parts):
        os.makedirs(self.spectograms, exist_ok=True)
        y, sr = librosa.load(audio_file_path, sr=None, duration=4)
        hop_length = int(sr * 10 / 1000)
        
        for i, part in enumerate(normalized_parts):
            # Plot the normalized spectrogram without labels
            plt.figure(figsize=(6, 4))
            librosa.display.specshow(part, sr=sr, hop_length=hop_length, x_axis=None, y_axis=None)
            plt.axis('off')

            # Save the image
            image_path = os.path.join(self.spectograms, f'{os.path.splitext(audio_file)[0]}-0{i+1}.png')
            plt.savefig(image_path, bbox_inches='tight', pad_inches=0)
            plt.close()

    def save_normalized_spectrogram_images_from_folder(self):
        os.makedirs(self.spectograms, exist_ok=True)

        # Process each audio file in the input folder
        for audio_file in os.listdir(self.audio_folder):
            if audio_file.endswith('.wav'):
                audio_file_path = os.path.join(self.audio_folder, audio_file)
                spectrogram_parts = self.generate_spectrogram(audio_file_path)
                normalized_parts = self.normalize_sequences(spectrogram_parts)
                self.save_normalized_spectrogram_images(audio_file_path, audio_file, normalized_parts)
            

# Example usage
input_audio_folder = 'Single_Audio_01'
output_spectrogram_folder = 'Single_Spectogram_01'

save_normalized_spectrogram_images_from_folder(input_audio_folder, output_spectrogram_folder)


import os
import shutil
import numpy as np
import librosa
import soundfile as sf
import argparse

# Parameters
AUDIO_THRESHOLD = 0.02  # Define a threshold for what is considered audio (adjustable)
SILENCE_DURATION = 0.25  # Silence the first 0.25 seconds (adjustable)
BACKUP_FOLDER = "backup_audio"  # Folder to store original backups

def process_audio(file_path):
    try:
        # Load the audio file
        audio, sr = librosa.load(file_path, sr=None)

        # Calculate how many samples correspond to 0.25 seconds
        silence_samples = int(SILENCE_DURATION * sr)

        # Check for audio presence in the first 0.25 seconds
        if np.max(np.abs(audio[:silence_samples])) > AUDIO_THRESHOLD:
            # If audio is detected, silence that portion by setting the values to 0
            audio[:silence_samples] = 0

            # Backup the original file
            if not os.path.exists(BACKUP_FOLDER):
                os.makedirs(BACKUP_FOLDER)
            backup_path = os.path.join(BACKUP_FOLDER, os.path.basename(file_path))
            shutil.copyfile(file_path, backup_path)

            # Save the modified audio back to the original file
            sf.write(file_path, audio, sr)
            print(f"Processed and silenced first {SILENCE_DURATION} seconds of: {file_path}")
        else:
            print(f"No significant audio detected in the first {SILENCE_DURATION} seconds of: {file_path}")
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")

def scan_directory(directory):
    for filename in os.listdir(directory):
        if filename.endswith(".wav"):
            file_path = os.path.join(directory, filename)
            process_audio(file_path)

if __name__ == "__main__":
    # Argument parser to get directory from command line
    parser = argparse.ArgumentParser(description="Process .wav files to silence the first 0.25 seconds of audio if sound is detected.")
    parser.add_argument("directory", type=str, help="Path to the directory containing .wav files")

    # Parse arguments
    args = parser.parse_args()

    # Scan and process the directory provided via command line
    scan_directory(args.directory)

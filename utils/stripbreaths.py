<<<<<<< HEAD
import os
import argparse
from pydub import AudioSegment
import numpy as np

# Function to calculate RMS of an audio segment in dB
def rms_in_db(audio_segment):
    samples = np.array(audio_segment.get_array_of_samples())
    rms = np.sqrt(np.mean(samples**2))
    if rms > 0:
        return 20 * np.log10(rms)
    else:
        return -float('inf')

# Function to detect the start of speech, including silence and breath trimming
def detect_start_of_speech(audio, silence_threshold_db=-70, breath_threshold_db=-50, speech_threshold_db=-30, segment_length_ms=50):
    """
    Detects the start of speech by scanning through the audio, first detecting silence, then a breath, and finally speech.
    Args:
        audio (AudioSegment): The audio file to analyze.
        silence_threshold_db (float): Threshold for detecting silence in dB.
        breath_threshold_db (float): Threshold for detecting breath in dB.
        speech_threshold_db (float): Threshold for detecting speech in dB.
        segment_length_ms (int): Length of segments to analyze (default 50 ms).
    Returns:
        int: The position (in ms) where speech begins.
    """
    trim_point = 0
    for i in range(0, len(audio), segment_length_ms):
        segment = audio[i:i + segment_length_ms]
        rms = rms_in_db(segment)

        # Silence or breath (below the speech threshold)
        if rms <= breath_threshold_db:
            trim_point = i + segment_length_ms
        # Speech detected (strong increase in RMS above speech threshold)
        elif rms > speech_threshold_db:
            # We have detected the start of speech, stop trimming
            break

    return trim_point

def process_audio_folder(folder_path, output_folder, silence_threshold_db=-70, breath_threshold_db=-50, speech_threshold_db=-30, max_analysis_duration=1000):
    """
    Process each audio file in the folder, trimming leading breath sounds and silence based on RMS thresholds.
    Args:
        folder_path (str): Path to the folder containing audio files.
        output_folder (str): Path to save the processed files.
        silence_threshold_db (float): RMS threshold for detecting silence.
        breath_threshold_db (float): RMS threshold for detecting breath.
        speech_threshold_db (float): RMS threshold for detecting speech.
        max_analysis_duration (int): Maximum duration (ms) to analyze for trimming.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(folder_path):
        if filename.endswith('.wav') or filename.endswith('.mp3'):
            file_path = os.path.join(folder_path, filename)
            audio = AudioSegment.from_file(file_path)
            
            # Analyze only the first second or less if the audio is shorter
            analysis_duration = min(max_analysis_duration, len(audio))
            leading_audio = audio[:analysis_duration]

            # Detect where the speech starts (trim silence and breath)
            speech_start = detect_start_of_speech(leading_audio, silence_threshold_db, breath_threshold_db, speech_threshold_db)

            # Trim the audio up to the speech start
            trimmed_audio = audio[speech_start:]

            # Export the processed audio
            output_path = os.path.join(output_folder, filename)
            trimmed_audio.export(output_path, format="wav")
            print(f"Processed {filename} and saved to {output_path}")

def test_single_file(file_path, silence_threshold_db=-70, breath_threshold_db=-50, speech_threshold_db=-30, max_analysis_duration=1000):
    """
    Test mode: Analyze a single file and print the detected region for trimming.
    Args:
        file_path (str): Path to the audio file.
        silence_threshold_db (float): RMS threshold for detecting silence.
        breath_threshold_db (float): RMS threshold for detecting breath.
        speech_threshold_db (float): RMS threshold for detecting speech.
        max_analysis_duration (int): Maximum duration (ms) to analyze for trimming.
    """
    audio = AudioSegment.from_file(file_path)
    
    # Analyze only the first second or less if the audio is shorter
    analysis_duration = min(max_analysis_duration, len(audio))
    leading_audio = audio[:analysis_duration]

    # Detect where the speech starts (trim silence and breath)
    speech_start = detect_start_of_speech(leading_audio, silence_threshold_db, breath_threshold_db, speech_threshold_db)

    # Print details for test mode
    print(f"File: {file_path}")
    print(f"Detected start of speech at: {speech_start} ms")
    print(f"Would trim off: {speech_start} ms of leading audio")

    # Optionally play or save the trimmed audio for verification
    trimmed_audio = audio[speech_start:]
    trimmed_audio.export("test_trimmed_output.wav", format="wav")
    print(f"Trimmed test audio saved as 'test_trimmed_output.wav'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Trim leading breath and silence from audio files.")
    
    parser.add_argument('--folder', type=str, default="wavs", help="Folder containing audio files to process.")
    parser.add_argument('--output_folder', type=str, default="dbreath_wavs", help="Folder to save processed audio files.")
    parser.add_argument('--file', type=str, help="Single audio file to analyze in test mode.")
    parser.add_argument('--silence_threshold', type=float, default=-70, help="RMS threshold for silence (default: -70 dB).")
    parser.add_argument('--breath_threshold', type=float, default=-50, help="RMS threshold for breath (default: -50 dB).")
    parser.add_argument('--speech_threshold', type=float, default=-30, help="RMS threshold for speech (default: -30 dB).")
    parser.add_argument('--max_duration', type=int, default=1000, help="Maximum analysis duration in ms (default: 1000 ms).")
    
    args = parser.parse_args()

    if args.file:
        print("Running in test mode...")
        test_single_file(
            args.file,
            silence_threshold_db=args.silence_threshold,
            breath_threshold_db=args.breath_threshold,
            speech_threshold_db=args.speech_threshold,
            max_analysis_duration=args.max_duration
        )
    elif args.folder and args.output_folder:
        print("Processing audio files in folder...")
        process_audio_folder(
            args.folder,
            args.output_folder,
            silence_threshold_db=args.silence_threshold,
            breath_threshold_db=args.breath_threshold,
            speech_threshold_db=args.speech_threshold,
            max_analysis_duration=args.max_duration
        )
    else:
        print("Please provide either a folder with audio files or a single file for test mode.")
=======
import os
import argparse
from pydub import AudioSegment
import numpy as np

# Function to calculate RMS of an audio segment in dB
def rms_in_db(audio_segment):
    samples = np.array(audio_segment.get_array_of_samples())
    rms = np.sqrt(np.mean(samples**2))
    if rms > 0:
        return 20 * np.log10(rms)
    else:
        return -float('inf')

# Function to detect the start of speech, including silence and breath trimming
def detect_start_of_speech(audio, silence_threshold_db=-70, breath_threshold_db=-50, speech_threshold_db=-30, segment_length_ms=50):
    """
    Detects the start of speech by scanning through the audio, first detecting silence, then a breath, and finally speech.
    Args:
        audio (AudioSegment): The audio file to analyze.
        silence_threshold_db (float): Threshold for detecting silence in dB.
        breath_threshold_db (float): Threshold for detecting breath in dB.
        speech_threshold_db (float): Threshold for detecting speech in dB.
        segment_length_ms (int): Length of segments to analyze (default 50 ms).
    Returns:
        int: The position (in ms) where speech begins.
    """
    trim_point = 0
    for i in range(0, len(audio), segment_length_ms):
        segment = audio[i:i + segment_length_ms]
        rms = rms_in_db(segment)

        # Silence or breath (below the speech threshold)
        if rms <= breath_threshold_db:
            trim_point = i + segment_length_ms
        # Speech detected (strong increase in RMS above speech threshold)
        elif rms > speech_threshold_db:
            # We have detected the start of speech, stop trimming
            break

    return trim_point

def process_audio_folder(folder_path, output_folder, silence_threshold_db=-70, breath_threshold_db=-50, speech_threshold_db=-30, max_analysis_duration=1000):
    """
    Process each audio file in the folder, trimming leading breath sounds and silence based on RMS thresholds.
    Args:
        folder_path (str): Path to the folder containing audio files.
        output_folder (str): Path to save the processed files.
        silence_threshold_db (float): RMS threshold for detecting silence.
        breath_threshold_db (float): RMS threshold for detecting breath.
        speech_threshold_db (float): RMS threshold for detecting speech.
        max_analysis_duration (int): Maximum duration (ms) to analyze for trimming.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(folder_path):
        if filename.endswith('.wav') or filename.endswith('.mp3'):
            file_path = os.path.join(folder_path, filename)
            audio = AudioSegment.from_file(file_path)
            
            # Analyze only the first second or less if the audio is shorter
            analysis_duration = min(max_analysis_duration, len(audio))
            leading_audio = audio[:analysis_duration]

            # Detect where the speech starts (trim silence and breath)
            speech_start = detect_start_of_speech(leading_audio, silence_threshold_db, breath_threshold_db, speech_threshold_db)

            # Trim the audio up to the speech start
            trimmed_audio = audio[speech_start:]

            # Export the processed audio
            output_path = os.path.join(output_folder, filename)
            trimmed_audio.export(output_path, format="wav")
            print(f"Processed {filename} and saved to {output_path}")

def test_single_file(file_path, silence_threshold_db=-70, breath_threshold_db=-50, speech_threshold_db=-30, max_analysis_duration=1000):
    """
    Test mode: Analyze a single file and print the detected region for trimming.
    Args:
        file_path (str): Path to the audio file.
        silence_threshold_db (float): RMS threshold for detecting silence.
        breath_threshold_db (float): RMS threshold for detecting breath.
        speech_threshold_db (float): RMS threshold for detecting speech.
        max_analysis_duration (int): Maximum duration (ms) to analyze for trimming.
    """
    audio = AudioSegment.from_file(file_path)
    
    # Analyze only the first second or less if the audio is shorter
    analysis_duration = min(max_analysis_duration, len(audio))
    leading_audio = audio[:analysis_duration]

    # Detect where the speech starts (trim silence and breath)
    speech_start = detect_start_of_speech(leading_audio, silence_threshold_db, breath_threshold_db, speech_threshold_db)

    # Print details for test mode
    print(f"File: {file_path}")
    print(f"Detected start of speech at: {speech_start} ms")
    print(f"Would trim off: {speech_start} ms of leading audio")

    # Optionally play or save the trimmed audio for verification
    trimmed_audio = audio[speech_start:]
    trimmed_audio.export("test_trimmed_output.wav", format="wav")
    print(f"Trimmed test audio saved as 'test_trimmed_output.wav'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Trim leading breath and silence from audio files.")
    
    parser.add_argument('--folder', type=str, default="wavs", help="Folder containing audio files to process.")
    parser.add_argument('--output_folder', type=str, default="dbreath_wavs", help="Folder to save processed audio files.")
    parser.add_argument('--file', type=str, help="Single audio file to analyze in test mode.")
    parser.add_argument('--silence_threshold', type=float, default=-70, help="RMS threshold for silence (default: -70 dB).")
    parser.add_argument('--breath_threshold', type=float, default=-50, help="RMS threshold for breath (default: -50 dB).")
    parser.add_argument('--speech_threshold', type=float, default=-30, help="RMS threshold for speech (default: -30 dB).")
    parser.add_argument('--max_duration', type=int, default=1000, help="Maximum analysis duration in ms (default: 1000 ms).")
    
    args = parser.parse_args()

    if args.file:
        print("Running in test mode...")
        test_single_file(
            args.file,
            silence_threshold_db=args.silence_threshold,
            breath_threshold_db=args.breath_threshold,
            speech_threshold_db=args.speech_threshold,
            max_analysis_duration=args.max_duration
        )
    elif args.folder and args.output_folder:
        print("Processing audio files in folder...")
        process_audio_folder(
            args.folder,
            args.output_folder,
            silence_threshold_db=args.silence_threshold,
            breath_threshold_db=args.breath_threshold,
            speech_threshold_db=args.speech_threshold,
            max_analysis_duration=args.max_duration
        )
    else:
        print("Please provide either a folder with audio files or a single file for test mode.")
>>>>>>> ceac1a01c6a4b34b4e5f6d0a848bfa9025349a88

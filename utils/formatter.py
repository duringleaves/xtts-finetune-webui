import os
import gc
import torchaudio
import pandas as pd
from faster_whisper import WhisperModel
from glob import glob
from tqdm import tqdm
from TTS.tts.layers.xtts.tokenizer import multilingual_cleaners
import torch
import argparse

torch.set_num_threads(16)

audio_types = (".wav", ".mp3", ".flac")

def find_latest_best_model(folder_path):
    search_path = os.path.join(folder_path, '**', 'best_model.pth')
    files = glob(search_path, recursive=True)
    latest_file = max(files, key=os.path.getctime, default=None)
    return latest_file

def list_audios(basePath, contains=None):
    return list_files(basePath, validExts=audio_types, contains=contains)

def list_files(basePath, validExts=None, contains=None):
    for (rootDir, dirNames, filenames) in os.walk(basePath):
        for filename in filenames:
            if contains is not None and filename.find(contains) == -1:
                continue
            ext = filename[filename.rfind("."):].lower()
            if validExts is None or ext.endswith(validExts):
                audioPath = os.path.join(rootDir, filename)
                yield audioPath

def format_audio_list(audio_files, asr_model, target_language="en", buffer=0.2, eval_percentage=0.15, speaker_name="coqui", display_limit=10):
    audio_total_size = 0
    metadata = {"audio_file": [], "text": [], "speaker_name": []}

    # Prepare tqdm progress bar for console or Gradio if provided
    tqdm_object = tqdm(audio_files)

    display_count = 0
    for audio_path in tqdm_object:
        if display_count >= display_limit:
            break

        wav, sr = torchaudio.load(audio_path)
        if wav.size(0) != 1:
            wav = torch.mean(wav, dim=0, keepdim=True)
        wav = wav.squeeze()
        audio_total_size += (wav.size(-1) / sr)

        segments, _ = asr_model.transcribe(audio_path, vad_filter=True, word_timestamps=True, language=target_language)
        segments = list(segments)
        i = 0
        sentence = ""
        sentence_start = None
        first_word = True
        words_list = []
        for _, segment in enumerate(segments):
            words = list(segment.words)
            words_list.extend(words)

        for word_idx, word in enumerate(words_list):
            if first_word:
                sentence_start = word.start
                if word_idx == 0:
                    # Start of the sentence, use buffer before the start of the word
                    sentence_start = max(sentence_start - buffer, 0)
                else:
                    previous_word_end = words_list[word_idx - 1].end
                    sentence_start = max(sentence_start - buffer, (previous_word_end + sentence_start) / 2)
                sentence = word.word
                first_word = False
            else:
                sentence += word.word

            # When the sentence ends (punctuation), process it as a complete sentence
            if word.word[-1] in ["!", "。", ".", "?"]:
                sentence = sentence[1:]  # Remove leading space if present
                sentence = multilingual_cleaners(sentence, target_language)

                audio_file_name, _ = os.path.splitext(os.path.basename(audio_path))
                audio_file = f"{audio_file_name}_{str(i).zfill(8)}.wav"

                # Set the end of the sentence to just before the next word starts, or the end of the waveform
                if word_idx + 1 < len(words_list):
                    next_word_start = words_list[word_idx + 1].start
                    word_end = min(next_word_start - buffer, next_word_start)
                else:
                    # If no next word, use the end of the waveform
                    word_end = wav.shape[0] / sr

                # Clip the audio between sentence_start and word_end
                audio = wav[int(sr * sentence_start):int(sr * word_end)].unsqueeze(0)
                
                # Only save if the audio clip is at least 1/3 second long
                if audio.size(-1) >= sr / 3:
                    # Output metadata for debug purposes
                    print(f"Audio File: {audio_file}, Text: {sentence}, Speaker: {speaker_name}")
                    metadata["audio_file"].append(audio_file)
                    metadata["text"].append(sentence)
                    metadata["speaker_name"].append(speaker_name)

                i += 1
                first_word = True

        display_count += 1

    return metadata, audio_total_size


# Main function for command line execution
if __name__ == "__main__":
    # Argument parser for command line inputs
    parser = argparse.ArgumentParser(description="ASR Audio Processing Script")
    parser.add_argument("input_audio", type=str, help="Path to input audio file or directory")
    parser.add_argument("--language", type=str, default="en", help="Target language for transcription")
    parser.add_argument("--limit", type=int, default=10, help="Number of audio segments to display")
    
    args = parser.parse_args()

    # Load Whisper ASR model
    asr_model = WhisperModel("small", device="cuda" if torch.cuda.is_available() else "cpu")

    # Check if input is a single file or directory
    if os.path.isfile(args.input_audio):
        audio_files = [args.input_audio]
    else:
        audio_files = list(list_audios(args.input_audio))

    # Process the audio list and display the limited metadata
    metadata, total_audio_size = format_audio_list(audio_files, asr_model, target_language=args.language, display_limit=args.limit)
    
    print(f"Processed total audio size: {total_audio_size} seconds")

import os
import gc
import torchaudio
import pandas
from faster_whisper import WhisperModel
from glob import glob

from tqdm import tqdm

from TTS.tts.layers.xtts.tokenizer import multilingual_cleaners  # Keep this as required
import torch

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

def format_audio_list(audio_files, asr_model, target_language="en", out_path=None, buffer=0.4, eval_percentage=0.15, speaker_name="coqui", gradio_progress=None):
    audio_total_size = 0
    os.makedirs(out_path, exist_ok=True)

    # Sentence-level transcript export path
    transcript_path = os.path.join(out_path, "sentence_transcript.csv")
    transcript_data = {"audio_file": [], "sentence_start": [], "sentence_end": [], "sentence": []}

    print("Checking lang.txt")
    lang_file_path = os.path.join(out_path, "lang.txt")
    current_language = None
    if os.path.exists(lang_file_path):
        with open(lang_file_path, 'r', encoding='utf-8') as existing_lang_file:
            current_language = existing_lang_file.read().strip()

    if current_language != target_language:
        with open(lang_file_path, 'w', encoding='utf-8') as lang_file:
            lang_file.write(target_language + '\n')
        print("Warning, existing language does not match target language. Updated lang.txt with target language.")
    else:
        print("Existing language matches target language")

    print("Checking for existing metadata_train.csv and metadata_eval.csv")
    metadata = {"audio_file": [], "text": [], "speaker_name": []}
    train_metadata_path = os.path.join(out_path, "metadata_train.csv")
    eval_metadata_path = os.path.join(out_path, "metadata_eval.csv")

    existing_metadata = {'train': None, 'eval': None}
    if os.path.exists(train_metadata_path):
        existing_metadata['train'] = pandas.read_csv(train_metadata_path, sep="|")
        print("Existing training metadata found and loaded.")

    if os.path.exists(eval_metadata_path):
        existing_metadata['eval'] = pandas.read_csv(eval_metadata_path, sep="|")
        print("Existing evaluation metadata found and loaded.")

    if gradio_progress is not None:
        tqdm_object = gradio_progress.tqdm(audio_files, desc="Formatting...")
    else:
        tqdm_object = tqdm(audio_files)

    for audio_path in tqdm_object:
        audio_file_name_without_ext, _= os.path.splitext(os.path.basename(audio_path))
        prefix_check = f"wavs/{audio_file_name_without_ext}_"

        skip_processing = False
        for key in ['train', 'eval']:
            if existing_metadata[key] is not None:
                mask = existing_metadata[key]['audio_file'].str.startswith(prefix_check)
                if mask.any():
                    print(f"Segments from {audio_file_name_without_ext} have been previously processed; skipping...")
                    skip_processing = True
                    break

        if skip_processing:
            continue

        wav, sr = torchaudio.load(audio_path)
        if wav.size(0) != 1:
            wav = torch.mean(wav, dim=0, keepdim=True)

        wav = wav.squeeze()
        audio_total_size += (wav.size(-1) / sr)

        segments, _ = asr_model.transcribe(audio_path, vad_filter=True, word_timestamps=True, language=target_language)
        segments = list(segments)
        print(f"Found {len(segments)} segments")

        words_list = []
        for segment in segments:
            words = list(segment.words)
            words_list.extend(words)
            print(f"Found {len(words)} words in segment.")

        # Sentence processing logic (group words into sentences)
        i = 0
        sentence = ""
        sentence_start = None
        first_word = True
        sentence_end = None
        for word_idx, word in enumerate(words_list):
            if first_word:
                sentence_start = word.start
                if word_idx == 0:
                    sentence_start = max(sentence_start - buffer, 0)
                else:
                    previous_word_end = words_list[word_idx - 1].end
                    sentence_start = max(sentence_start - buffer, (previous_word_end + sentence_start) / 2)

                sentence = word.word
                first_word = False
            else:
                sentence += " " + word.word

            is_last_word = word.word[-1] in ["!", ".", "?"]
            if is_last_word:
                sentence = multilingual_cleaners(sentence, target_language)

                sentence_end = word.end + buffer

                # Store sentence data in the transcript_data
                transcript_data["audio_file"].append(audio_file_name_without_ext)
                transcript_data["sentence_start"].append(sentence_start)
                transcript_data["sentence_end"].append(sentence_end)
                transcript_data["sentence"].append(sentence)

                # Reset for the next sentence
                first_word = True
                i += 1

    # Save the sentence-level transcript to a CSV
    transcript_df = pandas.DataFrame(transcript_data)
    transcript_df.to_csv(transcript_path, index=False)

    # Return the original 3 values as expected
    return train_metadata_path, eval_metadata_path, audio_total_size


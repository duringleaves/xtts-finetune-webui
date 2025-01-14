import os
import gc
import torch
import torchaudio
import pandas
import numpy as np
from faster_whisper import WhisperModel
from glob import glob
from tqdm import tqdm
from TTS.tts.layers.xtts.tokenizer import multilingual_cleaners

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

def detect_speech_boundaries(wav, sr, start_time, end_time, buffer_ms=500, energy_threshold=0.1, min_speech_duration=0.1):
    """
    Detect precise speech boundaries using signal energy and zero-crossing rate
    
    Args:
        wav: Audio waveform tensor
        sr: Sample rate
        start_time: Initial start time estimate
        end_time: Initial end time estimate
        buffer_ms: Analysis buffer in milliseconds
        energy_threshold: Threshold for speech detection (relative to max energy)
        min_speech_duration: Minimum speech duration in seconds
    """
    # Convert times to samples
    buffer_samples = int((buffer_ms / 1000) * sr)
    start_sample = max(0, int(start_time * sr) - buffer_samples)
    end_sample = min(len(wav), int(end_time * sr) + buffer_samples)
    
    # Extract segment with buffer
    segment = wav[start_sample:end_sample]
    if len(segment) == 0:
        return start_time, end_time
    
    # Calculate frame-wise energy using windowing
    frame_length = int(0.02 * sr)  # 20ms frames
    hop_length = int(0.01 * sr)    # 10ms hop
    
    # Ensure segment is long enough for framing
    if len(segment) < frame_length:
        return start_time, end_time

    # Create frames manually
    num_frames = (len(segment) - frame_length) // hop_length + 1
    frames = []
    
    for i in range(num_frames):
        frame_start = i * hop_length
        frame_end = frame_start + frame_length
        frame = segment[frame_start:frame_end]
        frames.append(frame)
    
    if not frames:  # If no frames were created
        return start_time, end_time
        
    frames = torch.stack(frames)
    
    # Calculate energy for each frame
    energy = torch.mean(frames.pow(2), dim=1)
    
    # Calculate zero-crossing rate
    zcr = torch.sum(
        torch.sign(frames[:, :-1]) != torch.sign(frames[:, 1:]),
        dim=1
    ).float() / frame_length
    
    # Combine features with weighted zero-crossing rate
    features = energy * (1 - 0.5 * zcr)  # Reduced weight for high-frequency noise
    
    # Find speech regions using dynamic threshold
    threshold = energy_threshold * torch.max(features)
    speech_mask = features > threshold
    
    # Find speech boundaries
    speech_indices = torch.nonzero(speech_mask).squeeze()
    if len(speech_indices) == 0 or not isinstance(speech_indices, torch.Tensor):
        return start_time, end_time
        
    # If only one frame is detected as speech
    if speech_indices.dim() == 0:
        speech_indices = speech_indices.unsqueeze(0)
    
    # Convert frame indices to time
    precise_start = start_sample + (speech_indices[0] * hop_length)
    precise_end = start_sample + (speech_indices[-1] * hop_length + frame_length)
    
    # Ensure minimum duration
    min_samples = int(min_speech_duration * sr)
    if precise_end - precise_start < min_samples:
        pad_samples = (min_samples - (precise_end - precise_start)) // 2
        precise_start = max(0, precise_start - pad_samples)
        precise_end = min(len(wav), precise_end + pad_samples)
    
    # Add safety margins
    final_start = max(0, precise_start - int(0.1 * sr))  # 100ms margin
    final_end = min(len(wav), precise_end + int(0.15 * sr))  # 150ms margin
    
    return final_start / sr, final_end / sr

def apply_audio_effects(audio, sr):
    """
    Apply audio processing effects including fade in/out and normalization
    """
    # Apply fade in/out
    fade_samples = int(0.02 * sr)  # 20ms fade
    if len(audio) > 2 * fade_samples:
        fade_in = torch.linspace(0, 1, fade_samples)
        fade_out = torch.linspace(1, 0, fade_samples)
        audio[:fade_samples] *= fade_in
        audio[-fade_samples:] *= fade_out
    
    # Normalize audio
    if torch.max(torch.abs(audio)) > 0:
        audio = audio / torch.max(torch.abs(audio)) * 0.95
    
    return audio

def format_audio_list(audio_files, asr_model, target_language="en", out_path=None, buffer=0.2, eval_percentage=0.15, speaker_name="coqui", gradio_progress=None):
    """
    Process audio files with improved segmentation and audio quality
    """
    audio_total_size = 0
    os.makedirs(out_path, exist_ok=True)

    # Handle language file
    lang_file_path = os.path.join(out_path, "lang.txt")
    current_language = None
    if os.path.exists(lang_file_path):
        with open(lang_file_path, 'r', encoding='utf-8') as existing_lang_file:
            current_language = existing_lang_file.read().strip()

    if current_language != target_language:
        with open(lang_file_path, 'w', encoding='utf-8') as lang_file:
            lang_file.write(target_language + '\n')
        print("Warning: Updated lang.txt with target language.")

    # Set up metadata handling
    metadata = {"audio_file": [], "text": [], "speaker_name": []}
    train_metadata_path = os.path.join(out_path, "metadata_train.csv")
    eval_metadata_path = os.path.join(out_path, "metadata_eval.csv")

    # Load existing metadata if available
    existing_metadata = {'train': None, 'eval': None}
    if os.path.exists(train_metadata_path):
        existing_metadata['train'] = pandas.read_csv(train_metadata_path, sep="|")
    if os.path.exists(eval_metadata_path):
        existing_metadata['eval'] = pandas.read_csv(eval_metadata_path, sep="|")

    # Setup progress bar
    if gradio_progress is not None:
        tqdm_object = gradio_progress.tqdm(audio_files, desc="Processing audio files...")
    else:
        tqdm_object = tqdm(audio_files)

    # Process each audio file
    for audio_path in tqdm_object:
        wav, sr = torchaudio.load(audio_path)
        if wav.size(0) != 1:
            wav = torch.mean(wav, dim=0, keepdim=True)
        wav = wav.squeeze()
        audio_total_size += (wav.size(-1) / sr)

        segments, _ = asr_model.transcribe(
            audio_path,
            vad_filter=True,
            word_timestamps=True,
            language=target_language
        )

        i = 0
        current_sentence = []
        current_words = []
        
        for segment in segments:
            words = list(segment.words)
            
            for word_idx, word in enumerate(words):
                current_words.append(word)
                current_sentence.append(word.word)
                
                # Calculate current duration from start of sentence
                current_duration = word.end - current_words[0].start
                
                # Check if we should split (either end of sentence or max duration reached)
                should_split = False
                split_reason = ""
                
                if word.word[-1] in ["!", "。", ".", "?"]:
                    # Natural sentence end - verify it's really the end with a proper gap
                    if word_idx + 1 >= len(words) or words[word_idx + 1].start - word.end >= 0.2:
                        should_split = True
                        split_reason = "punctuation"
                
                if current_duration >= 14.0:  # Split before reaching 15 seconds
                    should_split = True
                    split_reason = "duration"
                
                if should_split and current_words:
                    # Clean and prepare text
                    text = " ".join(current_sentence)
                    text = text.strip()
                    text = multilingual_cleaners(text, target_language)
                    
                    # Calculate audio boundaries - start exactly at first word, small padding only at end
                    start_time = current_words[0].start
                    end_time = min(len(wav)/sr, word.end + 0.15)  # Small padding only at end
                    
                    # Ensure we don't exceed duration limit
                    if (end_time - start_time) > 15.0:
                        end_time = start_time + 14.8  # Leave room for fade out
                    
                    # Extract audio
                    audio = wav[int(sr * start_time):int(sr * end_time)]
                    
                    # Apply fade out only
                    fade_samples = int(0.02 * sr)  # 20ms fade
                    if len(audio) > fade_samples:
                        fade_out = torch.linspace(1, 0, fade_samples)
                        audio[-fade_samples:] *= fade_out
                    
                    # Verify minimum duration
                    if len(audio) >= sr * 0.5:  # At least 0.5 seconds
                        # Save audio segment
                        audio_file = f"wavs/{os.path.splitext(os.path.basename(audio_path))[0]}_{str(i).zfill(8)}.wav"
                        absolute_path = os.path.join(out_path, audio_file)
                        os.makedirs(os.path.dirname(absolute_path), exist_ok=True)
                        
                        torchaudio.save(absolute_path, audio.unsqueeze(0), sr)
                        
                        # Update metadata
                        metadata["audio_file"].append(audio_file)
                        metadata["text"].append(text)
                        metadata["speaker_name"].append(speaker_name)
                        
                        # Save metadata incrementally
                        df = pandas.DataFrame(metadata)
                        mode = 'w' if not os.path.exists(train_metadata_path) else 'a'
                        header = not os.path.exists(train_metadata_path)
                        df.to_csv(train_metadata_path, sep="|", index=False, mode=mode, header=header)
                        
                        metadata = {"audio_file": [], "text": [], "speaker_name": []}
                        i += 1
                    
                    # Reset collectors
                    current_sentence = []
                    current_words = []
                    
                    # If split due to duration, keep current word for next segment
                    if split_reason == "duration":
                        current_words.append(word)
                        current_sentence.append(word.word)

    # Final metadata processing
    if os.path.exists(train_metadata_path):
        # Load all data
        full_df = pandas.read_csv(train_metadata_path, sep="|")
        
        # Shuffle and split
        shuffled_df = full_df.sample(frac=1, random_state=42)
        split_idx = int(len(shuffled_df) * eval_percentage)
        
        # Create evaluation and training sets
        eval_df = shuffled_df[:split_idx]
        train_df = shuffled_df[split_idx:]
        
        # Save final datasets
        train_df.sort_values('audio_file').to_csv(train_metadata_path, sep='|', index=False)
        eval_df.sort_values('audio_file').to_csv(eval_metadata_path, sep='|', index=False)

    return train_metadata_path, eval_metadata_path, audio_total_size
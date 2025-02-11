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
    Process audio files and create metadata using original audio files
    """
    audio_total_size = 0
    os.makedirs(out_path, exist_ok=True)
    os.makedirs(os.path.join(out_path, "wavs"), exist_ok=True)

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

        # Store using original audio file path
        audio_file = f"wavs/{os.path.basename(audio_path)}"
        
        # Create symbolic link to original file in wavs directory
        symlink_path = os.path.join(out_path, audio_file)
        os.makedirs(os.path.dirname(symlink_path), exist_ok=True)
        if not os.path.exists(symlink_path):
            try:
                os.symlink(audio_path, symlink_path)
            except:
                # If symlink fails, copy the file
                torchaudio.save(symlink_path, wav.unsqueeze(0), sr)

        # Combine all segments into one text
        full_text = []
        for segment in segments:
            words = [word.word for word in segment.words]
            segment_text = " ".join(words)
            full_text.append(segment_text)

        text = " ".join(full_text)
        text = text.strip()
        text = multilingual_cleaners(text, target_language)

        metadata["audio_file"].append(audio_file)
        metadata["text"].append(text)
        metadata["speaker_name"].append(speaker_name)

    # Create final metadata DataFrame
    df = pandas.DataFrame(metadata)
    
    # Shuffle and split
    shuffled_df = df.sample(frac=1, random_state=42)
    split_idx = int(len(shuffled_df) * eval_percentage)
    
    # Create evaluation and training sets
    eval_df = shuffled_df[:split_idx]
    train_df = shuffled_df[split_idx:]
    
    # Save final datasets
    train_df.sort_values('audio_file').to_csv(train_metadata_path, sep='|', index=False)
    eval_df.sort_values('audio_file').to_csv(eval_metadata_path, sep='|', index=False)

    return train_metadata_path, eval_metadata_path, audio_total_size
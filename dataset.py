# =============================================================================
# Import required libraries
# =============================================================================
import os
import torchaudio
from torch.utils.data import Dataset, DataLoader

# Supported audio extensions
AUDIO_EXTENSIONS = [
    '.wav', '.WAV', '.mp3', '.MP3', '.flac', '.FLAC', '.ogg', '.OGG', '.m4a'
]


def is_audio_file(filename):
    return any(filename.endswith(extension) for extension in AUDIO_EXTENSIONS)

def make_dataset(dataset_root):
    audios = []
    cnt = 0 
    assert os.path.isdir(dataset_root), f'{dataset_root} is not a valid directory'

    for speaker_id in sorted(os.listdir(dataset_root)):
        video_path = os.path.join(dataset_root, speaker_id)
        for vid in os.listdir(video_path):
            audio_file_path = os.path.join(video_path, vid)
            for audio_file in os.listdir(audio_file_path):
                if is_audio_file(audio_file):
                    audio_path = os.path.join(audio_file_path, audio_file)
                    audios.append((speaker_id, audio_path))

                break 
            break 
        
        if cnt == 3:
            break
         
        cnt += 1

    return audios


class AudioDataset(Dataset):
    def __init__(self, source_root, source_transform=None, target_sample_rate=None):
        """
        Args:
            source_root (str): Path to the directory containing audio files.
            source_transform (callable, optional): Optional transform to be applied
                on a sample (e.g., mel spectrogram).
            target_sample_rate (int, optional): Resample to this rate if given.
        """
        self.source_paths = sorted(make_dataset(source_root)) 
        self.source_transform = source_transform
        self.target_sample_rate = target_sample_rate

    def __len__(self):
        return len(self.source_paths)

    def __getitem__(self, index):
        fname, audio_path = self.source_paths[index]

        # Load audio (returns waveform, sample_rate)
        waveform, sample_rate = torchaudio.load(audio_path)

        # Optional resampling
        if self.target_sample_rate and sample_rate != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate,
                                                       new_freq=self.target_sample_rate)
            waveform = resampler(waveform)
            sample_rate = self.target_sample_rate

        # Optional waveform transform (e.g., convert to Mel, normalize, etc.)
        if self.source_transform:
            waveform = self.source_transform(waveform)

        return fname, waveform




# def make_dataset(dir):
#     audios = []
#     assert os.path.isdir(dir), f'{dir} is not a valid directory'
#     for root, _, fnames in sorted(os.walk(dir)):
#         for fname in fnames:
#             if is_audio_file(fname):
#                 path = os.path.join(root, fname)
#                 fname = fname.split('.')[0]
#                 audios.append((fname, path))
#     return audios


# class AudioDataset(Dataset):
#     def __init__(self, source_root, source_transform=None, target_sample_rate=None):
#         """
#         Args:
#             source_root (str): Path to the directory containing audio files.
#             source_transform (callable, optional): Optional transform to be applied
#                 on a sample (e.g., mel spectrogram).
#             target_sample_rate (int, optional): Resample to this rate if given.
#         """
#         self.source_paths = sorted(make_dataset(source_root)) 
#         self.source_transform = source_transform
#         self.target_sample_rate = target_sample_rate

#     def __len__(self):
#         return len(self.source_paths)

#     def __getitem__(self, index):
#         fname, audio_path = self.source_paths[index]

#         # Load audio (returns waveform, sample_rate)
#         waveform, sample_rate = torchaudio.load(audio_path)

#         # Optional resampling
#         if self.target_sample_rate and sample_rate != self.target_sample_rate:
#             resampler = torchaudio.transforms.Resample(orig_freq=sample_rate,
#                                                        new_freq=self.target_sample_rate)
#             waveform = resampler(waveform)
#             sample_rate = self.target_sample_rate

#         # Optional waveform transform (e.g., convert to Mel, normalize, etc.)
#         if self.source_transform:
#             waveform = self.source_transform(waveform)

#         return fname, waveform


# dataset_root = "dataset_root/wav"

# dataset = AudioDataset(source_root=dataset_root, target_sample_rate=16000)
# dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

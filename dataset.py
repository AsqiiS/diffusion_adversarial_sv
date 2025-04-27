import os
import torchaudio
from torch.utils.data import Dataset, DataLoader
import pandas as pd 

AUDIO_EXTENSIONS = [
    '.wav', '.WAV', '.mp3', '.MP3', '.flac', '.FLAC', '.ogg', '.OGG', '.m4a'
]

def is_audio_file(filename):
    return any(filename.endswith(extension) for extension in AUDIO_EXTENSIONS)


def make_dataset(dataset_root, transcription_dict=None): 
    assert os.path.isdir(dataset_root), f'{dataset_root} is not a valid directory'
    
    if transcription_dict == None: 
        audios = []

        for speaker_id in sorted(os.listdir(dataset_root)):
            video_path = os.path.join(dataset_root, speaker_id)
            for vid in os.listdir(video_path):
                audio_file_path = os.path.join(video_path, vid)
                for audio_file in os.listdir(audio_file_path):
                    if is_audio_file(audio_file):
                        audio_path = os.path.join(audio_file_path, audio_file)
                        audios.append((speaker_id, audio_path))

        return audios, None 
    
    else: 
        audios = []
        transcripts = []
        for audio_path, transcript in transcription_dict.items(): 
            if not os.path.exists(audio_path):
                continue 

            speaker_id = audio_path.split('/')[1]
            audios.append((speaker_id, audio_path))
            transcripts.append(transcript)

        return audios, transcripts 

class AudioDataset(Dataset):
    def __init__(self, source_root, transcription_dict=None, source_transform=None, target_sample_rate=None, take_subset=None):
        self.transcription_dict = transcription_dict or {}
        self.source_paths, self.transripts = make_dataset(source_root, transcription_dict=transcription_dict)
        self.source_paths = self.source_paths[:take_subset] # resize the dataset size 
        self.source_root = source_root 

        self.source_transform = source_transform
        self.target_sample_rate = target_sample_rate

    def __len__(self):
        return len(self.source_paths)

    def __getitem__(self, index):
        fname, audio_path = self.source_paths[index]
        transcription = self.transripts[index]

        waveform, sample_rate = torchaudio.load(audio_path)

        if self.target_sample_rate and sample_rate != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate,
                                                       new_freq=self.target_sample_rate)
            waveform = resampler(waveform)
            sample_rate = self.target_sample_rate

        if self.source_transform:
            waveform = self.source_transform(waveform)

        return fname, waveform, transcription

df = pd.read_csv('transcriptions (1).tsv', sep='\t')

transcription_dict = {
    row.audio_path.replace('data/', '').strip(): row.transcription
    for _, row in df.iterrows()
}

dataset_root = "wav"

# dataset = AudioDataset(source_root=dataset_root, transcription_dict=transcription_dict, target_sample_rate=16000)
# dataset = dataset[:100]

# dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

# print(len(dataset))


# for fname, waveform, transcription in dataloader:
#     print(fname, waveform.shape, transcription)
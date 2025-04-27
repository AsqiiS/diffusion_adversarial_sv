import scipy
import torch
from diffusers import AudioLDM2Pipeline
from diffusers import AudioLDM2UNet2DConditionModel
from speechbrain.inference.speaker import EncoderClassifier
from dataset import AudioDataset
from torch.utils.data import DataLoader
from adversarial_optimization import AdversarialAudioOpt
import pandas as pd 

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", run_opts={'device':'cuda'}) # 16k sampling_rate

repo_id = "anhnct/audioldm2_gigaspeech"
pipe = AudioLDM2Pipeline.from_pretrained(repo_id, torch_dtype=torch.float16)
pipe = pipe.to("cuda")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

df = pd.read_csv('transcriptions (1).tsv', sep='\t')

transcription_dict = {
    row.audio_path.replace('data/', '').strip(): row.transcription
    for _, row in df.iterrows()
}

dataset_root = "wav"

# choose higher number for take_subset for bigger dataset or set None to use full
dataset = AudioDataset(source_root=dataset_root, transcription_dict=transcription_dict, target_sample_rate=16000, take_subset=2)

dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

print(len(dataloader.dataset))

target_path = '/home/askhat.sametov/Downloads/ldm_adversarial_project/wav/id10005/1geDB-I2TjE/00001.wav'

class Args:
    def __init__(self):
        self.device = device
        self.source_dir = dataset_root
        self.protected_audio_dir = "./protected_audio"
        self.comparison_null_text = ""
        self.image_size = 64
        self.prot_steps = 30 
        self.diffusion_steps = 20 
        self.start_step = 17 
        self.null_optimization_steps = 10  
        self.adv_optim_weight = 1.0
        self.is_obfuscation = False
        self.target_choice = target_path  
        self.test_model_name = 'ecapa'
        self.surrogate_models = [classifier]
        self.dataloader = dataloader


args = Args()
attack = AdversarialAudioOpt(args, pipe)
attack.run()
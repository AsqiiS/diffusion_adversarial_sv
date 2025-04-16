import scipy
import torch
from diffusers import AudioLDM2Pipeline
from diffusers import AudioLDM2UNet2DConditionModel
from speechbrain.inference.speaker import EncoderClassifier
from dataset import AudioDataset
from torch.utils.data import DataLoader
from adversarial_optimization import AdversarialAudioOpt

classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", run_opts={'device':'cuda'}) # 16k sampling_rate

repo_id = "anhnct/audioldm2_gigaspeech"
pipe = AudioLDM2Pipeline.from_pretrained(repo_id, torch_dtype=torch.float16)
pipe = pipe.to("cuda")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset_root = "./wav"

audio_dataset = AudioDataset(dataset_root, target_sample_rate=16000)

dataloader = DataLoader(audio_dataset, batch_size=1, shuffle=False)

target_path = 'wav/id10001/1zcIwhmdeo4/00002.wav'

class Args:
    def __init__(self):
        self.device = device
        self.source_dir = dataset_root
        self.protected_audio_dir = "./protected_audio"
        self.comparison_null_text = ""
        self.image_size = 64
        self.prot_steps = 10 # should be 5
        self.diffusion_steps = 20 # change to bigger number
        self.start_step = 3
        self.null_optimization_steps = 2 # should be 2
        self.adv_optim_weight = 1.0
        self.is_obfuscation = True
        self.target_choice = target_path  # <-- path to target audio
        self.test_model_name = 'ecapa'
        self.surrogate_models = [classifier]
        self.dataloader = dataloader


args = Args()
attack = AdversarialAudioOpt(args, pipe)
attack.run()
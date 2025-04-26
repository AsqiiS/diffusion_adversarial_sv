import torch
from torch import nn, optim
import torchaudio
from tqdm import tqdm

from torchvision import transforms
import numpy as np
import pandas as pd
import os
import gc
import transformers 
import logging
import matplotlib.pyplot as plt


# from audioldm import AudioDiffusionModel  # e.g., placeholder
# from speaker_verification import ECAPA_TDNN  # placeholder
# from loss import CosineLoss, CLAPLoss        # hypothetical

def visualize_spec(audio):
    spectrogram = torchaudio.transforms.Spectrogram(n_fft=1024, hop_length=256)(audio)
    log_spec = torch.log1p(spectrogram)   

    plt.figure(figsize=(10, 4))
    plt.imshow(log_spec.squeeze().numpy(), origin='lower', aspect='auto', cmap='magma')
    plt.title("Original audio")
    plt.xlabel("Time")
    plt.ylabel("Frequency")
    plt.colorbar(label="Log Power")
    plt.tight_layout()
    plt.show()

def cosine_similarity(emb_1, emb_2):
    emb_1 = emb_1.view(1, -1) if emb_1.ndim == 1 else emb_1
    emb_2 = emb_2.view(1, -1) if emb_2.ndim == 1 else emb_2
    dot = torch.sum(emb_1 * emb_2, dim=1)
    norm1 = emb_1.norm(dim=1)
    norm2 = emb_2.norm(dim=1)
    return torch.mean(dot / (norm1 * norm2 + 1e-8))  # stability epsilon

# protected_feature: embeddings from adversarially modified input 
# target_feature: embeddings of the target person to impersonate
# source_feature: embeddings of the original source identity 

def cosine_loss(protected_feature, target_feature, source_feature=None, is_obfuscation=False):
    cos_loss_list = []
    for i in range(len(protected_feature)):
        if not is_obfuscation: # impersonation task 
            loss = 1 - cosine_similarity(protected_feature[i], target_feature[i].detach())
        else: # avoid impersonation and preserve identity 
            imp = 1 - cosine_similarity(protected_feature[i], target_feature[i].detach())
            obf = 1 - cosine_similarity(protected_feature[i], source_feature[i].detach())
            loss = imp - obf
        cos_loss_list.append(loss)
    return torch.sum(torch.stack(cos_loss_list))


def get_target_audio(target_path, device):
    audio, sr = torchaudio.load(target_path)

    if audio.shape[0] > 1:
        audio = torch.mean(audio, dim=0, keepdim=True)  # shape: [1, num_samples]

    if sr != 16000:
        audio = torchaudio.transforms.Resample(sr, 16000)(audio)

    return audio, None

def save_audio(audio_tensor, out_dir, name):
    os.makedirs(out_dir, exist_ok=True)

    if not name.endswith(('.wav', '.mp3', '.flac', '.ogg')):
        name = name + '.wav'  # Default to .wav if no extension is provided

    path = os.path.join(out_dir, name)
    print('perturbed audio', audio_tensor.shape)

    audio_tensor = audio_tensor.to(torch.float32)

    torchaudio.save(path, audio_tensor.cpu(), 16000)


class AdversarialAudioOpt:
    def __init__(self, args, model):
        self.device = args.device
        self.diff_model = model  # e.g., AudioLDM UNet
        self.diff_model.vae.requires_grad_(False)
        self.diff_model.text_encoder.requires_grad_(False)
        self.diff_model.unet.requires_grad_(False)
        self.diff_model.vae.eval()

        self.source_dir = args.source_dir
        self.protected_audio_dir = args.protected_audio_dir
        self.comparison_null_text = args.comparison_null_text

        self.image_size = args.image_size
        self.prot_steps = args.prot_steps
        self.diffusion_steps = args.diffusion_steps
        self.start_step = args.start_step
        self.null_optimization_steps = args.null_optimization_steps

        self.dataloader = args.dataloader

        self.adv_optim_weight = args.adv_optim_weight
        self.is_obfuscation = args.is_obfuscation
        # self.cosine_loss = CosineLoss(self.is_obfuscation)

        # set up SV models
        # self.surrogate_models = load_SV_models(args, args.surrogate_model_names)
        self.surrogate_models = {'ecapa': args.surrogate_models[0]} # change later with other surrogate models

        self.test_model_name = args.test_model_name
        self.target_choice = args.target_choice
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.diff_model.vocoder.config.sampling_rate,
            n_fft=1024,
            hop_length=256,
            n_mels=self.diff_model.vocoder.config.model_in_dim,
        ).to(self.device)

    def audio2latent(self, audio):
        mel = self.mel_transform(audio)

        # convert to log mel
        log_mel = torch.log(mel + 1e-6)
        log_mel = log_mel.unsqueeze(0)
        log_mel = log_mel.to(dtype=self.diff_model.vae.dtype)

        # encode to latent space
        with torch.no_grad():
            latents = self.diff_model.vae.encode(log_mel).latent_dist.sample()
            latents = latents * self.diff_model.vae.config.scaling_factor

        return latents


    def latent2audio(self, latents):
        with torch.no_grad():
            mel = self.diff_model.vae.decode(latents).sample
            mel = mel.squeeze(1)

            if mel.shape[1] == 64:
                mel = mel.transpose(1, 2)

            generated_audio = self.diff_model.vocoder(mel)

            return generated_audio

    def get_SV_embeddings(self, audio):
        # returns features from sv models: [batch, 1, feature_dimension]
        features = []
        audio = audio.to(self.device)

        if audio.ndim == 3 and audio.shape[1] > 1:
          audio = torch.mean(audio, dim=1, keepdim=True)

        for model_name in self.surrogate_models.keys():
            sv_model = self.surrogate_models[model_name]

            for param in sv_model.parameters():
                param.requires_grad = False 

            emb = sv_model.encode_batch(audio)  # assume it returns [B, D]

            if emb.ndim == 1:
              emb = emb.unsqueeze(0)
            features.append(emb)
        return features
    
    # add classifier-free guidance 
    def diffusion_step(self, latent, prompt_embeds, attention_mask, generated_prompt_embeds, t):
        latent_input = latent 
        
        noise_pred = self.diff_model.unet(
            latent_input, 
            t, 
            encoder_hidden_states=generated_prompt_embeds, 
            encoder_hidden_states_1=prompt_embeds, 
            encoder_attention_mask_1=attention_mask,
            return_dict=False,
        )[0]

        return self.diff_model.scheduler.step(noise_pred, t, latent)['prev_sample']

    def null_embeddings(self, prompt=None, transcription=None, return_gpt=False):
        # Setup logger
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        logger.addHandler(logging.StreamHandler())

        tokenizers = [self.diff_model.tokenizer, self.diff_model.tokenizer_2]
        batch_size = 1 
        is_vits_text_encoder = isinstance(self.diff_model.text_encoder_2, transformers.VitsModel)

        if is_vits_text_encoder: 
            text_encoders = [self.diff_model.text_encoder, self.diff_model.text_encoder_2.text_encoder]
        else: 
            text_encoders = [self.diff_model.text_encoder, self.diff_model.text_encoder_2]

        # prompt = 'Person speaking'
        if prompt == None: 
            prompt = '' 
            null_prompt = True 
        else: 
            null_prompt = False 

        if transcription == None: 
            transcription = '<unk>' # will be rewritten by null embeddings, good to pass smth 
            null_trans = True 
        else: 
            null_trans = False 
        

        max_new_tokens = None # change to 8 if necessary 
        prompt_embeds_list = []
        attention_mask_list = []

        for tokenizer, text_encoder in zip(tokenizers, text_encoders):
            use_prompt = isinstance(tokenizer, (transformers.RobertaTokenizer, transformers.RobertaTokenizerFast, transformers.T5Tokenizer, transformers.T5TokenizerFast)) 
            text_inputs = tokenizer(
                prompt if use_prompt else transcription, 
                padding='max_length' 
                if isinstance(tokenizer, (transformers.RobertaTokenizer, transformers.RobertaTokenizerFast, transformers.VitsTokenizer))
                else True, 
                max_length=tokenizer.model_max_length,
                truncation=True,  
                return_tensors='pt',         
                )
            
            text_input_ids = text_inputs.input_ids
            attention_mask = text_inputs.attention_mask 
            untruncated_ids = tokenizer(prompt, padding='longest', return_tensors='pt').input_ids 

            if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(
                text_input_ids, untruncated_ids
            ):
                removed_text = tokenizer.batch_decode(untruncated_ids[:, tokenizer.model_max_length - 1 : -1])
                logger.warning(
                    f"The following part of your input was truncated because {text_encoder.config.model_type} can "
                    f"only handle sequences up to {tokenizer.model_max_length} tokens: {removed_text}"
                )

            text_input_ids = text_input_ids.to(self.device)
            attention_mask = attention_mask.to(self.device)

            if text_encoder.config.model_type == 'clap': 
                prompt_embeds = text_encoder.get_text_features(
                    text_input_ids,
                    attention_mask=attention_mask,
                )
                # append the seq-len dim: (bs, hidden_size) -> (bs, seq_len, hidden_size)
                prompt_embeds = prompt_embeds[:, None, :]
                # make sure that we attend to this single hidden-state
                attention_mask = attention_mask.new_ones((batch_size, 1))

            elif is_vits_text_encoder: 
                for text_input_id, text_attention_mask in zip(text_input_ids, attention_mask):
                    for idx, phoneme_id in enumerate(text_input_id): 
                        if phoneme_id == 0: 
                            text_input_id[idx] = 182 
                            text_attention_mask[idx] = 1 
                            break 
                        
                prompt_embeds = text_encoder(text_input_ids, attention_mask=attention_mask, padding_mask=attention_mask.unsqueeze(-1))
                prompt_embeds = prompt_embeds[0]

            else: 
                prompt_embeds = text_encoder(
                    text_input_ids,
                    attention_mask=attention_mask,
                )
                prompt_embeds = prompt_embeds[0]
                

            prompt_embeds_list.append(prompt_embeds)
            attention_mask_list.append(attention_mask)

        # zero_embeddings = prompt_embeds_list[0]
        if null_prompt and not null_trans: 
            zero_embeddings = torch.zeros_like(prompt_embeds_list[0])
            prompt_embeds_list[0] = zero_embeddings

            projection_output = self.diff_model.projection_model(
                    # hidden_states=prompt_embeds_list[0], # prompt 
                    hidden_states=zero_embeddings, 
                    hidden_states_1=prompt_embeds_list[1], # transcription 
                    attention_mask=attention_mask_list[0],
                    attention_mask_1=attention_mask_list[1],
            )

        elif not null_prompt and not null_trans: 
            projection_output = self.diff_model.projection_model(
                    hidden_states=prompt_embeds_list[0], # prompt 
                    hidden_states_1=prompt_embeds_list[1], # transcription 
                    attention_mask=attention_mask_list[0],
                    attention_mask_1=attention_mask_list[1],
            )

        else: 
            projection_output = self.diff_model.projection_model(
                    hidden_states=torch.zeros_like(prompt_embeds_list[0]), 
                    hidden_states_1=torch.zeros_like(prompt_embeds_list[1]), # transcription 
                    attention_mask=attention_mask_list[0],
                    attention_mask_1=attention_mask_list[1],
            )

            prompt_embeds_list[0] = torch.zeros_like(prompt_embeds_list[0])
            prompt_embeds_list[1] = torch.zeros_like(prompt_embeds_list[1])

        projected_prompt_embeds = projection_output.hidden_states
        projected_attention_mask = projection_output.attention_mask

        generated_prompt_embeds = self.diff_model.generate_language_model(
            projected_prompt_embeds,
            attention_mask=projected_attention_mask,
            max_new_tokens=max_new_tokens,
        )

        prompt_embeds = prompt_embeds.to(dtype=self.diff_model.text_encoder_2.dtype, device=self.device)
        attention_mask = (
            attention_mask.to(device=self.device)
            if attention_mask is not None
            else torch.ones(prompt_embeds.shape[:2], dtype=torch.long, device=self.device)
        )
        generated_prompt_embeds = generated_prompt_embeds.to(dtype=self.diff_model.language_model.dtype, device=self.device)

        if return_gpt: 
            return prompt_embeds, attention_mask, generated_prompt_embeds
        else: 
            return prompt_embeds_list[0], attention_mask_list[0], prompt_embeds_list[1], attention_mask_list[1]

    @torch.no_grad()
    def ddim_inversion(self, audio, prompt=None, transcription=None):
        prompt_embeds, attention_mask, generated_prompt_embeds = self.null_embeddings(prompt=prompt, transcription=transcription, return_gpt=True)
        self.diff_model.scheduler.set_timesteps(self.diffusion_steps)

        latents = self.audio2latent(audio)
        all_latents = [latents]
    
        self.diff_model.scheduler.set_timesteps(self.diffusion_steps)

        for i in tqdm(range(self.diffusion_steps - 1)): 
            t = self.diff_model.scheduler.timesteps[self.diffusion_steps - i - 1]
            with torch.no_grad(): 
                noise_pred = self.diff_model.unet(
                    latents,
                    t, 
                    encoder_hidden_states=generated_prompt_embeds, 
                    encoder_hidden_states_1=prompt_embeds, 
                    encoder_attention_mask_1=attention_mask, 
                    return_dict=False,  
                )[0]

                next_timestep = t + self.diff_model.scheduler.config.num_train_timesteps // self.diff_model.scheduler.num_inference_steps
                alpha_bar_next = self.diff_model.scheduler.alphas_cumprod[next_timestep] \
                    if next_timestep <= self.diff_model.scheduler.config.num_train_timesteps else torch.tensor(0.0)
                reverse_x0 = (1 / torch.sqrt(self.diff_model.scheduler.alphas_cumprod[t]) * (
                    latents - noise_pred * torch.sqrt(1 - self.diff_model.scheduler.alphas_cumprod[t])))
                latents = reverse_x0 * \
                    torch.sqrt(alpha_bar_next) + \
                    torch.sqrt(1 - alpha_bar_next) * noise_pred

                all_latents.append(latents)

        return all_latents 


    def null_optimization(self, inversion_latents, prompt=None, transcription=None):
        all_uncond_embs = []
        latent = inversion_latents[self.start_step - 1]
        max_new_tokens = None 

        prompt_embeds, attention_mask, transcription_embeds, attention_mask_1 = self.null_embeddings(prompt=prompt, transcription=transcription, return_gpt=False) 

        prompt_embeds = prompt_embeds.detach().clone().to(torch.float64).requires_grad_(True)

        optimizer = optim.AdamW([prompt_embeds], lr=1e-1)
        criterion = torch.nn.MSELoss()

        for i in tqdm(range(self.start_step, self.diffusion_steps)):
            t = self.diff_model.scheduler.timesteps[i]
            for _ in range(self.null_optimization_steps):
                projection_output = self.diff_model.projection_model(
                    hidden_states=prompt_embeds.to(self.diff_model.text_encoder_2.dtype), 
                    hidden_states_1=transcription_embeds.detach(), 
                    attention_mask=attention_mask, 
                    attention_mask_1=attention_mask_1, 
                )
                
                projected_prompt_embeds = projection_output.hidden_states
                projected_attention_mask = projection_output.attention_mask

                generated_prompt_embeds = self.diff_model.generate_language_model(
                    projected_prompt_embeds,
                    attention_mask=projected_attention_mask,
                    max_new_tokens=max_new_tokens,
                )

                transcription_embeds = transcription_embeds.to(dtype=self.diff_model.text_encoder_2.dtype, device=self.device)
                attention_mask_1 = (
                    attention_mask_1.to(device=self.device)
                    if attention_mask_1 is not None
                    else torch.ones(transcription_embeds.shape[:2], dtype=torch.long, device=self.device)
                )
                generated_prompt_embeds = generated_prompt_embeds.to(dtype=self.diff_model.language_model.dtype, device=self.device)

                out_latent = self.diffusion_step(latent, transcription_embeds, attention_mask_1, generated_prompt_embeds, t) 
                optimizer.zero_grad()

                loss = criterion(out_latent, inversion_latents[i])
                loss.backward()
                optimizer.step()

                torch.cuda.empty_cache()
                gc.collect()

            with torch.no_grad():
                latent = self.diffusion_step(latent, transcription_embeds, attention_mask_1, generated_prompt_embeds, t, True).detach()

                all_uncond_embs.append(( 
                    transcription_embeds.detach().clone(), 
                    attention_mask_1.detach().clone(), 
                    generated_prompt_embeds.detach().clone()
                ))

        prompt_embeds.requires_grad_(False)
        # generated_prompt_embeds.requires_grad_(False)
        # return prompt_embeds, attention_mask, transcription_embeds, attention_mask_1
        return all_uncond_embs 


    def attacker(self, audio, prompt=None, transcription=None, source_embeddings=None, target_embeddings=None):
        all_uncond_embs = self.null_optimization(inversion_latents)

        tr_guidance = [all_uncond_embs[i][0].detach().repeat(2, 1, 1).to(torch.float32) for i in range(len(all_uncond_embs))]
        amask_guidance = [all_uncond_embs[i][1].detach().repeat(2, 1) for i in range(len(all_uncond_embs))]
        gen_guidance = [all_uncond_embs[i][2].detach().repeat(2, 1, 1).to(torch.float32) for i in range(len(all_uncond_embs))]
        
        # lat[0], lat[1], lat[2], ...
        inversion_latents = self.ddim_inversion(audio, prompt, transcription)[::-1]
        # reverse
        latent = inversion_latents[self.start_step - 1].detach().to(torch.float32)

        init_latent = latent.detach().clone()
        latent.requires_grad_(True)
        optimizer = optim.AdamW([latent], lr=1e-2)

        for _ in tqdm(range(self.prot_steps)):
            latents = torch.cat([init_latent, latent])
            for i in range(self.start_step, self.diffusion_steps):
                t = self.diff_model.scheduler.timesteps[i]
                latents = self.diffusion_step(latents.to(self.diff_model.dtype), tr_guidance[i-self.start_step].to(self.diff_model.dtype), 
                                 amask_guidance[i-self.start_step], gen_guidance[i-self.start_step].to(self.diff_model.dtype), t)

            mel = self.diff_model.vae.decode(latents).sample
            mel = mel.squeeze(1)

            if mel.shape[1] == 64: 
                mel = mel.transpose(1, 2)

            generated_audio = self.diff_model.vocoder(mel)

            output_embeddings = self.get_SV_embeddings(generated_audio)

            adv_loss = cosine_loss(output_embeddings, target_embeddings, source_embeddings, self.is_obfuscation) * self.adv_optim_weight 
            print('adv_loss: ', adv_loss.item())
            
            optimizer.zero_grad()
            adv_loss.backward()
            optimizer.step()

            torch.cuda.empty_cache()
            gc.collect()

        with torch.no_grad():
            latents = torch.cat([init_latent, latent])
            for i in range(self.start_step, self.diffusion_steps):
                t = self.diff_model.scheduler.timesteps[i]
                latents = self.diffusion_step(latents.to(self.diff_model.dtype), tr_guidance[i-self.start_step].to(self.diff_model.dtype), 
                                 amask_guidance[i-self.start_step], gen_guidance[i-self.start_step].to(self.diff_model.dtype), t)

        return latents.detach()

    def run(self):
        # target_choice should be a path to target audio
        target_audio, _ = get_target_audio(self.target_choice, self.device)

        with torch.no_grad():
            target_embeddings = self.get_SV_embeddings(target_audio)


        for fname, audio, transcription in self.dataloader:
            audio_name = fname[0]
            audio = audio.to(self.device)

            if audio.shape[1] > 1:
              audio = torch.mean(audio, dim=1, keepdim=True)

            if audio.shape[1] == 1:
                audio = audio.squeeze(1)

            if self.is_obfuscation:
                with torch.no_grad():
                    source_embeddings = self.get_SV_embeddings(audio)
            else:
                source_embeddings = None

            latents = self.attacker(audio, prompt=None, transcription=transcription, 
                                    source_embeddings=source_embeddings, target_embeddings=target_embeddings)

            if latents is not None:
                protected_audio = self.latent2audio(latents)[1:]
                visualize_spec(protected_audio.to(torch.float32).cpu().detach())
                # protected_audio = self.latent2audio(latents)
                save_audio(protected_audio, self.protected_audio_dir, audio_name)
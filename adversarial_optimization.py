import torch
from torch import nn, optim
import torchaudio
from tqdm import tqdm

from torchvision import transforms
import numpy as np
import pandas as pd
import os
import gc

class CosineLoss(torch.nn.Module):
    def __init__(self, is_obfuscation):
        super(CosineLoss, self).__init__()
        self.is_obfuscation = is_obfuscation

    def cos_simi(self, emb_1, emb_2):
        # if emb_1.dim() == 1:
        #     emb_1 = emb_1.unsqueeze(0)
        # if emb_2.dim() == 1:
        #     emb_2 = emb_2.unsqueeze(0)

        emb_1 = emb_1.view(1, -1) if emb_1.ndim == 1 else emb_1
        emb_2 = emb_2.view(1, -1) if emb_2.ndim == 1 else emb_2
        dot = torch.sum(emb_1 * emb_2, dim=1)
        norm1 = emb_1.norm(dim=1)
        norm2 = emb_2.norm(dim=1)
        return torch.mean(dot / (norm1 * norm2 + 1e-8))  # eps for stability
        # return torch.mean(torch.sum(torch.mul(emb_2, emb_1), dim=1) / emb_2.norm(dim=1) / emb_1.norm(dim=1))

    def forward(self, protected_feature, target_feature, source_feature=None):
      cos_loss_list = []
      for i in range(len(protected_feature)):
          if not self.is_obfuscation:
              cos_loss_list.append(
                  1 - self.cos_simi(protected_feature[i], target_feature[i].detach()))
          else:
              imp = 1 - self.cos_simi(protected_feature[i], target_feature[i].detach())
              obf = 1 - self.cos_simi(protected_feature[i], source_feature[i].detach())
              cos_loss_list.append(imp - obf)
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
        # self.makeup_weight = args.makeup_weight

        # self.sv_model = ECAPA_TDNN(pretrained=True).to(self.device)
        # self.sv_model.eval()

        self.cosine_loss = CosineLoss(self.is_obfuscation)
        # self.clap_loss = CLAPLoss()  # if using content-preserving loss (optional)

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
        # # resample if necessary
        # sr = 48000
        # if sr != self.diff_model.vocoder.config.sampling_rate:
        #     resampler = torch.audio.transforms.Resample(orig_freq=sr, new_freq=self.diff_model.vocoder.config.sampling_rate)
        #     audio = resampler(audio)

        # convert audio to mel
        mel = self.mel_transform(audio)

        # convert to log mel
        log_mel = torch.log(mel + 1e-6)
        log_mel = log_mel.unsqueeze(0)
        log_mel = log_mel.to(dtype=self.diff_model.vae.dtype)

        # encode to latent space
        with torch.no_grad():
            latents = self.diff_model.vae.encode(log_mel).latent_dist.sample()
            latents = latents * self.diff_model.vae.scaling_factor

        return latents

    def latent2audio(self, latents):
        latents = latents / self.diff_model.vae.scaling_factor

        vocoder_upsample_factor = np.prod(self.diff_model.vocoder.config.upsample_rates) / self.diff_model.vocoder.config.sampling_rate
        audio_length_in_s = self.diff_model.unet.config.sample_size * self.diff_model.vae_scale_factor * vocoder_upsample_factor
        height = int(audio_length_in_s / vocoder_upsample_factor)

        batch_size = latents.shape[0]
        num_waveforms_per_prompt = 1 # set 1 for now
        generator = torch.Generator().manual_seed(8888)
        num_channels_latents = self.diff_model.unet.config.in_channels

        latents = self.diff_model.prepare_latents(
            batch_size * num_waveforms_per_prompt,
            num_channels_latents,
            height,
            self.diff_model.dtype,
            self.diff_model.device,
            generator,
            latents
        )

        with torch.no_grad():
            mel = self.diff_model.vae.decode(latents).sample
            mel = mel.squeeze(1)

            if mel.shape[1] == 64:
                mel = mel.transpose(1, 2)

            generated_audio = self.diff_model.vocoder(mel)

            return generated_audio

    # def get_speaker_embedding(self, audio):
    #     with torch.no_grad():
    #         return self.sv_model(audio)

    def get_SV_embeddings(self, audio):
        # returns features from sv models: [batch, 1, feature_dimension]

        features = []
        audio = audio.to(self.device)

        # if audio.shape[0] > 1:
        #   audio = torch.mean(audio, dim=0, keepdim=True)  # shape: [1, num_samples]


        if audio.ndim == 3 and audio.shape[1] > 1:
          audio = torch.mean(audio, dim=1, keepdim=True)

        for model_name in self.surrogate_models.keys():
            sv_model = self.surrogate_models[model_name]

            with torch.no_grad():
              emb = sv_model.encode_batch(audio)  # assume it returns [B, D]

            # Re-enable gradients for the computed embedding
            emb = emb.detach().clone().requires_grad_(True)
            if emb.ndim == 1:
              emb = emb.unsqueeze(0)
            features.append(emb)
        return features


    # def diffusion_step(self, latents, t, context=None):
    #     noise_pred = self.diff_model.unet(latents, t, context)["sample"]
    #     return self.scheduler.step(noise_pred, t, latents)["prev_sample"]

    def diffusion_step(self, latent, null_context, t, is_null_optimization=False):
        # audioldm unet expects:
        # latent_input, t, encoder_hidden_states (generated_prompt_embeds)
        # encoder_hidden_states_1 (prompt_embeds), encoder_attention_mask_1 (attention_mask)
        # return_dict (False)
        prompt_embeds, attention_mask, generated_prompt_embeds = null_context

        if not is_null_optimization:
            latent_input = torch.cat([latent] * 2)
            # latent_input = self.diff_model.scheduler.scale_model_input(latent_input, t)

            # print(f"[DEBUG] latent_input shape: {latent_input.shape}")
            # print(f"[DEBUG] prompt_embeds shape: {prompt_embeds.shape}")
            # print(f"[DEBUG] attention_mask shape: {attention_mask.shape}")
            # print(f"[DEBUG] generated_prompt_embeds shape: {generated_prompt_embeds.shape}")


            noise_pred = self.diff_model.unet(
                latent_input,
                t,
                encoder_hidden_states=generated_prompt_embeds,
                encoder_hidden_states_1=prompt_embeds,
                encoder_attention_mask_1=attention_mask,
                return_dict=False,
            )[0]

            noise_pred, _ = noise_pred.chunk(2)
        else:
            latent_input = latent
            noise_pred = self.diff_model.unet(
                latent_input,
                t,
                encoder_hidden_states=generated_prompt_embeds,
                encoder_hidden_states_1=prompt_embeds,
                encoder_attention_mask_1=attention_mask,
                return_dict=False,
            )[0]

        # return previous noisy sample x_t -> x_t-1
        return self.diff_model.scheduler.step(noise_pred, t, latent)["prev_sample"]

    def null_embeddings(self):
        # add batch_size later
        prompt_embeds, attention_mask, generated_prompt_embeds = self.diff_model.encode_prompt(
            prompt='',
            num_waveforms_per_prompt=1,
            transcription=[''],
            device=self.diff_model.device,
            do_classifier_free_guidance=False,
            # max_new_tokens=64 # remove
        )

        # null_text = [""]  # null prompt
        # input_ids = self.diff_model.text_encoder.tokenize(null_text).to(self.device)
        # return self.diff_model.text_encoder(input_ids) # add [0]

        return prompt_embeds, attention_mask, generated_prompt_embeds

    @torch.no_grad()
    def ddim_inversion(self, audio):
        uncond_embeddings = self.null_embeddings()
        self.diff_model.scheduler.set_timesteps(self.diffusion_steps)

        latent = self.audio2latent(audio)
        # maybe add prepare_latents() ?
        all_latents = [latent]

        prompt_embeds, attention_mask, generated_prompt_embeds = uncond_embeddings

        for i in tqdm(range(self.diffusion_steps - 1)):
            t = self.diff_model.scheduler.timesteps[self.diffusion_steps - i - 1]
            noise_pred = self.diff_model.unet(
                latent,
                t,
                encoder_hidden_states=generated_prompt_embeds,
                encoder_hidden_states_1=prompt_embeds,
                encoder_attention_mask_1=attention_mask,
                return_dict=False,
            )[0]
            latent = self.diff_model.scheduler.step(noise_pred, t, latent)["prev_sample"]
            all_latents.append(latent)
        return all_latents


    def null_optimization(self, inversion_latents):
        all_uncond_embs = []
        latent = inversion_latents[self.start_step - 1]

        uncond_embeddings = self.null_embeddings()
        prompt_embeds, attention_mask, generated_prompt_embeds = uncond_embeddings


        # prompt_embeds.requires_grad_(True)
        # generated_prompt_embeds.requires_grad_(True)
        prompt_embeds = prompt_embeds.detach().clone().requires_grad_()
        generated_prompt_embeds = generated_prompt_embeds.detach().clone().requires_grad_()


        optimizer = optim.AdamW([prompt_embeds, generated_prompt_embeds], lr=1e-1)
        criterion = torch.nn.MSELoss()

        for i in tqdm(range(self.start_step, self.diffusion_steps)):
            t = self.diff_model.scheduler.timesteps[i]
            for _ in range(self.null_optimization_steps):
                uncond_embeddings = (prompt_embeds, attention_mask, generated_prompt_embeds)

                out_latent = self.diffusion_step(latent, uncond_embeddings, t, True)
                optimizer.zero_grad()
                loss = criterion(out_latent, inversion_latents[i])
                loss.backward()
                optimizer.step()

                torch.cuda.empty_cache()
                gc.collect()

            with torch.no_grad():
                uncond_embeddings = (prompt_embeds, attention_mask, generated_prompt_embeds)
                latent = self.diffusion_step(latent, uncond_embeddings, t, True).detach()
                # all_uncond_embs.append(uncond_embeddings.detach().clone())
                all_uncond_embs.append((
                    prompt_embeds.detach().clone(),
                    attention_mask.clone(),  # assuming attention_mask is not updated
                    generated_prompt_embeds.detach().clone()
                ))

        # uncond_embeddings.requires_grad_(False)
        prompt_embeds.requires_grad_(False)
        generated_prompt_embeds.requires_grad_(False)
        return all_uncond_embs


    def attacker(self, audio, audio_name, source_embeddings, target_embeddings):
        # lat[0], lat[1], lat[2], ...
        inversion_latents = self.ddim_inversion(audio)[::-1]
        # reverse
        latent = inversion_latents[self.start_step - 1]

        # null_optimization returns list of (prompt embeds, gpt_hidden_states, and attention_mask)
        all_uncond_embs = self.null_optimization(inversion_latents)

        # null_context_guidance = [
        #     (
        #         torch.cat([all_uncond_embs[i][0], all_uncond_embs[i][0]]),  # prompt_embeds * 2
        #         all_uncond_embs[i][1],                                      # attention_mask
        #         torch.cat([all_uncond_embs[i][2], all_uncond_embs[i][2]])   # generated_prompt_embeds * 2
        #     )
        #     for i in range(len(all_uncond_embs))
        # ]

        null_context_guidance = [
            (
                torch.cat([all_uncond_embs[i][0], all_uncond_embs[i][0], all_uncond_embs[i][0], all_uncond_embs[i][0]]),  # prompt_embeds * 2
                torch.cat([all_uncond_embs[i][1], all_uncond_embs[i][1], all_uncond_embs[i][1], all_uncond_embs[i][1]]),  # ❗ attention_mask * 2
                torch.cat([all_uncond_embs[i][2], all_uncond_embs[i][2], all_uncond_embs[i][2], all_uncond_embs[i][2]])   # generated_prompt_embeds * 2
            )
            for i in range(len(all_uncond_embs))
        ]


        init_latent = latent.detach().clone()
        latent.requires_grad_(True)
        optimizer = optim.AdamW([latent], lr=1e-2)

        for _ in tqdm(range(self.prot_steps)):
            latents = torch.cat([init_latent, latent])
            for i in range(self.start_step, self.diffusion_steps):
                t = self.diff_model.scheduler.timesteps[i]
                # prompt_embeds, attention_mask, generated_prompt_embeds = null_context_guidance[i - self.start_step]
                latents = self.diffusion_step(latents, null_context_guidance[i - self.start_step], t)

            out_audio = self.latent2audio(latents)[1:]  # take perturbed one
            output_embeddings = self.get_SV_embeddings(out_audio)

            # cosine loss with +1 target
            # loss = sum(
            #     self.cosine_loss(output, target, torch.ones((target.shape[0], ), device=target.device))
            #     for output, target in zip(output_embeddings, target_embeddings)
            # ) * self.adv_optim_weight # was torch.ones_line(target[:, 0])
            if self.is_obfuscation:
              loss = sum(
                  self.cosine_loss(output, target, source)
                  for output, target, source in zip(output_embeddings, target_embeddings, source_embeddings)
              )
            else:
                loss = sum(
                    self.cosine_loss(output, target, None)
                    for output, target in zip(output_embeddings, target_embeddings)
                )


            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            torch.cuda.empty_cache()
            gc.collect()

        with torch.no_grad():
            latents = torch.cat([init_latent, latent])
            for i in range(self.start_step, self.diffusion_steps):
                t = self.diff_model.scheduler.timesteps[i]
                latents = self.diffusion_step(latents, null_context_guidance[i - self.start_step], t)

        return latents.detach()

    def run(self):
        # target_choice should be a path to target audio
        target_audio, _ = get_target_audio(self.target_choice, self.device)

        with torch.no_grad():
            target_embeddings = self.get_SV_embeddings(target_audio)


        # print(target_embeddings[0].shape)

        for fname, audio in self.dataloader:

            audio_name = fname[0]
            audio = audio.to(self.device)

            if audio.shape[1] > 1:
              audio = torch.mean(audio, dim=1, keepdim=True)

            if audio.shape[1] == 1:
                audio = audio.squeeze(1)
            print(audio.shape)

            if self.is_obfuscation:
                with torch.no_grad():
                    source_embeddings = self.get_SV_embeddings(audio)
            else:
                source_embeddings = None

            latents = self.attacker(audio, audio_name, source_embeddings, target_embeddings)

            if latents is not None:
                protected_audio = self.latent2audio(latents)[1:]
                # protected_audio = self.latent2audio(latents)
                save_audio(protected_audio, self.protected_audio_dir, audio_name)
# Diffusion-Based Adversarial Attacks for Speaker Verification using AudioLDM2 

This project implements a diffusion-based framework for generating adversarial audio examples designed to fool speaker verification systems such as ECAPA, X-Vector, and ResNet-TDNN. It includes per audio adversarial attack generation (test.py) and evaluation code (eval_new.py) for assessing attack success.

## Install dependencies:

<pre>pip install -r requirements.txt</pre>

## Generating Adversarial Examples 

To generate adversarial audio examples using diffusion and null-text optimization:

<pre> python test.py </pre>

This will run the attack pipeline and save generated adversarial examples to adversarial_examples/.

## Evaluating the Attack 

Once adversarial examples are generated, run evaluation using: 

<pre> 
python eval_new.py \
  --data_root wav \
  --adv_dir adversarial_examples \
  --transcription_file transcriptions.tsv \
  --output_dir evaluation_results \
  --model_name ecapa \
  --attack_type impersonation
</pre>

Arguments:

- `--data_root`: Path to clean/original audio files.
- `--adv_dir`: Path to adversarial examples.
- `--transcription_file`: TSV file with transcription and audio paths.
- `--model_name`: Model to evaluate on (`ecapa`, `xvect`, or `resnet`).
- `--attack_type`: Either `impersonation` or `evasion`.



Code adapted from: https://github.com/parham1998/Facial-Privacy-Protection, https://huggingface.co/docs/diffusers/main/api/pipelines/audioldm2

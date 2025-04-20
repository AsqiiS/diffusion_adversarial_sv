#!/usr/bin/env python3
"""
PGD Attack for SpeechBrain Speaker Verification Models
This script implements a Projected Gradient Descent attack against SpeechBrain
speaker verification models. Supports X-Vector, ResNet, and ECAPA-TDNN.
"""

import torch
import torchaudio
import numpy as np
import os
import argparse
from tqdm import tqdm
from speechbrain.inference import EncoderClassifier
from scipy.spatial.distance import cosine

# Model hub paths for supported models
MODEL_PATHS = {
    "xvector": "speechbrain/spkrec-xvect-voxceleb",
    "resnet": "speechbrain/spkrec-resnet-voxceleb",
    "ecapa": "speechbrain/spkrec-ecapa-voxceleb"
}

class PGDAttack:
    def __init__(self, model, eps=0.002, alpha=0.0004, steps=100, device="cuda", debug=False):
        """
        Initialize PGD Attack
        
        Args:
            model: SpeechBrain EncoderClassifier model
            eps: Maximum perturbation size (epsilon)
            alpha: Step size for each iteration
            steps: Number of iterations
            device: Device to run the attack on
            debug: Enable debug prints
        """
        self.model = model
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.device = device
        self.debug = debug
    
    def compute_embeddings(self, wav_tensor):
        """Extract embeddings from the model"""
        with torch.no_grad():
            embeddings = self.model.encode_batch(wav_tensor)
            if len(embeddings.shape) > 2:
                embeddings = embeddings.squeeze(1)
            return embeddings
    
    def compute_similarity(self, emb1, emb2):
        """Compute cosine similarity between embeddings"""
        # Ensure embeddings are flattened to 1-D for cosine distance calculation
        emb1 = emb1.cpu().numpy().flatten()
        emb2 = emb2.cpu().numpy().flatten()
        return 1 - cosine(emb1, emb2)
    
    def generate_adversarial(self, source_wav, target_wav=None, targeted=False):
        """
        Generate adversarial example using PGD
        
        Args:
            source_wav: Source waveform tensor
            target_wav: Target waveform tensor (for targeted attacks)
            targeted: If True, tries to make source_wav similar to target_wav
                      If False, tries to maximize distance from original embedding
        
        Returns:
            Adversarial waveform tensor
        """
        # Clone the source waveform to avoid modifying the original
        adv_wav = source_wav.clone().to(self.device)
        
        # Compute original embeddings
        with torch.no_grad():
            orig_embedding = self.model.encode_batch(source_wav.to(self.device))
            if len(orig_embedding.shape) > 2:
                orig_embedding = orig_embedding.squeeze(1)
        
        # For targeted attacks, compute target embeddings
        if targeted and target_wav is not None:
            with torch.no_grad():
                target_embedding = self.model.encode_batch(target_wav.to(self.device))
                if len(target_embedding.shape) > 2:
                    target_embedding = target_embedding.squeeze(1)
        
        # Enable gradients for the input
        adv_wav.requires_grad = True
        
        # PGD attack loop
        for step in range(self.steps):
            # Zero gradients
            if adv_wav.grad is not None:
                adv_wav.grad.zero_()
            
            # Forward pass - DIRECTLY USING model.encode_batch
            adv_embedding = self.model.encode_batch(adv_wav)
            if len(adv_embedding.shape) > 2:
                adv_embedding = adv_embedding.squeeze(1)
            
            # Compute loss based on attack type
            if targeted and target_wav is not None:
                # For targeted attack, minimize distance to target
                loss = -torch.nn.functional.cosine_similarity(adv_embedding, target_embedding, dim=1).mean()
            else:
                # For untargeted attack, maximize distance from original
                loss = torch.nn.functional.cosine_similarity(adv_embedding, orig_embedding, dim=1).mean()
            
            # Print diagnostics
            if self.debug and step % 10 == 0:
                print(f"Step {step}, Loss: {loss.item():.4f}")
            
            # Backward pass
            loss.backward()
            
            # Check for gradient issues
            if adv_wav.grad is None or torch.isnan(adv_wav.grad).any() or torch.isinf(adv_wav.grad).any():
                if self.debug:
                    print(f"Warning: Invalid gradients at step {step}")
                # Create random gradient if needed
                if adv_wav.grad is None:
                    adv_wav.grad = torch.randn_like(adv_wav)
            
            # Update with gradient step
            with torch.no_grad():
                grad_sign = adv_wav.grad.sign()
                adv_wav = adv_wav - self.alpha * grad_sign
                
                # Project back onto epsilon ball around original sample
                delta = adv_wav - source_wav.to(self.device)
                delta = torch.clamp(delta, -self.eps, self.eps)
                adv_wav = source_wav.to(self.device) + delta
                
                # Ensure values remain in valid audio range [-1, 1]
                adv_wav = torch.clamp(adv_wav, -1.0, 1.0)
                adv_wav.requires_grad = True
        
        return adv_wav.detach()

def load_audio(file_path, audio_dir="data/wav", sample_rate=16000):
    """Load audio file and resample if necessary"""
    # Handle path construction - allow for both relative and absolute paths
    if not os.path.isabs(file_path):
        # If path already contains audio_dir, don't add it again
        if not file_path.startswith(audio_dir):
            file_path = os.path.join(audio_dir, file_path)
    
    # Check if file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Audio file not found: {file_path}")
        
    # Load and resample audio
    waveform, sr = torchaudio.load(file_path)
    if sr != sample_rate:
        waveform = torchaudio.functional.resample(waveform, sr, sample_rate)
    return waveform

def process_embeddings(embeddings):
    """Process embeddings based on shape, ensuring they're flattened properly"""
    if len(embeddings.shape) > 2:
        embeddings = embeddings.squeeze(1)
    return embeddings.cpu().numpy().flatten()

def evaluate_model(model, pairs_file, audio_dir="data/wav", device="cuda"):
    """Evaluate model accuracy on verification pairs"""
    correct = 0
    total = 0
    
    with open(pairs_file, 'r') as f:
        pairs = f.readlines()
    
    for pair in tqdm(pairs, desc="Evaluating baseline model"):
        parts = pair.strip().split()
        label = int(parts[0])
        file1 = parts[1]
        file2 = parts[2]
        
        try:
            wav1 = load_audio(file1, audio_dir).to(device)
            wav2 = load_audio(file2, audio_dir).to(device)
            
            with torch.no_grad():
                emb1 = model.encode_batch(wav1)
                emb2 = model.encode_batch(wav2)
                
                # Process embeddings to ensure proper shape
                emb1_flat = process_embeddings(emb1)
                emb2_flat = process_embeddings(emb2)
                
                similarity = 1 - cosine(emb1_flat, emb2_flat)
                prediction = 1 if similarity > 0.5 else 0
                
                if prediction == label:
                    correct += 1
                total += 1
        except Exception as e:
            print(f"Error processing pair {file1} and {file2}: {e}")
            continue
    
    accuracy = correct / total if total > 0 else 0
    return accuracy

def attack_verification_pairs(model, pgd_attack, pairs_file, audio_dir="data/wav", 
                              output_dir="adversarial_examples", device="cuda", 
                              threshold=0.5, max_pairs=None, multi_speaker=True):
    """
    Attack verification pairs and save adversarial examples
    
    Args:
        model: SpeechBrain EncoderClassifier model
        pgd_attack: PGD attack instance
        pairs_file: File containing verification pairs
        audio_dir: Directory containing audio files
        output_dir: Directory to save adversarial examples
        device: Device to run the attack on
        threshold: Similarity threshold for verification
        max_pairs: Maximum number of pairs to process (for testing)
        multi_speaker: If True, ensure adversarial examples come from multiple speakers
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Load verification pairs
    with open(pairs_file, 'r') as f:
        pairs = f.readlines()
    
    # Track used speakers when multi_speaker is enabled
    used_speakers = set()
    speaker_count = {}  # Count how many times each speaker is used
    
    # Limit number of pairs for testing if specified
    if max_pairs is not None:
        pairs = pairs[:max_pairs]
        print(f"Processing only the first {max_pairs} pairs for testing")
    
    successful_attacks = 0
    total_attacks = 0
    failed_pairs = 0
    
    # Calculate SNR for each adversarial example
    snr_values = []
    
    for pair in tqdm(pairs, desc="Generating adversarial examples"):
        parts = pair.strip().split()
        label = int(parts[0])
        file1 = parts[1]
        file2 = parts[2]
        
        # Extract speaker ID from filename (assuming format like /spk_id/file.wav)
        try:
            speaker1 = file1.split("/")[0] if "/" in file1 else os.path.basename(file1).split("-")[0]
            speaker2 = file2.split("/")[0] if "/" in file2 else os.path.basename(file2).split("-")[0]
        except:
            speaker1 = "unknown"
            speaker2 = "unknown"
        
        # If using multi-speaker and we've already used this speaker extensively, consider skipping
        if multi_speaker and speaker_count:
            # Skip if both speakers have been used multiple times already (to promote diversity)
            if (speaker1 in speaker_count and speaker_count[speaker1] > 3 and 
                speaker2 in speaker_count and speaker_count[speaker2] > 3):
                # Only skip with some probability to ensure we still process enough pairs
                if np.random.random() > 0.3:
                    continue
        
        try:
            # Only attack pairs that are correctly classified by the model
            wav1 = load_audio(file1, audio_dir).to(device)
            wav2 = load_audio(file2, audio_dir).to(device)
            
            with torch.no_grad():
                emb1 = model.encode_batch(wav1)
                emb2 = model.encode_batch(wav2)
                
                # Process embeddings to ensure proper shape
                emb1_flat = process_embeddings(emb1)
                emb2_flat = process_embeddings(emb2)
                
                initial_similarity = 1 - cosine(emb1_flat, emb2_flat)
                initial_prediction = 1 if initial_similarity > threshold else 0
            
            # Skip if model already misclassifies
            if initial_prediction != label:
                print(f"Skipping pair {file1} and {file2} - already misclassified")
                continue
            
            # Determine attack strategy based on pair label
            if label == 1:  # Same speaker pair, try to make them different
                print(f"Attacking same speaker pair to make them different: {file1} and {file2}")
                # Attack first audio to make it different from second
                adv_wav1 = pgd_attack.generate_adversarial(wav1, targeted=False)
                
                # Create directory structure for saving files
                adv_file_basename = os.path.basename(file1)
                adv_file_dir = os.path.join(output_dir, os.path.dirname(file1))
                os.makedirs(adv_file_dir, exist_ok=True)
                adv_file1 = os.path.join(adv_file_dir, f"adv_{adv_file_basename}")
                
                # Save adversarial example
                torchaudio.save(adv_file1, adv_wav1.cpu(), 16000)
                
                # Check if attack was successful
                with torch.no_grad():
                    adv_emb1 = model.encode_batch(adv_wav1)
                    
                    # Process embeddings to ensure proper shape
                    adv_emb1_flat = process_embeddings(adv_emb1)
                    
                    new_similarity = 1 - cosine(adv_emb1_flat, emb2_flat)
                    new_prediction = 1 if new_similarity > threshold else 0
                
                # Calculate SNR for all cases (successful or not)
                original = wav1.cpu().numpy().flatten()
                adversarial = adv_wav1.cpu().numpy().flatten()
                noise = adversarial - original
                signal_power = np.mean(original ** 2)
                noise_power = np.mean(noise ** 2)
                snr = 10 * np.log10(signal_power / noise_power) if noise_power > 0 else float('inf')
                
                if new_prediction != label:
                    successful_attacks += 1
                    # Only add successful attacks to the SNR calculation for averaging
                    snr_values.append(snr)
                    
                total_attacks += 1
                
                # Update speaker count for diversity tracking
                if multi_speaker:
                    used_speakers.add(speaker1)
                    used_speakers.add(speaker2)
                    speaker_count[speaker1] = speaker_count.get(speaker1, 0) + 1
                    speaker_count[speaker2] = speaker_count.get(speaker2, 0) + 1
                
                print(f"Original similarity: {initial_similarity:.4f}, Adversarial similarity: {new_similarity:.4f}, SNR: {snr:.2f} dB")
                
            else:  # Different speaker pair, try to make them same
                print(f"Attacking different speaker pair to make them similar: {file1} and {file2}")
                # Attack first audio to make it similar to second
                adv_wav1 = pgd_attack.generate_adversarial(wav1, wav2, targeted=True)
                
                # Create directory structure for saving files
                adv_file_basename = os.path.basename(file1)
                adv_file_dir = os.path.join(output_dir, os.path.dirname(file1))
                os.makedirs(adv_file_dir, exist_ok=True)
                adv_file1 = os.path.join(adv_file_dir, f"adv_{adv_file_basename}")
                
                # Save adversarial example
                torchaudio.save(adv_file1, adv_wav1.cpu(), 16000)
                
                # Check if attack was successful
                with torch.no_grad():
                    adv_emb1 = model.encode_batch(adv_wav1)
                    
                    # Process embeddings to ensure proper shape
                    adv_emb1_flat = process_embeddings(adv_emb1)
                    
                    new_similarity = 1 - cosine(adv_emb1_flat, emb2_flat)
                    new_prediction = 1 if new_similarity > threshold else 0
                
                # Calculate SNR for all cases (successful or not)
                original = wav1.cpu().numpy().flatten()
                adversarial = adv_wav1.cpu().numpy().flatten()
                noise = adversarial - original
                signal_power = np.mean(original ** 2)
                noise_power = np.mean(noise ** 2)
                snr = 10 * np.log10(signal_power / noise_power) if noise_power > 0 else float('inf')
                
                if new_prediction != label:
                    successful_attacks += 1
                    # Only add successful attacks to the SNR calculation for averaging
                    snr_values.append(snr)
                    
                total_attacks += 1
                
                # Update speaker count for diversity tracking
                if multi_speaker:
                    used_speakers.add(speaker1)
                    used_speakers.add(speaker2)
                    speaker_count[speaker1] = speaker_count.get(speaker1, 0) + 1
                    speaker_count[speaker2] = speaker_count.get(speaker2, 0) + 1
                
                print(f"Original similarity: {initial_similarity:.4f}, Adversarial similarity: {new_similarity:.4f}, SNR: {snr:.2f} dB")
        except Exception as e:
            print(f"Error processing pair {file1} and {file2}: {e}")
            failed_pairs += 1
            continue
    
    # Calculate average SNR
    avg_snr = np.mean(snr_values) if snr_values else float('nan')
    
    attack_results = {
        'total_attacks': total_attacks,
        'successful_attacks': successful_attacks,
        'failed_pairs': failed_pairs,
        'success_rate': successful_attacks/total_attacks if total_attacks > 0 else 0,
        'unique_speakers': len(used_speakers),
        'avg_snr': avg_snr
    }
    
    if total_attacks > 0:
        print(f"Attack success rate: {attack_results['success_rate']:.4f} ({successful_attacks}/{total_attacks})")
        print(f"Failed to process {failed_pairs} pairs")
        print(f"Unique speakers used: {len(used_speakers)}")
        print(f"Average SNR: {avg_snr:.2f} dB")
    else:
        print("No attacks were performed. Check audio files and paths.")
    
    return attack_results

def main():
    parser = argparse.ArgumentParser(description='PGD Attack on SpeechBrain Speaker Verification Models')
    
    # Model parameters
    parser.add_argument('--model_type', type=str, choices=["xvector", "resnet", "ecapa"], default="resnet",
                        help='Type of SpeechBrain model to attack')
    parser.add_argument('--custom_model_path', type=str, default=None,
                        help='Custom model path (if not using the standard SpeechBrain models)')
    
    # Data parameters
    parser.add_argument('--pairs_file', type=str, default="data/veri_test.txt", 
                        help='File containing verification pairs')
    parser.add_argument('--audio_dir', type=str, default="data/wav", 
                        help='Directory containing audio files')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Directory to save adversarial examples')
    
    # Attack parameters
    parser.add_argument('--eps', type=float, default=0.002, 
                        help='Maximum perturbation size (epsilon)')
    parser.add_argument('--alpha', type=float, default=0.0004, 
                        help='Step size for each iteration')
    parser.add_argument('--steps', type=int, default=100, 
                        help='Number of iterations')
    parser.add_argument('--threshold', type=float, default=0.5, 
                        help='Similarity threshold for verification')
    
    # Other parameters
    parser.add_argument('--evaluate', action='store_true', 
                        help='Evaluate model accuracy before attack')
    parser.add_argument('--device', type=str, default="cuda" if torch.cuda.is_available() else "cpu", 
                        help='Device to run the attack on')
    parser.add_argument('--max_pairs', type=int, default=None,
                        help='Maximum number of pairs to process')
    parser.add_argument('--multi_speaker', action='store_true', 
                        help='Ensure adversarial examples come from multiple speakers')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug prints')
    
    args = parser.parse_args()
    
    # Determine model path
    model_path = args.custom_model_path if args.custom_model_path else MODEL_PATHS[args.model_type]
    
    # Set up output directory
    if args.output_dir is None:
        args.output_dir = f"adv_examples_{args.model_type}"
    
    # Print configuration
    print("\n" + "="*50)
    print("PGD Attack Configuration:")
    print("="*50)
    print(f"  Model type: {args.model_type}")
    print(f"  Model path: {model_path}")
    print(f"  Pairs file: {args.pairs_file}")
    print(f"  Audio directory: {args.audio_dir}")
    print(f"  Output directory: {args.output_dir}")
    print(f"  Epsilon: {args.eps}")
    print(f"  Alpha: {args.alpha}")
    print(f"  Steps: {args.steps}")
    print(f"  Threshold: {args.threshold}")
    print(f"  Device: {args.device}")
    print(f"  Max pairs: {args.max_pairs if args.max_pairs else 'All'}")
    print(f"  Multi-speaker mode: {args.multi_speaker}")
    print(f"  Debug mode: {args.debug}")
    print("="*50 + "\n")
    
    # Check for necessary files
    if not os.path.exists(args.audio_dir):
        print(f"Warning: Audio directory '{args.audio_dir}' does not exist. Creating it...")
        os.makedirs(args.audio_dir, exist_ok=True)
    
    if not os.path.exists(args.pairs_file):
        print(f"Error: Pairs file '{args.pairs_file}' does not exist.")
        return
    
    # Load model
    print(f"Loading model: {model_path}")
    try:
        savedir = f"pretrained_models/{args.model_type}" if args.custom_model_path is None else "pretrained_models"
        model = EncoderClassifier.from_hparams(source=model_path, savedir=savedir, 
                                              run_opts={"device": args.device})
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Evaluate model accuracy if requested
    if args.evaluate:
        print("Evaluating model accuracy before attack...")
        accuracy = evaluate_model(model, args.pairs_file, args.audio_dir, args.device)
        print(f"Model accuracy: {accuracy:.4f}")
    
    # Create PGD attack instance
    pgd_attack = PGDAttack(model, eps=args.eps, alpha=args.alpha, 
                          steps=args.steps, device=args.device, debug=args.debug)
    
    # Attack verification pairs
    print("Starting PGD attack...")
    attack_results = attack_verification_pairs(
        model, pgd_attack, args.pairs_file, args.audio_dir, 
        args.output_dir, args.device, args.threshold, args.max_pairs,
        args.multi_speaker
    )
    
    # Save attack configuration and results
    with open(os.path.join(args.output_dir, "attack_info.txt"), "w") as f:
        f.write("PGD Attack Configuration and Results\n")
        f.write("=================================\n\n")
        f.write(f"Model: {model_path}\n")
        f.write(f"Epsilon: {args.eps}\n")
        f.write(f"Alpha: {args.alpha}\n")
        f.write(f"Steps: {args.steps}\n")
        f.write(f"Threshold: {args.threshold}\n")
        f.write(f"Multi-speaker mode: {args.multi_speaker}\n\n")
        f.write("Results:\n")
        f.write(f"  Total attacks: {attack_results['total_attacks']}\n")
        f.write(f"  Successful attacks: {attack_results['successful_attacks']}\n")
        f.write(f"  Success rate: {attack_results['success_rate']:.4f}\n")
        f.write(f"  Failed pairs: {attack_results['failed_pairs']}\n")
        f.write(f"  Unique speakers used: {attack_results['unique_speakers']}\n")
        f.write(f"  Average SNR: {attack_results['avg_snr']:.2f} dB\n")

    print(f"Attack completed. Results saved to {args.output_dir}")

if __name__ == "__main__":
    main()

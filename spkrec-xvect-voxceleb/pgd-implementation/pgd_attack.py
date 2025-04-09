#!/usr/bin/env python3
"""
PGD Attack for SpeechBrain Speaker Verification Model
This script implements a Projected Gradient Descent attack against a SpeechBrain
x-vector model trained on VoxCeleb. The attack aims to fool the speaker verification
system by adding imperceptible perturbations to audio samples.
"""

import torch
import torchaudio
import numpy as np
import os
import argparse
from tqdm import tqdm
from speechbrain.inference import EncoderClassifier
from scipy.spatial.distance import cosine

class PGDAttack:
    def __init__(self, model, eps=0.002, alpha=0.0004, steps=100, device="cuda"):
        """
        Initialize PGD Attack
        
        Args:
            model: SpeechBrain EncoderClassifier model
            eps: Maximum perturbation size (epsilon)
            alpha: Step size for each iteration
            steps: Number of iterations
            device: Device to run the attack on
        """
        self.model = model
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.device = device
    
    def compute_embeddings(self, wav_tensor):
        """Extract embeddings from the model"""
        with torch.no_grad():
            embeddings = self.model.encode_batch(wav_tensor)
            return embeddings.squeeze(1)
    
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
        orig_embedding = self.compute_embeddings(source_wav.to(self.device))
        
        # For targeted attacks, compute target embeddings
        if targeted and target_wav is not None:
            target_embedding = self.compute_embeddings(target_wav.to(self.device))
        
        # Enable gradients for the input
        adv_wav.requires_grad = True
        
        # PGD attack loop
        for _ in range(self.steps):
            # Zero gradients
            if adv_wav.grad is not None:
                adv_wav.grad.zero_()
            
            # Forward pass
            adv_embedding = self.model.encode_batch(adv_wav).squeeze(1)
            
            # Compute loss based on attack type
            if targeted and target_wav is not None:
                # For targeted attack, minimize distance to target
                loss = -torch.nn.functional.cosine_similarity(adv_embedding, target_embedding, dim=1).mean()
            else:
                # For untargeted attack, maximize distance from original
                loss = torch.nn.functional.cosine_similarity(adv_embedding, orig_embedding, dim=1).mean()
            
            # Backward pass
            loss.backward()
            
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

def evaluate_model(model, pairs_file, audio_dir="data/wav", device="cuda"):
    """Evaluate model accuracy on verification pairs"""
    correct = 0
    total = 0
    
    with open(pairs_file, 'r') as f:
        pairs = f.readlines()
    
    for pair in tqdm(pairs):
        parts = pair.strip().split()
        label = int(parts[0])
        file1 = parts[1]
        file2 = parts[2]
        
        try:
            wav1 = load_audio(file1, audio_dir).to(device)
            wav2 = load_audio(file2, audio_dir).to(device)
            
            with torch.no_grad():
                emb1 = model.encode_batch(wav1).squeeze(1)
                emb2 = model.encode_batch(wav2).squeeze(1)
                
                # Flatten embeddings to ensure 1-D for cosine distance
                emb1_flat = emb1.cpu().numpy().flatten()
                emb2_flat = emb2.cpu().numpy().flatten()
                
                similarity = 1 - cosine(emb1_flat, emb2_flat)
                prediction = 1 if similarity > 0.5 else 0
                
                if prediction == label:
                    correct += 1
                total += 1
        except Exception as e:
            print(f"Error processing pair {file1} and {file2}: {e}")
            continue
    
    return correct / total if total > 0 else 0

def attack_verification_pairs(model, pgd_attack, pairs_file, audio_dir="data/wav", output_dir="adversarial_examples", device="cuda", threshold=0.5, max_pairs=None):
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
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Load verification pairs
    with open(pairs_file, 'r') as f:
        pairs = f.readlines()
    
    # Limit number of pairs for testing if specified
    if max_pairs is not None:
        pairs = pairs[:max_pairs]
        print(f"Processing only the first {max_pairs} pairs for testing")
    
    successful_attacks = 0
    total_attacks = 0
    failed_pairs = 0
    
    for pair in tqdm(pairs):
        parts = pair.strip().split()
        label = int(parts[0])
        file1 = parts[1]
        file2 = parts[2]
        
        try:
            # Only attack pairs that are correctly classified by the model
            wav1 = load_audio(file1, audio_dir).to(device)
            wav2 = load_audio(file2, audio_dir).to(device)
            
            with torch.no_grad():
                emb1 = model.encode_batch(wav1).squeeze(1)
                emb2 = model.encode_batch(wav2).squeeze(1)
                
                # Flatten embeddings to ensure 1-D for cosine distance
                emb1_flat = emb1.cpu().numpy().flatten()
                emb2_flat = emb2.cpu().numpy().flatten()
                
                initial_similarity = 1 - cosine(emb1_flat, emb2_flat)
                initial_prediction = 1 if initial_similarity > threshold else 0
            
            # Skip if model already misclassifies
            if initial_prediction != label:
                continue
            
            # Determine attack strategy based on pair label
            if label == 1:  # Same speaker pair, try to make them different
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
                    adv_emb1 = model.encode_batch(adv_wav1).squeeze(1)
                    # Flatten embeddings
                    adv_emb1_flat = adv_emb1.cpu().numpy().flatten()
                    emb2_flat = emb2.cpu().numpy().flatten()
                    
                    new_similarity = 1 - cosine(adv_emb1_flat, emb2_flat)
                    new_prediction = 1 if new_similarity > threshold else 0
                
                if new_prediction != label:
                    successful_attacks += 1
                total_attacks += 1
                
                print(f"Original similarity: {initial_similarity:.4f}, Adversarial similarity: {new_similarity:.4f}")
                
            else:  # Different speaker pair, try to make them same
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
                    adv_emb1 = model.encode_batch(adv_wav1).squeeze(1)
                    # Flatten embeddings
                    adv_emb1_flat = adv_emb1.cpu().numpy().flatten()
                    emb2_flat = emb2.cpu().numpy().flatten()
                    
                    new_similarity = 1 - cosine(adv_emb1_flat, emb2_flat)
                    new_prediction = 1 if new_similarity > threshold else 0
                
                if new_prediction != label:
                    successful_attacks += 1
                total_attacks += 1
                
                print(f"Original similarity: {initial_similarity:.4f}, Adversarial similarity: {new_similarity:.4f}")
        except Exception as e:
            print(f"Error processing pair {file1} and {file2}: {e}")
            failed_pairs += 1
            continue
    
    if total_attacks > 0:
        print(f"Attack success rate: {successful_attacks/total_attacks:.4f} ({successful_attacks}/{total_attacks})")
        print(f"Failed to process {failed_pairs} pairs")
    else:
        print("No attacks were performed. Check audio files and paths.")

def main():
    parser = argparse.ArgumentParser(description='PGD Attack on SpeechBrain Speaker Verification Model')
    parser.add_argument('--model', type=str, default="speechbrain/spkrec-xvect-voxceleb", 
                        help='SpeechBrain model path or name')
    parser.add_argument('--pairs_file', type=str, default="data/veri_test.txt", 
                        help='File containing verification pairs')
    parser.add_argument('--audio_dir', type=str, default="data/wav", 
                        help='Directory containing audio files')
    parser.add_argument('--output_dir', type=str, default="adversarial_examples", 
                        help='Directory to save adversarial examples')
    parser.add_argument('--eps', type=float, default=0.002, 
                        help='Maximum perturbation size (epsilon)')
    parser.add_argument('--alpha', type=float, default=0.0004, 
                        help='Step size for each iteration')
    parser.add_argument('--steps', type=int, default=100, 
                        help='Number of iterations')
    parser.add_argument('--threshold', type=float, default=0.5, 
                        help='Similarity threshold for verification')
    parser.add_argument('--evaluate', action='store_true', 
                        help='Evaluate model accuracy before attack')
    parser.add_argument('--device', type=str, default="cuda" if torch.cuda.is_available() else "cpu", 
                        help='Device to run the attack on')
    parser.add_argument('--max_pairs', type=int, default=None,
                        help='Maximum number of pairs to process (for testing)')
    
    args = parser.parse_args()
    
    # Print configuration
    print("PGD Attack Configuration:")
    print(f"  Model: {args.model}")
    print(f"  Pairs file: {args.pairs_file}")
    print(f"  Audio directory: {args.audio_dir}")
    print(f"  Output directory: {args.output_dir}")
    print(f"  Epsilon: {args.eps}")
    print(f"  Alpha: {args.alpha}")
    print(f"  Steps: {args.steps}")
    print(f"  Threshold: {args.threshold}")
    print(f"  Device: {args.device}")
    print(f"  Max pairs: {args.max_pairs if args.max_pairs else 'All'}")
    
    # Check if audio directory exists
    if not os.path.exists(args.audio_dir):
        print(f"Warning: Audio directory '{args.audio_dir}' does not exist. Creating it...")
        os.makedirs(args.audio_dir, exist_ok=True)
    
    # Check if pairs file exists
    if not os.path.exists(args.pairs_file):
        print(f"Error: Pairs file '{args.pairs_file}' does not exist.")
        return

    # Load the SpeechBrain model
    print(f"Loading model: {args.model}")
    try:
        model = EncoderClassifier.from_hparams(source=args.model, savedir="pretrained_models", 
                                              run_opts={"device": args.device})
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Evaluate model accuracy if requested
    if args.evaluate:
        print("Evaluating model accuracy...")
        accuracy = evaluate_model(model, args.pairs_file, args.audio_dir, args.device)
        print(f"Model accuracy: {accuracy:.4f}")
    
    # Create PGD attack instance
    pgd_attack = PGDAttack(model, eps=args.eps, alpha=args.alpha, steps=args.steps, device=args.device)
    
    # Attack verification pairs
    print("Generating adversarial examples...")
    attack_verification_pairs(model, pgd_attack, args.pairs_file, args.audio_dir, 
                             args.output_dir, args.device, args.threshold, args.max_pairs)

if __name__ == "__main__":
    main()

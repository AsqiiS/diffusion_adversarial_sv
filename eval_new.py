import os
import random
import torch
import torchaudio
import numpy as np
import pandas as pd
from speechbrain.inference.speaker import EncoderClassifier
from scipy.spatial.distance import cosine
from sklearn.metrics import roc_curve, auc
from tqdm import tqdm
import matplotlib.pyplot as plt
import argparse
import sys

def calculate_eer(labels, scores):
    """Calculate the Equal Error Rate (EER) and the corresponding threshold."""
    if len(np.unique(labels)) < 2:
        return float('nan'), float('nan')
    
    fpr, tpr, thresholds = roc_curve(labels, scores)
    fnr = 1 - tpr
    eer_threshold_idx = np.nanargmin(np.abs(fnr - fpr))
    eer_threshold = thresholds[eer_threshold_idx]
    eer = np.mean([fpr[eer_threshold_idx], fnr[eer_threshold_idx]])
    return eer, eer_threshold

def plot_roc_curve(labels, scores, save_path=None):
    """Plot the ROC curve for evaluation."""
    fpr, tpr, thresholds = roc_curve(labels, scores)
    auc_score = auc(fpr, tpr)
    
    plt.figure(figsize=(10, 7))
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc_score:.4f})')
    
    # Plot EER point
    fnr = 1 - tpr
    eer_threshold_idx = np.nanargmin(np.abs(fnr - fpr))
    eer = np.mean([fpr[eer_threshold_idx], fnr[eer_threshold_idx]])
    plt.plot(fpr[eer_threshold_idx], tpr[eer_threshold_idx], 'ro', markersize=8, 
             label=f'EER = {eer:.4f}')
    
    plt.plot([0, 1], [1, 0], 'k--', label='Random Guess')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve for Speaker Verification System')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"ROC curve saved to {save_path}")
    
    plt.show()

def get_random_other_speaker_audio(current_speaker, all_speakers_dir):
    """Randomly sample a different speaker's audio file."""
    speakers = [spk for spk in os.listdir(all_speakers_dir) 
                if os.path.isdir(os.path.join(all_speakers_dir, spk)) and spk != current_speaker]
    
    if not speakers:
        return None  # No other speakers found
    
    # Try up to 5 times to find a valid audio file
    for _ in range(5):
        random_speaker = random.choice(speakers)
        speaker_path = os.path.join(all_speakers_dir, random_speaker)
        
        if not os.path.isdir(speaker_path):
            continue
            
        vid_dirs = [v for v in os.listdir(speaker_path) 
                   if os.path.isdir(os.path.join(speaker_path, v))]
        if not vid_dirs:
            continue
            
        random_vid = random.choice(vid_dirs)
        vid_path = os.path.join(speaker_path, random_vid)
        
        audio_files = [f for f in os.listdir(vid_path) 
                      if f.endswith(('.wav', '.WAV', '.mp3', '.MP3', '.flac', '.FLAC'))]
        if not audio_files:
            continue
            
        random_file = random.choice(audio_files)
        return os.path.join(speaker_path, random_vid, random_file)
    
    return None  # Couldn't find a suitable file after multiple attempts

def process_audio_file(audio_path, device, target_sr=16000):
    """Load and process an audio file for speaker verification."""
    try:
        waveform, sr = torchaudio.load(audio_path)
        
        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
            
        # Resample if needed
        if sr != target_sr:
            waveform = torchaudio.functional.resample(waveform, sr, target_sr)
            
        # Move to device
        waveform = waveform.to(device)
        return waveform
    except Exception as e:
        print(f"Error processing audio file {audio_path}: {e}")
        return None

def find_all_audio_files(root_dir, speaker_id=None):
    """Find all audio files in the directory structure."""
    audio_files = []
    
    # If speaker_id is provided, only look in that speaker's directory
    if speaker_id:
        speaker_path = os.path.join(root_dir, speaker_id)
        if not os.path.exists(speaker_path):
            print(f"Speaker directory not found: {speaker_path}")
            return []
            
        dirs_to_search = [speaker_path]
    else:
        # Otherwise, search all speaker directories
        if not os.path.exists(root_dir):
            print(f"Root directory not found: {root_dir}")
            return []
            
        dirs_to_search = [os.path.join(root_dir, d) for d in os.listdir(root_dir) 
                          if os.path.isdir(os.path.join(root_dir, d))]
    
    # Search through the directory structure
    for speaker_dir in dirs_to_search:
        speaker_id = os.path.basename(speaker_dir)
        
        for vid in os.listdir(speaker_dir):
            vid_path = os.path.join(speaker_dir, vid)
            if not os.path.isdir(vid_path):
                continue
                
            for audio_file in os.listdir(vid_path):
                if audio_file.endswith(('.wav', '.WAV', '.mp3', '.MP3', '.flac', '.FLAC')):
                    full_path = os.path.join(vid_path, audio_file)
                    audio_files.append((speaker_id, full_path))
    
    return audio_files

def main():
    parser = argparse.ArgumentParser(description="Evaluate adversarial audio attacks")
    parser.add_argument("--data_root", type=str, default="../data/wav", 
                       help="Root directory of clean audio dataset")
    parser.add_argument("--adv_dir", type=str, default="adversarial_examples", 
                       help="Directory containing adversarial examples")
    parser.add_argument("--transcription_file", type=str, default="transcriptions.tsv", 
                       help="TSV file with transcriptions")
    parser.add_argument("--output_dir", type=str, default="evaluation_results", 
                       help="Directory to save evaluation results")
    parser.add_argument("--attack_type", type=str, choices=["impersonation", "evasion"], default="impersonation",
                       help="Type of attack being evaluated")
    parser.add_argument("--target_speaker", type=str, default=None,
                       help="Target speaker ID (for impersonation attacks)")
    parser.add_argument("--model_name", type=str, default="ecapa",
                       help="Speaker verification model to use: ecapa, xvect")
    parser.add_argument("--threshold", type=float, default=0.5,
                       help="Similarity threshold for initial attack success calculation")
    
    args = parser.parse_args()
    
    # Verify data root directory exists
    if not os.path.exists(args.data_root):
        print(f"ERROR: Data root directory not found: {args.data_root}")
        print(f"Current working directory: {os.getcwd()}")
        print("Available directories:")
        print("\n".join(["- " + d for d in os.listdir() if os.path.isdir(d)]))
        sys.exit(1)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load speaker verification model
    print(f"Loading {args.model_name} speaker verification model...")
    if args.model_name == "ecapa":
        model = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir="pretrained_models/ecapa",
            run_opts={"device": device}
        )
    elif args.model_name == "xvect":
        model = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-xvect-voxceleb",
            savedir="pretrained_models/xvect",
            run_opts={"device": device}
        )
    elif args.model_name == 'resnet': 
        model = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-resnet-voxceleb",
            savedir="pretrained_models/resnet",
            run_opts={"device": "cuda"}
        )

    else:
        raise ValueError(f"Unknown model name: {args.model_name}")
    
    model.eval()
    
    # Load transcription file
    print(f"Loading transcription file: {args.transcription_file}")
    try:
        df = pd.read_csv(args.transcription_file, sep='\t', header=0)
        print(f"Read {len(df)} records from transcription file")
        
        if len(df) > 0:
            print(f"Column names: {df.columns.tolist()}")
            print(f"Sample row from TSV: {df.iloc[0].tolist()}")
        
        # Create transcription dictionary using column names
        transcription_dict = {}
        audio_path_col = 'audio_path' if 'audio_path' in df.columns else 3  # Fallback to index if column name not found
        transcription_col = 'transcription' if 'transcription' in df.columns else 4
        
        for _, row in df.iterrows():
            # Get the audio path and transcription
            audio_path = row[audio_path_col] if audio_path_col in df.columns or audio_path_col < len(row) else None
            transcription = row[transcription_col] if transcription_col in df.columns or transcription_col < len(row) else None
            
            if audio_path and transcription:
                # Remove 'data/wav/' prefix if present
                key = audio_path.replace('data/wav/', '').strip()
                transcription_dict[key] = transcription
        
        print(f"Created transcription dictionary with {len(transcription_dict)} entries")
        
    except Exception as e:
        print(f"Error reading transcription file: {e}")
        transcription_dict = {}
    
    # Search for adversarial examples
    print(f"Searching for adversarial examples in: {args.adv_dir}")
    if not os.path.exists(args.adv_dir):
        print(f"Adversarial examples directory not found: {args.adv_dir}")
        sys.exit(1)
        
    adv_files = [f for f in os.listdir(args.adv_dir) if f.endswith('_adv.wav')]
    if not adv_files:
        print(f"No adversarial examples found in {args.adv_dir}")
        sys.exit(1)
        
    print(f"Found {len(adv_files)} adversarial examples")
    
    # Find all original audio files
    print(f"Finding original audio files in: {args.data_root}")
    all_audio_files = find_all_audio_files(args.data_root)
    if not all_audio_files:
        print(f"No audio files found in {args.data_root}")
        sys.exit(1)
        
    print(f"Found {len(all_audio_files)} original audio files")
    
    # Group audio files by speaker
    speakers_dict = {}
    for speaker_id, file_path in all_audio_files:
        if speaker_id not in speakers_dict:
            speakers_dict[speaker_id] = []
        speakers_dict[speaker_id].append(file_path)
    
    print(f"Found {len(speakers_dict)} speakers")
    
    # Initialize metrics
    all_labels = []  # ground-truth labels (1=same speaker, 0=different speaker)
    all_scores = []  # similarity scores
    
    # For attack success rate calculation
    total_samples = 0
    attack_successful = 0
    
    # Create detailed results log
    results_log = []
    
    # Process each adversarial example
    print("Evaluating adversarial examples...")
    for adv_file in tqdm(adv_files, desc="Processing"):
        # Extract speaker ID from adversarial filename (assuming format: speaker_id_adv.wav)
        speaker_id = adv_file.replace('_adv.wav', '')
        
        # Skip if no original files for this speaker
        if speaker_id not in speakers_dict:
            print(f"No original files found for speaker: {speaker_id}")
            continue
            
        # Full path to adversarial file
        adv_audio_path = os.path.join(args.adv_dir, adv_file)
        
        # Load adversarial audio
        adv_audio = process_audio_file(adv_audio_path, device)
        if adv_audio is None:
            continue
            
        # Extract embedding for adversarial audio
        with torch.no_grad():
            adv_emb = model.encode_batch(adv_audio).squeeze().cpu().numpy()
        
        # Get original audio files for this speaker (up to 5)
        original_files = speakers_dict[speaker_id][:5]
        
        # Process each original file
        for original_file in original_files:
            # Load original audio
            original_audio = process_audio_file(original_file, device)
            if original_audio is None:
                continue
                
            # Extract embedding for original audio
            with torch.no_grad():
                original_emb = model.encode_batch(original_audio).squeeze().cpu().numpy()
            
            # Calculate similarity between adversarial and original (same-speaker comparison)
            similarity_same = 1 - cosine(adv_emb, original_emb)
            all_labels.append(1)  # Same speaker label
            all_scores.append(similarity_same)
            
            # Log detailed result
            result_entry = {
                "speaker_id": speaker_id,
                "adv_path": adv_audio_path,
                "original_path": original_file,
                "comparison_type": "same_speaker",
                "similarity_score": similarity_same
            }
            results_log.append(result_entry)
            
            # Get a random different speaker for impostor comparison
            other_speaker_path = get_random_other_speaker_audio(speaker_id, args.data_root)
            if other_speaker_path:
                # Load other speaker audio
                other_audio = process_audio_file(other_speaker_path, device)
                if other_audio is None:
                    continue
                    
                # Extract embedding for other speaker
                with torch.no_grad():
                    other_emb = model.encode_batch(other_audio).squeeze().cpu().numpy()
                
                # Calculate similarity between adversarial and different speaker
                similarity_diff = 1 - cosine(adv_emb, other_emb)
                all_labels.append(0)  # Different speaker label
                all_scores.append(similarity_diff)
                
                # Get the other speaker's ID from the path
                other_speaker_id = os.path.basename(os.path.dirname(os.path.dirname(other_speaker_path)))
                
                # Log detailed result
                result_entry = {
                    "speaker_id": speaker_id,
                    "adv_path": adv_audio_path,
                    "comparison_path": other_speaker_path,
                    "comparison_type": "different_speaker",
                    "similarity_score": similarity_diff,
                    "other_speaker_id": other_speaker_id
                }
                results_log.append(result_entry)
            
            total_samples += 1
            
            # Evaluate attack success based on attack type
            if args.attack_type == "impersonation":
                # For impersonation, we want the adversarial to be accepted as the target
                if args.target_speaker and other_speaker_id == args.target_speaker:
                    attack_successful += (similarity_diff > args.threshold)
                else:
                    # If no specific target, success is when adversarial is accepted as any other speaker
                    attack_successful += (similarity_diff > args.threshold)
            elif args.attack_type == "evasion":
                # For evasion, we want the adversarial to be rejected by the original speaker
                attack_successful += (similarity_same <= args.threshold)
    
    # Calculate initial attack success rate
    attack_success_rate = attack_successful / total_samples if total_samples > 0 else 0
    
    # Calculate EER and threshold
    if len(all_labels) > 0 and len(np.unique(all_labels)) > 1:
        eer, eer_threshold = calculate_eer(all_labels, all_scores)
        
        # Calculate FAR and FRR at the EER threshold
        same_scores = [score for score, label in zip(all_scores, all_labels) if label == 1]
        diff_scores = [score for score, label in zip(all_scores, all_labels) if label == 0]
        
        far = sum([score > eer_threshold for score in diff_scores]) / len(diff_scores) if diff_scores else float('nan')
        frr = sum([score <= eer_threshold for score in same_scores]) / len(same_scores) if same_scores else float('nan')
        
        # Recalculate attack success rate using the EER threshold
        attack_successful_eer = 0
        for i, (label, score) in enumerate(zip(all_labels, all_scores)):
            if args.attack_type == "impersonation" and label == 0:
                # For impersonation against another speaker, success is when adversarial is accepted as that speaker
                attack_successful_eer += (score > eer_threshold)
            elif args.attack_type == "evasion" and label == 1:
                # For evasion against original speaker, success is when adversarial is rejected as original speaker
                attack_successful_eer += (score <= eer_threshold)
        
        attack_success_rate_eer = attack_successful_eer / total_samples if total_samples > 0 else 0
        
        print("\n===== Evaluation Results =====")
        print(f"Total samples evaluated: {total_samples}")
        print(f"Total comparisons: {len(all_labels)}")
        print(f"Attack type: {args.attack_type}")
        print(f"EER: {eer:.4f}")
        print(f"EER threshold: {eer_threshold:.4f}")
        print(f"FAR @ EER threshold: {far:.4f}")
        print(f"FRR @ EER threshold: {frr:.4f}")
        print(f"Attack success rate @ default threshold ({args.threshold}): {attack_success_rate:.4f}")
        print(f"Attack success rate @ EER threshold: {attack_success_rate_eer:.4f}")
        
        # Save ROC curve
        roc_path = os.path.join(args.output_dir, f"roc_curve_{args.model_name}_{args.attack_type}.png")
        plot_roc_curve(all_labels, all_scores, save_path=roc_path)
        
        # Save detailed metrics to CSV
        metrics_df = pd.DataFrame(results_log)
        metrics_path = os.path.join(args.output_dir, f"detailed_results_{args.model_name}_{args.attack_type}.csv")
        metrics_df.to_csv(metrics_path, index=False)
        print(f"Detailed results saved to {metrics_path}")
        
        # Save overall metrics summary
        summary = {
            "model": args.model_name,
            "attack_type": args.attack_type,
            "total_samples": total_samples,
            "eer": eer,
            "eer_threshold": eer_threshold,
            "far": far,
            "frr": frr,
            "attack_success_rate_default_threshold": attack_success_rate,
            "attack_success_rate_eer_threshold": attack_success_rate_eer
        }
        summary_df = pd.DataFrame([summary])
        summary_path = os.path.join(args.output_dir, f"summary_{args.model_name}_{args.attack_type}.csv")
        summary_df.to_csv(summary_path, index=False)
        print(f"Summary metrics saved to {summary_path}")
    else:
        print("\nNot enough data to compute EER. Need samples from at least two classes.")

if __name__ == "__main__":
    main()
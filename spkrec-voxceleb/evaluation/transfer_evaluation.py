#!/usr/bin/env python3
"""
Transfer Attack Evaluation for SpeechBrain Speaker Verification Models
This script evaluates how adversarial examples transfer between different model architectures
"""

import torch
import torchaudio
import numpy as np
import os
import argparse
from tqdm import tqdm
from speechbrain.inference import EncoderClassifier
from scipy.spatial.distance import cosine
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import roc_curve

# Model hub paths for supported models
MODEL_PATHS = {
    "xvector": "speechbrain/spkrec-xvect-voxceleb",
    "resnet": "speechbrain/spkrec-resnet-voxceleb",
    "ecapa": "speechbrain/spkrec-ecapa-voxceleb"
}

def load_audio(file_path, audio_dir=None, sample_rate=16000):
    """Load audio file and resample if necessary"""
    # Handle path construction - allow for both relative and absolute paths
    if audio_dir is not None and not os.path.isabs(file_path):
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

def compute_snr(original, adversarial):
    """Compute Signal-to-Noise Ratio between original and adversarial waveforms"""
    original = original.numpy().flatten()
    adversarial = adversarial.numpy().flatten()
    noise = adversarial - original
    
    signal_power = np.sum(original ** 2)
    noise_power = np.sum(noise ** 2)
    
    if noise_power == 0:
        return float('inf')
    
    snr = 10 * np.log10(signal_power / noise_power)
    return snr

def calculate_eer(labels, scores):
    """
    Calculate Equal Error Rate (EER)
    
    Args:
        labels: Ground truth labels (0 for different speakers, 1 for same speaker)
        scores: Similarity scores
    
    Returns:
        EER value and threshold at which EER occurs
    """
    if len(np.unique(labels)) < 2:
        return float('nan'), float('nan')
    
    fpr, tpr, thresholds = roc_curve(labels, scores)
    fnr = 1 - tpr
    
    # Find the threshold where FAR = FRR
    eer_threshold_idx = np.nanargmin(np.absolute(fnr - fpr))
    eer_threshold = thresholds[eer_threshold_idx]
    eer = np.mean([fpr[eer_threshold_idx], fnr[eer_threshold_idx]])
    
    return eer, eer_threshold

def evaluate_transfer_attacks(source_model_name, models, pairs_file, audio_dir="data/wav", 
                             adv_dir="adversarial_examples", device="cuda", threshold=0.5, max_pairs=None):
    """
    Evaluate the transfer effectiveness of adversarial examples across different models
    
    Args:
        source_model_name: Name of the source model used to generate adversarial examples
        models: Dictionary mapping model names to SpeechBrain EncoderClassifier models
        pairs_file: File containing verification pairs
        audio_dir: Directory containing original audio files
        adv_dir: Directory containing adversarial examples
        device: Device to run the evaluation on
        threshold: Similarity threshold for verification
        max_pairs: Maximum number of pairs to evaluate
    
    Returns:
        Dictionary containing evaluation metrics for all models
    """
    # Initialize results for each model
    results = {}
    for model_name in models.keys():
        results[model_name] = {
            "original_correct": 0,
            "adv_correct": 0,
            "adv_successful": 0,
            "total_pairs": 0,
            "snr_values": [],
            "original_labels": [],
            "original_scores": [],
            "adversarial_labels": [],
            "adversarial_scores": [],
            "detailed_results": {
                'pair_id': [],
                'label': [],
                'file1': [],
                'file2': [],
                'original_similarity': [],
                'adversarial_similarity': [],
                'original_prediction': [],
                'adversarial_prediction': [],
                'attack_success': [],
                'snr': []
            }
        }
    
    # Load verification pairs
    with open(pairs_file, 'r') as f:
        pairs = f.readlines()
    
    # Limit pairs if max_pairs is specified
    if max_pairs is not None:
        pairs = pairs[:max_pairs]
        print(f"Evaluating only the first {max_pairs} pairs for testing")
    
    for i, pair in enumerate(tqdm(pairs, desc="Evaluating transfer attack")):
        parts = pair.strip().split()
        label = int(parts[0])
        file1 = parts[1]
        file2 = parts[2]
        
        # Construct paths
        adv_file_dir = os.path.dirname(file1)
        adv_file_basename = os.path.basename(file1)
        adv_path1 = os.path.join(adv_dir, adv_file_dir, f"adv_{adv_file_basename}")
        
        # Skip if adversarial example doesn't exist
        if not os.path.exists(adv_path1):
            print(f"Skipping pair - adversarial example not found: {adv_path1}")
            continue
        
        try:
            # Load waveforms (do this once for all models)
            orig_wav1 = load_audio(file1, audio_dir).to(device)
            orig_wav2 = load_audio(file2, audio_dir).to(device)
            adv_wav1 = load_audio(adv_path1, None).to(device)  # No audio_dir for adv files
            
            # Calculate SNR (do this once)
            snr = compute_snr(orig_wav1.cpu(), adv_wav1.cpu())
            
            # Evaluate on each model
            for model_name, model in models.items():
                with torch.no_grad():
                    # Compute embeddings for this model
                    orig_emb1 = model.encode_batch(orig_wav1)
                    orig_emb2 = model.encode_batch(orig_wav2)
                    adv_emb1 = model.encode_batch(adv_wav1)
                    
                    # Process embeddings to ensure proper shape
                    orig_emb1_flat = process_embeddings(orig_emb1)
                    orig_emb2_flat = process_embeddings(orig_emb2)
                    adv_emb1_flat = process_embeddings(adv_emb1)
                    
                    # Compute similarities
                    orig_similarity = 1 - cosine(orig_emb1_flat, orig_emb2_flat)
                    adv_similarity = 1 - cosine(adv_emb1_flat, orig_emb2_flat)
                    
                    # Make predictions
                    orig_prediction = 1 if orig_similarity > threshold else 0
                    adv_prediction = 1 if adv_similarity > threshold else 0
                
                # Update metrics for this model
                if orig_prediction == label:
                    results[model_name]["original_correct"] += 1
                
                if adv_prediction == label:
                    results[model_name]["adv_correct"] += 1
                else:
                    if orig_prediction == label:  # Attack was successful
                        results[model_name]["adv_successful"] += 1
                
                # Add SNR
                results[model_name]["snr_values"].append(snr)
                
                # Save detailed results
                model_results = results[model_name]["detailed_results"]
                model_results['pair_id'].append(i)
                model_results['label'].append(label)
                model_results['file1'].append(file1)
                model_results['file2'].append(file2)
                model_results['original_similarity'].append(orig_similarity)
                model_results['adversarial_similarity'].append(adv_similarity)
                model_results['original_prediction'].append(orig_prediction)
                model_results['adversarial_prediction'].append(adv_prediction)
                model_results['attack_success'].append(1 if orig_prediction == label and adv_prediction != label else 0)
                model_results['snr'].append(snr)
                
                # Store data for EER calculation
                results[model_name]["original_labels"].append(label)
                results[model_name]["original_scores"].append(orig_similarity)
                results[model_name]["adversarial_labels"].append(label)
                results[model_name]["adversarial_scores"].append(adv_similarity)
                
                # Update total pairs for this model
                results[model_name]["total_pairs"] += 1
                
        except Exception as e:
            print(f"Error processing pair {file1} and {file2}: {e}")
            continue
    
    # Calculate final metrics for each model
    final_metrics = {}
    for model_name, model_results in results.items():
        # Calculate EER for original and adversarial examples
        if len(model_results["original_labels"]) > 0 and len(np.unique(model_results["original_labels"])) > 1:
            original_eer, original_eer_threshold = calculate_eer(
                model_results["original_labels"], model_results["original_scores"])
        else:
            original_eer, original_eer_threshold = float('nan'), float('nan')
        
        if len(model_results["adversarial_labels"]) > 0 and len(np.unique(model_results["adversarial_labels"])) > 1:
            adversarial_eer, adversarial_eer_threshold = calculate_eer(
                model_results["adversarial_labels"], model_results["adversarial_scores"])
        else:
            adversarial_eer, adversarial_eer_threshold = float('nan'), float('nan')
        
        # Calculate metrics
        total_pairs = model_results["total_pairs"]
        original_correct = model_results["original_correct"]
        
        final_metrics[model_name] = {
            'original_accuracy': original_correct / total_pairs if total_pairs > 0 else 0,
            'adversarial_accuracy': model_results["adv_correct"] / total_pairs if total_pairs > 0 else 0,
            'attack_success_rate': model_results["adv_successful"] / original_correct if original_correct > 0 else 0,
            'average_snr': np.mean(model_results["snr_values"]) if model_results["snr_values"] else 0,
            'original_eer': original_eer,
            'original_eer_threshold': original_eer_threshold,
            'adversarial_eer': adversarial_eer,
            'adversarial_eer_threshold': adversarial_eer_threshold,
            'results_df': pd.DataFrame(model_results["detailed_results"])
        }
    
    return final_metrics

def visualize_transfer_results(metrics, source_model_name, output_dir):
    """Visualize transfer attack results and save plots"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a summary DataFrame
    summary_data = []
    for model_name, model_metrics in metrics.items():
        model_short_name = os.path.basename(model_name)
        summary_data.append({
            'model': model_short_name,
            'original_eer': model_metrics['original_eer'],
            'adversarial_eer': model_metrics['adversarial_eer'],
            'attack_success_rate': model_metrics['attack_success_rate'],
            'original_accuracy': model_metrics['original_accuracy'],
            'adversarial_accuracy': model_metrics['adversarial_accuracy'],
            'eer_difference': model_metrics['adversarial_eer'] - model_metrics['original_eer'],
            'is_source': model_name == source_model_name
        })
    
    summary_df = pd.DataFrame(summary_data)
    
    # Calculate transfer rates
    source_asr = summary_df.loc[summary_df['is_source'], 'attack_success_rate'].values[0]
    if source_asr > 0:
        summary_df['transfer_rate'] = summary_df['attack_success_rate'] / source_asr
    else:
        summary_df['transfer_rate'] = 0
    
    # 1. EER Comparison
    plt.figure(figsize=(12, 6))
    bar_width = 0.35
    x = np.arange(len(summary_df))
    
    plt.bar(x - bar_width/2, summary_df['original_eer'], bar_width, label='Original EER', color='blue', alpha=0.7)
    plt.bar(x + bar_width/2, summary_df['adversarial_eer'], bar_width, label='Adversarial EER', color='red', alpha=0.7)
    
    plt.xlabel('Model')
    plt.ylabel('Equal Error Rate (EER)')
    plt.title('Original vs. Adversarial EER Across Models')
    plt.xticks(x, summary_df['model'])
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Highlight source model
    for i, is_source in enumerate(summary_df['is_source']):
        if is_source:
            plt.text(i, 0.01, 'Source', ha='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'eer_comparison.png'), dpi=300)
    
    # 2. Attack success rate across models
    plt.figure(figsize=(10, 6))
    colors = ['red' if x else 'blue' for x in summary_df['is_source']]
    bars = plt.bar(range(len(summary_df)), summary_df['attack_success_rate'], color=colors)
    
    # Add labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{height:.2f}', ha='center', va='bottom')
    
    plt.xlabel('Model')
    plt.ylabel('Attack Success Rate')
    plt.title(f'Adversarial Attack Success Rates (Source: {os.path.basename(source_model_name)})')
    plt.xticks(range(len(summary_df)), summary_df['model'])
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'attack_success_rates.png'), dpi=300)
    
    # 3. Transfer rate plot for target models
    plt.figure(figsize=(10, 6))
    transfer_data = summary_df[~summary_df['is_source']].copy()
    
    if len(transfer_data) > 0:
        bars = plt.bar(range(len(transfer_data)), transfer_data['transfer_rate'], color='green')
        
        # Add labels on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{height:.2f}', ha='center', va='bottom')
        
        plt.xlabel('Target Model')
        plt.ylabel('Transfer Attack Rate')
        plt.title(f'Transfer Attack Rates from {os.path.basename(source_model_name)}')
        plt.xticks(range(len(transfer_data)), transfer_data['model'])
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'transfer_rates.png'), dpi=300)
    
    # 4. ROC curves for each model
    plt.figure(figsize=(12, 10))
    
    for model_name, model_metrics in metrics.items():
        results_df = model_metrics['results_df']
        model_short_name = os.path.basename(model_name)
        
        # Get original and adversarial labels and scores
        original_labels = results_df['label'].values
        original_scores = results_df['original_similarity'].values
        adversarial_labels = results_df['label'].values
        adversarial_scores = results_df['adversarial_similarity'].values
        
        # Calculate ROC curve points
        thresholds = np.linspace(0, 1, 100)
        original_tpr = []
        original_fpr = []
        adversarial_tpr = []
        adversarial_fpr = []
        
        for t in thresholds:
            # Original
            original_predictions = (original_scores >= t).astype(int)
            tpr = np.sum((original_predictions == 1) & (original_labels == 1)) / max(1, np.sum(original_labels == 1))
            fpr = np.sum((original_predictions == 1) & (original_labels == 0)) / max(1, np.sum(original_labels == 0))
            original_tpr.append(tpr)
            original_fpr.append(fpr)
            
            # Adversarial
            adversarial_predictions = (adversarial_scores >= t).astype(int)
            tpr = np.sum((adversarial_predictions == 1) & (adversarial_labels == 1)) / max(1, np.sum(adversarial_labels == 1))
            fpr = np.sum((adversarial_predictions == 1) & (adversarial_labels == 0)) / max(1, np.sum(adversarial_labels == 0))
            adversarial_tpr.append(tpr)
            adversarial_fpr.append(fpr)
        
        # Plot ROC curves
        linestyle = '-' if model_name == source_model_name else '--'
        plt.plot(original_fpr, original_tpr, linestyle=linestyle, 
                label=f'{model_short_name} Original (EER={model_metrics["original_eer"]:.4f})')
        plt.plot(adversarial_fpr, adversarial_tpr, linestyle=linestyle, 
                label=f'{model_short_name} Adversarial (EER={model_metrics["adversarial_eer"]:.4f})')
    
    # Diagonal line
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.3)
    
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves For All Models')
    plt.grid(True, alpha=0.3)
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'roc_curves.png'), dpi=300)
    
    # Save summary to CSV
    summary_df.to_csv(os.path.join(output_dir, 'transfer_summary.csv'), index=False)
    
    # For each model, save detailed results
    for model_name, model_metrics in metrics.items():
        model_short_name = os.path.basename(model_name)
        model_metrics['results_df'].to_csv(os.path.join(output_dir, f'{model_short_name}_results.csv'), index=False)
    
    print(f"Visualization results saved to {output_dir}")

def create_transfer_summary(metrics, source_model_name):
    """Create a summary of transfer attack success rates"""
    # Prepare summary data
    summary = []
    
    # Get source model metrics
    source_metrics = metrics[source_model_name]
    source_asr = source_metrics['attack_success_rate']
    source_eer_original = source_metrics['original_eer']
    
    # Include source model in summary
    summary.append({
        'Source Model': os.path.basename(source_model_name),
        'Target Model': os.path.basename(source_model_name),
        'Source Model EER': source_eer_original,
        'Target Model EER': source_metrics['original_eer'],
        'Target Model Adversarial EER': source_metrics['adversarial_eer'],
        'Transfer Attack Success Rate': 1.0
    })
    
    # Include transfer targets
    for model_name, model_metrics in metrics.items():
        if model_name != source_model_name:  # Only include transfers from source to other models
            target_asr = model_metrics['attack_success_rate']
            transfer_rate = target_asr / source_asr if source_asr > 0 else 0
            
            summary.append({
                'Source Model': os.path.basename(source_model_name),
                'Target Model': os.path.basename(model_name),
                'Source Model EER': source_eer_original,
                'Target Model EER': model_metrics['original_eer'],
                'Target Model Adversarial EER': model_metrics['adversarial_eer'],
                'Transfer Attack Success Rate': transfer_rate
            })
    
    return pd.DataFrame(summary)

def main():
    parser = argparse.ArgumentParser(description='Evaluate Transfer Attacks between SpeechBrain Speaker Verification Models')
    
    # Model parameters
    parser.add_argument('--source_model_type', type=str, choices=["xvector", "resnet", "ecapa"], default="resnet",
                        help='Source model type that adversarial examples were created for')
    parser.add_argument('--custom_model_path', type=str, default=None,
                        help='Custom source model path (if not using the standard SpeechBrain models)')
    
    # Data parameters
    parser.add_argument('--pairs_file', type=str, default="data/veri_test.txt", 
                        help='File containing verification pairs')
    parser.add_argument('--audio_dir', type=str, default="data/wav", 
                        help='Directory containing original audio files')
    parser.add_argument('--adv_dir', type=str, default=None, 
                        help='Directory containing adversarial examples')
    parser.add_argument('--output_dir', type=str, default=None, 
                        help='Directory to save evaluation results')
    
    # Evaluation parameters
    parser.add_argument('--threshold', type=float, default=0.5, 
                        help='Similarity threshold for verification')
    parser.add_argument('--device', type=str, default="cuda" if torch.cuda.is_available() else "cpu", 
                        help='Device to run the evaluation on')
    parser.add_argument('--max_pairs', type=int, default=None,
                        help='Maximum number of pairs to evaluate')
    
    args = parser.parse_args()
    
    # Determine source model path
    source_model_path = args.custom_model_path if args.custom_model_path else MODEL_PATHS[args.source_model_type]
    
    # Define all models to evaluate
    model_types = ["xvector", "resnet", "ecapa"]
    
    # Set up directories
    if args.adv_dir is None:
        args.adv_dir = f"adv_examples_{args.source_model_type}"
    
    if args.output_dir is None:
        args.output_dir = f"transfer_results_{args.source_model_type}"
    
    # Print configuration
    print("\n" + "="*50)
    print("Transfer Attack Evaluation Configuration:")
    print("="*50)
    print(f"  Source Model Type: {args.source_model_type}")
    print(f"  Source Model Path: {source_model_path}")
    print(f"  Pairs file: {args.pairs_file}")
    print(f"  Audio directory: {args.audio_dir}")
    print(f"  Adversarial directory: {args.adv_dir}")
    print(f"  Output directory: {args.output_dir}")
    print(f"  Threshold: {args.threshold}")
    print(f"  Device: {args.device}")
    print(f"  Max pairs: {args.max_pairs if args.max_pairs else 'All'}")
    print("="*50 + "\n")
    
    # Check for necessary files and directories
    if not os.path.exists(args.audio_dir):
        print(f"Error: Audio directory '{args.audio_dir}' does not exist.")
        return
    
    if not os.path.exists(args.adv_dir):
        print(f"Error: Adversarial examples directory '{args.adv_dir}' does not exist.")
        return
    
    if not os.path.exists(args.pairs_file):
        print(f"Error: Pairs file '{args.pairs_file}' does not exist.")
        return
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load all models
    models = {}
    for model_type in model_types:
        model_path = MODEL_PATHS[model_type]
        print(f"Loading model: {model_path}")
        try:
            savedir = f"pretrained_models/{model_type}"
            model = EncoderClassifier.from_hparams(
                source=model_path, 
                savedir=savedir,
                run_opts={"device": args.device}
            )
            models[model_path] = model
            print(f"  Model {model_path} loaded successfully")
        except Exception as e:
            print(f"  Error loading model {model_path}: {e}")
    
    # Evaluate transfer attacks
    print("Evaluating transfer attacks...")
    metrics = evaluate_transfer_attacks(
        source_model_path, models, args.pairs_file, args.audio_dir, args.adv_dir, 
        args.device, args.threshold, args.max_pairs
    )
    
    # Print summary for each model
    print("\nTransfer Attack Results:")
    for model_name, model_metrics in metrics.items():
        model_short_name = os.path.basename(model_name)
        print(f"\n  Results for {model_short_name}:")
        print(f"    Original EER: {model_metrics['original_eer']:.4f}")
        print(f"    Adversarial EER: {model_metrics['adversarial_eer']:.4f}")
        print(f"    Attack success rate: {model_metrics['attack_success_rate']:.4f}")
        print(f"    Average SNR: {model_metrics['average_snr']:.2f} dB")
    
    # Create and print transfer summary
    transfer_summary = create_transfer_summary(metrics, source_model_path)
    print("\nTransfer Attack Summary:")
    print(transfer_summary.to_string(index=False))
    
    # Generate visualizations
    print("\nGenerating transfer visualizations...")
    visualize_transfer_results(metrics, source_model_path, args.output_dir)
    
    # Save transfer summary
    transfer_summary.to_csv(os.path.join(args.output_dir, 'transfer_summary.csv'), index=False)
    
    # Generate combined report
    summary_file = os.path.join(args.output_dir, 'summary_report.txt')
    with open(summary_file, 'w') as f:
        f.write("=== TRANSFER ATTACK SUMMARY ===\n\n")
        
        # Source model info
        src_name = os.path.basename(source_model_path)
        f.write(f"Source Model: {source_model_path}\n")
        f.write(f"Source Model EER: {metrics[source_model_path]['original_eer']:.4f}\n")
        f.write(f"Source Model Attack Success Rate: {metrics[source_model_path]['attack_success_rate']:.4f}\n")
        f.write(f"Average SNR: {metrics[source_model_path]['average_snr']:.2f} dB\n\n")
        
        # Show transfers to other models
        for model_name, model_metrics in metrics.items():
            if model_name != source_model_path:
                target_name = os.path.basename(model_name)
                source_asr = metrics[source_model_path]['attack_success_rate']
                transfer_rate = model_metrics['attack_success_rate'] / source_asr if source_asr > 0 else 0
                
                f.write(f"--- {src_name} to {target_name} Results ---\n")
                f.write(f"Source Model EER: {metrics[source_model_path]['original_eer']:.4f}\n")
                f.write(f"Target Model EER: {model_metrics['original_eer']:.4f}\n")
                f.write(f"Target Model Adversarial EER: {model_metrics['adversarial_eer']:.4f}\n")
                f.write(f"Transfer Attack Success Rate: {transfer_rate:.4f}\n\n")
        
        # Simplified summary section
        f.write("=== SUMMARY ===\n")
        f.write(f"1. {src_name} model:\n")
        f.write(f"   EER: {metrics[source_model_path]['original_eer']:.4f}\n")
        f.write("   Transfer to other models:\n")
        
        for model_name, model_metrics in metrics.items():
            if model_name != source_model_path:
                target_name = os.path.basename(model_name)
                source_asr = metrics[source_model_path]['attack_success_rate']
                transfer_rate = model_metrics['attack_success_rate'] / source_asr if source_asr > 0 else 0
                
                f.write(f"   -> {target_name}: {model_metrics['adversarial_eer']:.4f} ")
                f.write(f"(ASR: {transfer_rate:.4f})\n")
    
    print(f"Comprehensive summary report saved to {summary_file}")

if __name__ == "__main__":
    main()

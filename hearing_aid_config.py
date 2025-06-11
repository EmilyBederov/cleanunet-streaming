# Configuration and training adaptations for hearing aid CleanUNet

import json
import torch
from stft_loss import MultiResolutionSTFTLoss


# Hearing aid optimized configuration
HEARING_AID_CONFIG = {
    "train_config": {
        "exp_path": "hearing_aid_cleanunet",
        "log": {
            "directory": "./logs",
            "ckpt_iter": "max",
            "iters_per_ckpt": 1000,
            "iters_per_valid": 100
        },
        "optimization": {
            "n_iters": 50000,
            "learning_rate": 1e-4,
            "batch_size_per_gpu": 8  # Smaller batch size for memory efficiency
        },
        "loss_config": {
            "stft_lambda": 1.0,
            "l1_lambda": 1.0,
            "stft_config": {
                # Short STFT windows for low latency (~10ms at 16kHz)
                "fft_sizes": [256, 512, 1024],      # Reduced from original
                "hop_sizes": [32, 64, 128],         # Reduced for finer resolution
                "win_lengths": [160, 320, 640],     # ~10ms, 20ms, 40ms at 16kHz
                "window": "hann_window",
                "sc_lambda": 1.0,
                "mag_lambda": 1.0,
                "band": "full"
            }
        }
    },
    
    "network_config": {
        "channels_input": 1,
        "channels_output": 1,
        "channels_H": 32,          # Reduced for efficiency
        "max_H": 256,              # Reduced for efficiency  
        "encoder_n_layers": 6,     # Reduced for lower latency
        "kernel_size": 3,          # Smaller kernel for lower latency
        "stride": 2,
        "tsfm_n_layers": 2,        # Reduced for efficiency
        "tsfm_n_head": 4,          # Reduced for efficiency
        "tsfm_d_model": 256,       # Reduced for efficiency
        "tsfm_d_inner": 512,       # Reduced for efficiency
        "use_group_norm": True
    },
    
    "trainset_config": {
        "train_dir": "./data/train",
        "test_dir": "./data/test", 
        "crop_length_sec": 1.0,    # Shorter segments for hearing aid training
        "sample_rate": 16000
    },
    
    "dist_config": {
        "dist_backend": "nccl",
        "dist_url": "tcp://localhost:54321"
    }
}


class HearingAidLoss:
    """Enhanced loss function for hearing aid applications"""
    
    def __init__(self, config):
        self.l1_lambda = config.get("l1_lambda", 1.0)
        self.stft_lambda = config.get("stft_lambda", 1.0)
        self.perceptual_lambda = config.get("perceptual_lambda", 0.1)
        
        # Multi-resolution STFT loss with short windows
        if self.stft_lambda > 0:
            self.stft_loss = MultiResolutionSTFTLoss(**config["stft_config"])
        
        # Perceptual loss for hearing aid applications
        self.freq_weights = self._create_hearing_aid_weights()
        
    def _create_hearing_aid_weights(self):
        """Create frequency weights based on hearing aid requirements"""
        # Emphasize speech frequencies (300Hz - 8kHz)
        # This is a simplified model - in practice you'd use audiogram data
        freqs = torch.linspace(0, 8000, 129)  # For 256-point FFT
        weights = torch.ones_like(freqs)
        
        # Boost speech frequencies
        speech_mask = (freqs >= 300) & (freqs <= 8000)
        weights[speech_mask] *= 2.0
        
        # Boost most critical speech frequencies (1-4kHz)
        critical_mask = (freqs >= 1000) & (freqs <= 4000)
        weights[critical_mask] *= 1.5
        
        return weights.unsqueeze(0).unsqueeze(0)  # Shape for broadcasting
        
    def compute_perceptual_loss(self, pred, target):
        """Compute frequency-weighted perceptual loss"""
        # Simple spectral loss weighted by hearing importance
        pred_fft = torch.fft.rfft(pred, n=256, dim=-1)
        target_fft = torch.fft.rfft(target, n=256, dim=-1)
        
        pred_mag = torch.abs(pred_fft)
        target_mag = torch.abs(target_fft)
        
        # Apply frequency weights
        weights = self.freq_weights.to(pred.device)
        weighted_loss = torch.mean(weights * (pred_mag - target_mag) ** 2)
        
        return weighted_loss
        
    def __call__(self, model, data, **kwargs):
        """Compute total loss"""
        clean_audio, noisy_audio = data
        
        # Forward pass
        pred_audio = model(noisy_audio)
        
        # Ensure same length (handle any padding differences)
        min_length = min(pred_audio.shape[-1], clean_audio.shape[-1])
        pred_audio = pred_audio[..., :min_length]
        clean_audio = clean_audio[..., :min_length]
        
        # L1 loss
        l1_loss = torch.nn.functional.l1_loss(pred_audio, clean_audio)
        
        # STFT loss
        stft_sc_loss, stft_mag_loss = 0, 0
        if self.stft_lambda > 0:
            # Flatten for STFT loss (B, C, T) -> (B*C, T)
            pred_flat = pred_audio.view(-1, pred_audio.size(-1))
            clean_flat = clean_audio.view(-1, clean_audio.size(-1))
            stft_sc_loss, stft_mag_loss = self.stft_loss(pred_flat, clean_flat)
        
        # Perceptual loss
        perceptual_loss = 0
        if self.perceptual_lambda > 0:
            perceptual_loss = self.compute_perceptual_loss(pred_audio, clean_audio)
        
        # Total loss
        total_loss = (self.l1_lambda * l1_loss + 
                     self.stft_lambda * (stft_sc_loss + stft_mag_loss) +
                     self.perceptual_lambda * perceptual_loss)
        
        loss_dict = {
            "total_loss": total_loss,
            "l1_loss": l1_loss,
            "stft_sc_loss": stft_sc_loss,
            "stft_mag_loss": stft_mag_loss,
            "perceptual_loss": perceptual_loss
        }
        
        return total_loss, loss_dict


def create_hearing_aid_trainer():
    """Modified training setup for hearing aid applications"""
    
    def train_step(model, data, optimizer, loss_fn, device):
        """Single training step optimized for hearing aids"""
        model.train()
        
        clean_audio, noisy_audio = data
        clean_audio = clean_audio.to(device)
        noisy_audio = noisy_audio.to(device)
        
        # Data augmentation for hearing aids
        clean_audio, noisy_audio = hearing_aid_augmentation(clean_audio, noisy_audio)
        
        # Forward pass
        optimizer.zero_grad()
        loss, loss_dict = loss_fn(model, (clean_audio, noisy_audio))
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        return loss.item(), loss_dict
    
    return train_step


def hearing_aid_augmentation(clean_audio, noisy_audio):
    """Data augmentation specific to hearing aid applications"""
    
    # 1. Volume scaling (simulate different input levels)
    scale = torch.rand(clean_audio.shape[0], 1, 1) * 0.8 + 0.6  # 0.6 to 1.4
    clean_audio = clean_audio * scale.to(clean_audio.device)
    noisy_audio = noisy_audio * scale.to(noisy_audio.device)
    
    # 2. Add slight compression (simulate hearing aid preprocessing)
    clean_audio = torch.tanh(clean_audio * 2) / 2
    noisy_audio = torch.tanh(noisy_audio * 2) / 2
    
    # 3. Frequency emphasis (simulate hearing loss compensation)
    if torch.rand(1) > 0.5:  # 50% chance
        clean_audio, noisy_audio = apply_frequency_emphasis(clean_audio, noisy_audio)
    
    return clean_audio, noisy_audio


def apply_frequency_emphasis(clean_audio, noisy_audio):
    """Apply frequency emphasis to simulate hearing aid processing"""
    
    # Simple high-frequency emphasis using a high-pass filter
    # In practice, you'd use proper audiogram-based processing
    
    # Create a simple high-pass filter
    b, a = torch.tensor([0.95, -0.95]), torch.tensor([1.0, -0.9])
    
    # Apply filter (simplified - in practice use proper filtering)
    # This is just a demonstration
    return clean_audio, noisy_audio


def evaluate_hearing_aid_metrics(model, test_loader, device):
    """Evaluation metrics specific to hearing aid performance"""
    
    model.eval()
    total_loss = 0
    total_pesq = 0
    total_stoi = 0
    num_batches = 0
    
    with torch.no_grad():
        for clean_audio, noisy_audio in test_loader:
            clean_audio = clean_audio.to(device)
            noisy_audio = noisy_audio.to(device)
            
            # Forward pass
            pred_audio = model(noisy_audio)
            
            # Basic loss
            min_length = min(pred_audio.shape[-1], clean_audio.shape[-1])
            pred_audio = pred_audio[..., :min_length]
            clean_audio = clean_audio[..., :min_length]
            
            loss = torch.nn.functional.l1_loss(pred_audio, clean_audio)
            total_loss += loss.item()
            
            # In practice, you would compute PESQ and STOI here
            # These require specialized libraries like pesq and pystoi
            
            num_batches += 1
    
    metrics = {
        "avg_loss": total_loss / num_batches,
        "avg_pesq": total_pesq / num_batches if total_pesq > 0 else 0,
        "avg_stoi": total_stoi / num_batches if total_stoi > 0 else 0
    }
    
    return metrics


def save_hearing_aid_config(config_path="configs/hearing_aid.json"):
    """Save the hearing aid configuration"""
    with open(config_path, 'w') as f:
        json.dump(HEARING_AID_CONFIG, f, indent=2)
    print(f"Hearing aid configuration saved to {config_path}")


def load_hearing_aid_config(config_path="configs/hearing_aid.json"):
    """Load the hearing aid configuration"""
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config


# Example usage
if __name__ == "__main__":
    # Save configuration
    save_hearing_aid_config()
    
    # Test loss function
    config = HEARING_AID_CONFIG["train_config"]["loss_config"]
    loss_fn = HearingAidLoss(config)
    
    # Create dummy model and data
    from causal_cleanunet_backup import create_hearing_aid_cleanunet
    model = create_hearing_aid_cleanunet()
    
    # Test data
    clean_audio = torch.randn(2, 1, 1600)  # 100ms at 16kHz
    noisy_audio = clean_audio + 0.1 * torch.randn_like(clean_audio)
    
    # Compute loss
    loss, loss_dict = loss_fn(model, (clean_audio, noisy_audio))
    
    print("Loss computation test:")
    print(f"Total loss: {loss.item():.4f}")
    for key, value in loss_dict.items():
        if isinstance(value, torch.Tensor):
            print(f"{key}: {value.item():.4f}")
        else:
            print(f"{key}: {value}")
    
    print("\nHearing aid configuration created successfully!")
    print("Key features:")
    print("- Causal convolutions for real-time processing")
    print("- GroupNorm instead of BatchNorm")
    print("- Short STFT windows (~10ms)")
    print("- Reduced model size for efficiency")
    print("- Hearing-aid specific loss function")
    print("- Streaming processing support")

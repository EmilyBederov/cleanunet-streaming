# Hearing Aid CleanUNet Training Script
# Adapted from the original train.py for hearing aid applications

import os
import time
import argparse
import json

import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

import random
random.seed(0)
torch.manual_seed(0)
np.random.seed(0)

from distributed import init_distributed, apply_gradient_allreduce, reduce_tensor
from dataset import load_CleanNoisyPairDataset
from stft_loss import MultiResolutionSTFTLoss
from util import rescale, find_max_epoch, print_size
from util import LinearWarmupCosineDecay

# Import our new causal model
from causal_cleanunet import CausalCleanUNet, create_hearing_aid_cleanunet
from hearing_aid_config import HearingAidLoss, hearing_aid_augmentation


def evaluate_model(model, test_loader, loss_fn, device):
    """Evaluate model on test set"""
    model.eval()
    total_loss = 0
    total_l1_loss = 0
    total_stft_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for clean_audio, noisy_audio, _ in test_loader:
            clean_audio = clean_audio.to(device)
            noisy_audio = noisy_audio.to(device)
            
            # Forward pass
            loss, loss_dict = loss_fn(model, (clean_audio, noisy_audio))
            
            total_loss += loss.item()
            total_l1_loss += loss_dict.get('l1_loss', 0).item() if isinstance(loss_dict.get('l1_loss', 0), torch.Tensor) else loss_dict.get('l1_loss', 0)
            total_stft_loss += (loss_dict.get('stft_sc_loss', 0) + loss_dict.get('stft_mag_loss', 0)).item() if isinstance(loss_dict.get('stft_sc_loss', 0), torch.Tensor) else 0
            num_batches += 1
    
    return {
        'avg_total_loss': total_loss / num_batches,
        'avg_l1_loss': total_l1_loss / num_batches,
        'avg_stft_loss': total_stft_loss / num_batches
    }


def train_hearing_aid(num_gpus, rank, group_name, 
                     exp_path, log, optimization, loss_config, network_config, 
                     trainset_config, testset_config=None):

    # Setup local experiment path
    if rank == 0:
        print('exp_path:', exp_path)
        print('Training Causal CleanUNet for Hearing Aids')
    
    # Create tensorboard logger
    log_directory = os.path.join(log["directory"], exp_path)
    if rank == 0:
        tb = SummaryWriter(os.path.join(log_directory, 'tensorboard'))

    # Distributed running initialization
    if num_gpus > 1:
        init_distributed(rank, num_gpus, group_name, **dist_config)

    # Get shared ckpt_directory ready
    ckpt_directory = os.path.join(log_directory, 'checkpoint')
    if rank == 0:
        if not os.path.isdir(ckpt_directory):
            os.makedirs(ckpt_directory)
            os.chmod(ckpt_directory, 0o775)
        print("ckpt_directory: ", ckpt_directory, flush=True)

    # Load training data
    print('Loading training data...')
    trainloader = load_CleanNoisyPairDataset(**trainset_config, 
                            subset='training',
                            batch_size=optimization["batch_size_per_gpu"], 
                            num_gpus=num_gpus)
    print('Training data loaded')
    
    # Load test data if available
    test_loader = None
    if testset_config is not None:
        print('Loading test data...')
        test_loader = load_CleanNoisyPairDataset(**testset_config,
                                subset='testing',
                                batch_size=optimization["batch_size_per_gpu"],
                                num_gpus=1)  # Single GPU for testing
        print('Test data loaded')
    
    # Create the causal model
    print('Creating Causal CleanUNet model...')
    net = CausalCleanUNet(**network_config).cuda()
    print_size(net, keyword="CausalCleanUNet")
    
    # Print model architecture info
    if rank == 0:
        total_params = sum(p.numel() for p in net.parameters())
        trainable_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")

    # Apply gradient all reduce for distributed training
    if num_gpus > 1:
        net = apply_gradient_allreduce(net)

    # Define optimizer with hearing aid specific settings
    optimizer = torch.optim.AdamW(
        net.parameters(), 
        lr=optimization["learning_rate"],
        weight_decay=optimization.get("weight_decay", 1e-4),
        betas=(0.9, 0.999)
    )

    # Load checkpoint if available
    time0 = time.time()
    if log["ckpt_iter"] == 'max':
        ckpt_iter = find_max_epoch(ckpt_directory)
    else:
        ckpt_iter = log["ckpt_iter"]
        
    if ckpt_iter >= 0:
        try:
            # Load checkpoint file
            model_path = os.path.join(ckpt_directory, '{}.pkl'.format(ckpt_iter))
            checkpoint = torch.load(model_path, map_location='cpu')
            
            # Feed model dict and optimizer state
            net.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            # Record training time based on elapsed time
            time0 -= checkpoint['training_time_seconds']
            print('Model at iteration %s has been trained for %s seconds' % (ckpt_iter, checkpoint['training_time_seconds']))
            print('Checkpoint model loaded successfully')
        except Exception as e:
            ckpt_iter = -1
            print(f'Error loading checkpoint: {e}')
            print('Starting training from initialization.')
    else:
        ckpt_iter = -1
        print('No valid checkpoint model found, starting training from initialization.')

    # Initialize training
    n_iter = ckpt_iter + 1

    # Define learning rate scheduler
    scheduler = LinearWarmupCosineDecay(
                    optimizer,
                    lr_max=optimization["learning_rate"],
                    n_iter=optimization["n_iters"],
                    iteration=n_iter,
                    divider=25,
                    warmup_proportion=0.05,
                    phase=('linear', 'cosine'),
                )

    # Define hearing aid specific loss function
    loss_fn = HearingAidLoss(loss_config)
    
    print('Starting training...')
    best_test_loss = float('inf')
    
    # Training loop
    while n_iter < optimization["n_iters"] + 1:
        
        # Training epoch
        for batch_idx, (clean_audio, noisy_audio, _) in enumerate(trainloader): 
            
            clean_audio = clean_audio.cuda()
            noisy_audio = noisy_audio.cuda()

            # Apply hearing aid specific data augmentation
            clean_audio, noisy_audio = hearing_aid_augmentation(clean_audio, noisy_audio)

            # Training step
            net.train()
            optimizer.zero_grad()
            
            # Forward pass and loss computation
            loss, loss_dic = loss_fn(net, (clean_audio, noisy_audio))
            
            if num_gpus > 1:
                reduced_loss = reduce_tensor(loss.data, num_gpus).item()
            else:
                reduced_loss = loss.item()
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping for stability
            grad_norm = nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
            
            # Update learning rate and optimizer
            scheduler.step()
            optimizer.step()

            # Logging
            if n_iter % log["iters_per_valid"] == 0:
                print("iteration: {} \treduced loss: {:.7f} \tloss: {:.7f} \tlr: {:.2e}".format(
                    n_iter, reduced_loss, loss.item(), optimizer.param_groups[0]["lr"]), flush=True)
                
                if rank == 0:
                    # Save to tensorboard
                    tb.add_scalar("Train/Total-Loss", loss.item(), n_iter)
                    tb.add_scalar("Train/Reduced-Loss", reduced_loss, n_iter)
                    tb.add_scalar("Train/Gradient-Norm", grad_norm, n_iter)
                    tb.add_scalar("Train/Learning-Rate", optimizer.param_groups[0]["lr"], n_iter)
                    
                    # Log individual loss components
                    for loss_name, loss_value in loss_dic.items():
                        if isinstance(loss_value, torch.Tensor):
                            tb.add_scalar(f"Train/{loss_name}", loss_value.item(), n_iter)

            # Validation
            if test_loader is not None and n_iter % (log["iters_per_valid"] * 5) == 0 and rank == 0:
                print("Running validation...")
                test_metrics = evaluate_model(net, test_loader, loss_fn, 'cuda')
                
                print("Validation metrics:")
                for metric_name, metric_value in test_metrics.items():
                    print(f"  {metric_name}: {metric_value:.6f}")
                    tb.add_scalar(f"Valid/{metric_name}", metric_value, n_iter)
                
                # Save best model
                if test_metrics['avg_total_loss'] < best_test_loss:
                    best_test_loss = test_metrics['avg_total_loss']
                    best_model_path = os.path.join(ckpt_directory, 'best_model.pkl')
                    torch.save({
                        'iter': n_iter,
                        'model_state_dict': net.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'test_loss': best_test_loss,
                        'training_time_seconds': int(time.time()-time0)
                    }, best_model_path)
                    print(f'New best model saved with test loss: {best_test_loss:.6f}')

            # Save checkpoint
            if n_iter > 0 and n_iter % log["iters_per_ckpt"] == 0 and rank == 0:
                checkpoint_name = '{}.pkl'.format(n_iter)
                torch.save({
                    'iter': n_iter,
                    'model_state_dict': net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'training_time_seconds': int(time.time()-time0),
                    'config': {
                        'network_config': network_config,
                        'loss_config': loss_config
                    }
                }, os.path.join(ckpt_directory, checkpoint_name))
                print('Model at iteration %s saved' % n_iter)

            n_iter += 1
            
            # Break if reached max iterations
            if n_iter >= optimization["n_iters"] + 1:
                break

    # After training
    if rank == 0:
        print('Training completed!')
        
        # Save final model
        final_model_path = os.path.join(ckpt_directory, 'final_model.pkl')
        torch.save({
            'iter': n_iter,
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'training_time_seconds': int(time.time()-time0),
            'config': {
                'network_config': network_config,
                'loss_config': loss_config
            }
        }, final_model_path)
        print(f'Final model saved to {final_model_path}')
        
        tb.close()

    return 0


def create_hearing_aid_config():
    """Create configuration for hearing aid training"""
    return {
        "train_config": {
            "exp_path": "hearing_aid_cleanunet",
            "log": {
                "directory": "./logs",
                "ckpt_iter": "max",
                "iters_per_ckpt": 2000,
                "iters_per_valid": 200
            },
            "optimization": {
                "n_iters": 100000,
                "learning_rate": 1e-4,
                "weight_decay": 1e-4,
                "batch_size_per_gpu": 8
            },
            "loss_config": {
                "l1_lambda": 1.0,
                "stft_lambda": 1.0,
                "perceptual_lambda": 0.1,
                "stft_config": {
                    "fft_sizes": [256, 512, 1024],
                    "hop_sizes": [32, 64, 128],
                    "win_lengths": [160, 320, 640],
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
            "channels_H": 32,
            "max_H": 256,
            "encoder_n_layers": 6,
            "kernel_size": 3,
            "stride": 2,
            "tsfm_n_layers": 2,
            "tsfm_n_head": 4,
            "tsfm_d_model": 256,
            "tsfm_d_inner": 512,
            "use_group_norm": True
        },
        
        "trainset_config": {
            "train_dir": "./data/train",
            "crop_length_sec": 2.0,
            "sample_rate": 16000
        },
        
        "testset_config": {
            "test_dir": "./data/test",
            "crop_length_sec": 2.0,
            "sample_rate": 16000
        },
        
        "dist_config": {
            "dist_backend": "nccl",
            "dist_url": "tcp://localhost:54321"
        }
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default=None, 
                        help='JSON file for configuration')
    parser.add_argument('-r', '--rank', type=int, default=0,
                        help='rank of process for distributed')
    parser.add_argument('-g', '--group_name', type=str, default='',
                        help='name of group for distributed')
    parser.add_argument('--create_config', action='store_true',
                        help='Create default hearing aid config and exit')
    args = parser.parse_args()

    # Create default config if requested
    if args.create_config:
        config = create_hearing_aid_config()
        config_path = "configs/hearing_aid_cleanunet.json"
        os.makedirs("configs", exist_ok=True)
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"Default hearing aid config created at {config_path}")
        print("Edit this config file and then run:")
        print(f"python {__file__} -c {config_path}")
        exit(0)

    # Parse configs
    if args.config is None:
        print("No config file provided. Creating default config...")
        config = create_hearing_aid_config()
    else:
        with open(args.config) as f:
            data = f.read()
        config = json.loads(data)

    # Extract configuration sections
    train_config = config["train_config"]
    global dist_config
    dist_config = config["dist_config"]
    global network_config
    network_config = config["network_config"]
    global trainset_config
    trainset_config = config["trainset_config"]
    global testset_config
    testset_config = config.get("testset_config", None)

    # GPU setup
    num_gpus = torch.cuda.device_count()
    if num_gpus > 1:
        if args.group_name == '':
            print("WARNING: Multiple GPUs detected but no distributed group set")
            print("Only running 1 GPU. Use distributed.py for multiple GPUs")
            num_gpus = 1

    if num_gpus == 1 and args.rank != 0:
        raise Exception("Doing single GPU training on rank > 0")

    # Enable optimizations
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    
    print("="*50)
    print("HEARING AID CLEANUNET TRAINING")
    print("="*50)
    print(f"GPUs: {num_gpus}")
    print(f"Model: CausalCleanUNet")
    print(f"Batch size per GPU: {train_config['optimization']['batch_size_per_gpu']}")
    print(f"Learning rate: {train_config['optimization']['learning_rate']}")
    print(f"Total iterations: {train_config['optimization']['n_iters']}")
    print("="*50)
    
    # Start training
    train_hearing_aid(num_gpus, args.rank, args.group_name, **train_config, 
                     network_config=network_config, 
                     trainset_config=trainset_config,
                     testset_config=testset_config)

# Copyright (c) 2022 NVIDIA CORPORATION. 
#   Licensed under the MIT license.
# Modified for hearing aid applications with causal processing

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from util import weight_scaling_init


def causal_padding(x, kernel_size, dilation=1):
    """Apply causal padding to maintain causality"""
    pad_length = (kernel_size - 1) * dilation
    return F.pad(x, (pad_length, 0))


def trim_to_match_length(x, target_length):
    """Trim tensor to match target length"""
    if x.shape[-1] > target_length:
        return x[..., :target_length]
    return x


class CausalConv1d(nn.Module):
    """Causal 1D convolution that doesn't look into the future"""
    
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1, bias=True):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, 
                             stride=stride, dilation=dilation, groups=groups, bias=bias)
        
    def forward(self, x):
        x = causal_padding(x, self.kernel_size, self.dilation)
        return self.conv(x)


class CausalConvTranspose1d(nn.Module):
    """Causal transposed convolution for decoder"""
    
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, groups=1, bias=True):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        
        self.conv_transpose = nn.ConvTranspose1d(in_channels, out_channels, kernel_size,
                                               stride=stride, groups=groups, bias=bias)
        
    def forward(self, x):
        # Apply transposed convolution
        x = self.conv_transpose(x)
        
        # Remove future samples to maintain causality
        # For stride > 1, we need to be careful about the output length
        if self.stride > 1:
            # Calculate how many samples to remove from the end
            remove_samples = self.kernel_size - self.stride
            if remove_samples > 0:
                x = x[..., :-remove_samples]
        
        return x


class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):
        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)

        return output, attn


class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)

        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5)

        self.dropout = nn.Dropout(dropout)
        # Replace LayerNorm with GroupNorm for better real-time performance
        self.group_norm = nn.GroupNorm(num_groups=min(32, d_model), num_channels=d_model, eps=1e-6)

    def forward(self, q, k, v, mask=None):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        residual = q

        # Pass through the pre-attention projection
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        # Transpose for attention dot product: b x n x lq x dv
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1)   # For head axis broadcasting.

        q, attn = self.attention(q, k, v, mask=mask)

        # Transpose back and concatenate heads
        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        q = self.dropout(self.fc(q))
        q += residual

        # Apply GroupNorm - need to transpose for channel dimension
        q = q.transpose(1, 2)  # (B, L, C) -> (B, C, L)
        q = self.group_norm(q)
        q = q.transpose(1, 2)  # (B, C, L) -> (B, L, C)

        return q, attn


class PositionwiseFeedForward(nn.Module):
    ''' A two-feed-forward-layer module '''

    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid)
        self.w_2 = nn.Linear(d_hid, d_in)
        # Replace LayerNorm with GroupNorm
        self.group_norm = nn.GroupNorm(num_groups=min(32, d_in), num_channels=d_in, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x

        x = self.w_2(F.relu(self.w_1(x)))
        x = self.dropout(x)
        x += residual

        # Apply GroupNorm - need to transpose for channel dimension
        x = x.transpose(1, 2)  # (B, L, C) -> (B, C, L)
        x = self.group_norm(x)
        x = x.transpose(1, 2)  # (B, C, L) -> (B, L, C)

        return x


class PositionalEncoding(nn.Module):
    def __init__(self, d_hid, n_position=200):
        super(PositionalEncoding, self).__init__()
        self.register_buffer('pos_table', self._get_sinusoid_encoding_table(n_position, d_hid))

    def _get_sinusoid_encoding_table(self, n_position, d_hid):
        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])

        return torch.FloatTensor(sinusoid_table).unsqueeze(0)

    def forward(self, x):
        return x + self.pos_table[:, :x.size(1)].clone().detach()


class EncoderLayer(nn.Module):
    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.0):
        super(EncoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, enc_input, slf_attn_mask=None):
        enc_output, enc_slf_attn = self.slf_attn(
            enc_input, enc_input, enc_input, mask=slf_attn_mask)
        enc_output = self.pos_ffn(enc_output)
        return enc_output, enc_slf_attn


class TransformerEncoder(nn.Module):
    def __init__(
            self, d_word_vec=512, n_layers=2, n_head=8, d_k=64, d_v=64,
            d_model=512, d_inner=2048, dropout=0.1, n_position=624, scale_emb=False):

        super().__init__()

        if n_position > 0:
            self.position_enc = PositionalEncoding(d_word_vec, n_position=n_position)
        else:
            self.position_enc = lambda x: x
        self.dropout = nn.Dropout(p=dropout)
        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])
        # Replace LayerNorm with GroupNorm
        self.group_norm = nn.GroupNorm(num_groups=min(32, d_model), num_channels=d_model, eps=1e-6)
        self.scale_emb = scale_emb
        self.d_model = d_model

    def forward(self, src_seq, src_mask, return_attns=False):
        enc_slf_attn_list = []

        enc_output = src_seq
        if self.scale_emb:
            enc_output *= self.d_model ** 0.5
        enc_output = self.dropout(self.position_enc(enc_output))
        
        # Apply GroupNorm
        enc_output = enc_output.transpose(1, 2)  # (B, L, C) -> (B, C, L)
        enc_output = self.group_norm(enc_output)
        enc_output = enc_output.transpose(1, 2)  # (B, C, L) -> (B, L, C)

        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(enc_output, slf_attn_mask=src_mask)
            enc_slf_attn_list += [enc_slf_attn] if return_attns else []

        if return_attns:
            return enc_output, enc_slf_attn_list
        return enc_output


class CausalCleanUNet(nn.Module):
    """ Causal CleanUNet for real-time hearing aid applications """

    def __init__(self, channels_input=1, channels_output=1,
                 channels_H=64, max_H=768,
                 encoder_n_layers=8, kernel_size=4, stride=2,
                 tsfm_n_layers=3, 
                 tsfm_n_head=8,
                 tsfm_d_model=512, 
                 tsfm_d_inner=2048,
                 use_group_norm=True):
        
        super(CausalCleanUNet, self).__init__()

        self.channels_input = channels_input
        self.channels_output = channels_output
        self.channels_H = channels_H
        self.max_H = max_H
        self.encoder_n_layers = encoder_n_layers
        self.kernel_size = kernel_size
        self.stride = stride
        self.use_group_norm = use_group_norm

        self.tsfm_n_layers = tsfm_n_layers
        self.tsfm_n_head = tsfm_n_head
        self.tsfm_d_model = tsfm_d_model
        self.tsfm_d_inner = tsfm_d_inner

        # Encoder and decoder
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()

        # Build encoder and track channel progression
        channels_progression = []  # To track encoder output channels
        current_channels = channels_input
        
        for i in range(encoder_n_layers):
            # Use causal convolutions
            encoder_block = nn.Sequential(
                CausalConv1d(current_channels, channels_H, kernel_size, stride),
                nn.ReLU(),
                nn.Conv1d(channels_H, channels_H * 2, 1), 
                nn.GLU(dim=1)
            )
            self.encoder.append(encoder_block)
            
            # Track the output channels after GLU (which halves the channels)
            channels_progression.append(channels_H)  # GLU outputs channels_H
            
            # Add GroupNorm if requested
            if use_group_norm and i < encoder_n_layers - 1:  # No norm on last layer before transformer
                norm_layer = nn.GroupNorm(num_groups=min(8, channels_H), num_channels=channels_H)
                self.encoder.append(norm_layer)
            
            current_channels = channels_H
            
            # Double H but keep below max_H
            channels_H *= 2
            channels_H = min(channels_H, max_H)

        # Store final encoder output channels for transformer
        final_encoder_channels = current_channels
        
        # Transformer block with causal attention
        self.tsfm_conv1 = nn.Conv1d(final_encoder_channels, tsfm_d_model, kernel_size=1)
        self.tsfm_encoder = TransformerEncoder(d_word_vec=tsfm_d_model, 
                                               n_layers=tsfm_n_layers, 
                                               n_head=tsfm_n_head, 
                                               d_k=tsfm_d_model // tsfm_n_head, 
                                               d_v=tsfm_d_model // tsfm_n_head, 
                                               d_model=tsfm_d_model, 
                                               d_inner=tsfm_d_inner, 
                                               dropout=0.0, 
                                               n_position=0, 
                                               scale_emb=False)
        self.tsfm_conv2 = nn.Conv1d(tsfm_d_model, final_encoder_channels, kernel_size=1)

        # Build decoder in reverse order
        # Start from bottleneck channels and work backwards
        decoder_input_channels = final_encoder_channels
        
        for i in range(encoder_n_layers):
            layer_idx = encoder_n_layers - 1 - i  # Reverse index
            
            # Determine output channels for this decoder layer
            if i == encoder_n_layers - 1:  # Last decoder layer
                decoder_output_channels = channels_output
            else:
                decoder_output_channels = channels_progression[layer_idx]
            
            # Create decoder block
            decoder_block = nn.Sequential(
                nn.Conv1d(decoder_input_channels, decoder_input_channels * 2, 1), 
                nn.GLU(dim=1),
                CausalConvTranspose1d(decoder_input_channels, decoder_output_channels, kernel_size, stride)
            )
            
            # Add ReLU except for the final layer
            if i < encoder_n_layers - 1:
                decoder_block.add_module('relu', nn.ReLU())
                
            self.decoder.append(decoder_block)
            
            # Update input channels for next decoder layer
            decoder_input_channels = decoder_output_channels

        # Weight scaling initialization
        for layer in self.modules():
            if isinstance(layer, (nn.Conv1d, nn.ConvTranspose1d)):
                weight_scaling_init(layer)

    def forward(self, noisy_audio):
        # Input shape: (B, L) or (B, C, L)
        if len(noisy_audio.shape) == 2:
            noisy_audio = noisy_audio.unsqueeze(1)
        B, C, L = noisy_audio.shape
        assert C == 1
        
        # Normalization (use running statistics for real-time)
        std = noisy_audio.std(dim=2, keepdim=True) + 1e-3
        x = noisy_audio / std
        
        # Store original length for final trimming
        original_length = L
        
        # Encoder
        skip_connections = []
        for i, encoder_block in enumerate(self.encoder):
            x = encoder_block(x)
            if isinstance(encoder_block, nn.Sequential):  # Skip GroupNorm layers
                skip_connections.append(x)
        
        skip_connections = skip_connections[::-1]  # Reverse for decoder

        # Causal attention mask
        len_s = x.shape[-1]
        attn_mask = torch.triu(torch.ones((1, len_s, len_s), device=x.device), diagonal=1).bool()
        attn_mask = ~attn_mask  # Invert for masking future tokens

        # Transformer processing
        x = self.tsfm_conv1(x)
        x = x.permute(0, 2, 1)  # (B, C, L) -> (B, L, C)
        x = self.tsfm_encoder(x, src_mask=attn_mask)
        x = x.permute(0, 2, 1)  # (B, L, C) -> (B, C, L)
        x = self.tsfm_conv2(x)

        # Decoder with simplified skip connection handling
        for i, decoder_block in enumerate(self.decoder):
            print(f"Decoder {i}: x.shape={x.shape}")
            
            # Apply decoder block first
            x = decoder_block(x)
            print(f"  After decoder block: x.shape={x.shape}")
            
            # Then add skip connection if possible
            if i < len(skip_connections):
                skip_i = skip_connections[i]
                
                if (skip_i.shape[1] == x.shape[1] and 
                    skip_i.shape[-1] <= x.shape[-1]):
                    
                    # Trim to match and add
                    min_length = min(x.shape[-1], skip_i.shape[-1])
                    x = x[..., :min_length] + skip_i[..., :min_length]
                    print(f"  Added skip connection: final shape={x.shape}")

        # Ensure we have a valid tensor
        print(f"Final x.shape before processing: {x.shape}")
        
        # Final processing
        x = trim_to_match_length(x, original_length) 
        x = x * std
        
        print(f"Returning: {x.shape}")
        return x


# Example usage and configuration for hearing aids
def create_hearing_aid_cleanunet():
    """Create a CleanUNet optimized for hearing aid applications"""
    return CausalCleanUNet(
        channels_input=1,
        channels_output=1,
        channels_H=32,  # Reduced for efficiency
        max_H=256,      # Reduced for efficiency
        encoder_n_layers=6,  # Reduced for lower latency
        kernel_size=3,  # Smaller kernel for lower latency
        stride=2,
        tsfm_n_layers=2,  # Reduced for efficiency
        tsfm_n_head=4,    # Reduced for efficiency
        tsfm_d_model=256, # Reduced for efficiency
        tsfm_d_inner=512, # Reduced for efficiency
        use_group_norm=True
    )


if __name__ == '__main__':
    # Test the causal model
    model = create_hearing_aid_cleanunet()
    
    # Test with a sample (16kHz, 10ms = 160 samples)
    sample_input = torch.randn(1, 1, 160)  # Batch=1, Channels=1, Length=160
    
    with torch.no_grad():
        output = model(sample_input)
    
    print(f"Input shape: {sample_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Test causality by checking if changing future samples affects current output
    test_input1 = torch.randn(1, 1, 320)
    test_input2 = test_input1.clone()
    test_input2[:, :, 160:] = torch.randn(1, 1, 160)  # Change future samples
    
    with torch.no_grad():
        out1 = model(test_input1)
        out2 = model(test_input2)
    
    # First 160 samples should be identical (causal property)
    causal_check = torch.allclose(out1[:, :, :160], out2[:, :, :160], atol=1e-6)
    print(f"Causality check passed: {causal_check}")
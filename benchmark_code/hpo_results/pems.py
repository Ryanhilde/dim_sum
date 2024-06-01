"""

"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

PeMS = {
    "Autoformer": {
        "n_steps": 24,
        "n_features": 862,
        "epochs": 100,
        "patience": 10,
        "n_layers": 3,
        "d_model": 64,
        "d_ffn": 1024,
        "n_heads": 8,
        "factor": 3,
        "moving_avg_window_size": 3,
        "dropout": 0,
        "lr": 0.00019140834144175633,
    },
    "BRITS": {
        "n_steps": 24,
        "n_features": 862,
        "patience": 10,
        "epochs": 100,
        "rnn_hidden_size": 1024,
        "lr": 0.00039999665755697616,
    },
    "Crossformer": {
        "n_steps": 24,
        "n_features": 862,
        "epochs": 100,
        "patience": 10,
        "n_layers": 2,
        "d_model": 512,
        "d_ffn": 1024,
        "n_heads": 2,
        "factor": 5,
        "seg_len": 24,
        "win_size": 2,
        "dropout": 0,
        "lr": 0.0007680302177215948,
    },
    "CSDI": {
        "n_steps": 24,
        "n_features": 862,
        "patience": 10,
        "epochs": 100,
        "n_layers": 2,
        "n_heads": 1,
        "n_channels": 64,
        "d_time_embedding": 64,
        "d_feature_embedding": 32,
        "d_diffusion_embedding": 64,
        "lr": 0.007575643577032553,
    },
    "DLinear": {
        "n_steps": 24,
        "n_features": 862,
        "epochs": 100,
        "patience": 10,
        "moving_avg_window_size": 5,
        "d_model": 1024,
        "lr": 0.0030354145968992624,
    },
    "ETSformer": {
        "n_steps": 24,
        "n_features": 862,
        "epochs": 100,
        "patience": 10,
        "n_e_layers": 3,
        "n_d_layers": 3,
        "d_model": 512,
        "d_ffn": 128,
        "n_heads": 1,
        "top_k": 3,
        "dropout": 0,
        "lr": 0.004394308089349178,
    },
    "FiLM": {
        "n_steps": 24,
        "n_features": 862,
        "epochs": 100,
        "patience": 10,
        "window_size": [2],
        "multiscale": [1, 2],
        "modes1": 128,
        "dropout": 0.2,
        "mode_type": 2,
        "d_model": 1024,
        "lr": 0.0021723684546149907,
    },
    "FreTS": {
        "n_steps": 24,
        "n_features": 862,
        "epochs": 100,
        "patience": 10,
        "embed_size": 256,
        "hidden_size": 128,
        "channel_independence": True,
        "lr": 0.0011145462257368792,
    },
    "GPVAE": {
        "n_steps": 24,
        "n_features": 862,
        "latent_size": 862,
        "patience": 10,
        "epochs": 100,
        "lr": 0.0007007078661052569,
        "beta": 0.2,
        "sigma": 1.005,
        "length_scale": 7,
        "encoder_sizes": [64, 64],
        "decoder_sizes": [128, 128],
        "window_size": 36,
    },
    "GRUD": {
        "n_steps": 24,
        "n_features": 862,
        "epochs": 100,
        "patience": 10,
        "rnn_hidden_size": 1024,
        "lr": 0.0009122289002286868,
    },
    "Informer": {
        "n_steps": 24,
        "n_features": 862,
        "epochs": 100,
        "patience": 10,
        "n_layers": 2,
        "d_model": 1024,
        "d_ffn": 512,
        "n_heads": 2,
        "factor": 5,
        "dropout": 0.1,
        "lr": 0.0005192256575896875,
    },
    "iTransformer": {
        "n_steps": 24,
        "n_features": 862,
        "epochs": 100,
        "patience": 10,
        "n_layers": 4,
        "d_model": 1024,
        "d_ffn": 512,
        "n_heads": 2,
        "d_k": 128,
        "d_v": 64,
        "dropout": 0,
        "attn_dropout": 0.4,
        "lr": 0.0015078206063479898,
    },
    "Koopa": {
        "n_steps": 24,
        "n_features": 862,
        "epochs": 100,
        "patience": 10,
        "n_seg_steps": 12,
        "d_dynamic": 64,
        "d_hidden": 512,
        "n_hidden_layers": 3,
        "n_blocks": 2,
        "lr": 0.003580034914232453,
    },
    "MICN": {
        "n_steps": 24,
        "n_features": 862,
        "epochs": 100,
        "patience": 10,
        "n_layers": 1,
        "d_model": 512,
        "conv_kernel": [4, 8],
        "dropout": 0,
        "lr": 0.00028841545104174083,
    },
    "MRNN": {
        "n_steps": 24,
        "n_features": 862,
        "patience": 10,
        "epochs": 100,
        "rnn_hidden_size": 128,
        "lr": 0.0011088355494532644,
    },
    "NonstationaryTransformer": {
        "n_steps": 24,
        "n_features": 862,
        "epochs": 100,
        "patience": 10,
        "n_layers": 2,
        "d_model": 64,
        "n_heads": 2,
        "d_ffn": 128,
        "n_projector_hidden_layers": 2,
        "d_projector_hidden": [32, 32],
        "dropout": 0.1,
        "lr": 0.005611524751291667,
    },
    "PatchTST": {
        "n_steps": 24,
        "n_features": 862,
        "epochs": 100,
        "patience": 10,
        "patch_len": 8,
        "stride": 4,
        "n_layers": 1,
        "d_model": 64,
        "d_ffn": 1024,
        "n_heads": 2,
        "d_k": 256,
        "d_v": 32,
        "dropout": 0,
        "attn_dropout": 0.2,
        "lr": 0.0006715379437301997,
    },
    "Pyraformer": {
        "n_steps": 24,
        "n_features": 862,
        "epochs": 100,
        "patience": 10,
        "n_layers": 1,
        "d_model": 512,
        "d_ffn": 512,
        "n_heads": 8,
        "window_size": [4, 4],
        "inner_size": 3,
        "dropout": 0,
        "attn_dropout": 0.4,
        "lr": 0.0012611477317167373,
    },
    "SAITS": {
        "n_steps": 24,
        "n_features": 862,
        "epochs": 100,
        "patience": 10,
        "n_layers": 3,
        "d_model": 64,
        "d_ffn": 128,
        "n_heads": 8,
        "d_k": 256,
        "d_v": 64,
        "dropout": 0,
        "attn_dropout": 0.3,
        "lr": 0.00010782665575297283,
    },
    "SCINet": {
        "n_steps": 24,
        "n_features": 862,
        "epochs": 100,
        "patience": 10,
        "n_stacks": 1,
        "n_levels": 1,
        "n_groups": 1,
        "n_decoder_layers": 1,
        "d_hidden": 64,
        "dropout": 0.3,
        "lr": 0.003395853146689833,
    },
    "StemGNN": {
        "n_steps": 24,
        "n_features": 862,
        "epochs": 100,
        "patience": 10,
        "n_layers": 1,
        "n_stacks": 2,
        "d_model": 512,
        "dropout": 0.5,
        "lr": 0.00229167728845157,
    },
    "TimesNet": {
        "n_steps": 24,
        "n_features": 862,
        "patience": 10,
        "epochs": 100,
        "n_layers": 1,
        "top_k": 3,
        "d_model": 1024,
        "d_ffn": 512,
        "n_kernels": 4,
        "dropout": 0,
        "lr": 0.0006214910395456774,
    },
    "Transformer": {
        "n_steps": 24,
        "n_features": 862,
        "epochs": 100,
        "patience": 10,
        "n_layers": 1,
        "d_model": 256,
        "d_ffn": 256,
        "n_heads": 8,
        "d_k": 256,
        "d_v": 256,
        "dropout": 0,
        "attn_dropout": 0.2,
        "lr": 0.00033857435295409376,
    },
    "USGAN": {
        "n_steps": 24,
        "n_features": 862,
        "patience": 10,
        "epochs": 100,
        "lr": 0.00038444134431723316,
        "rnn_hidden_size": 1024,
        "dropout": 0.2,
    },
}
"""

"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

BeijingAir ={'MICN': {'n_steps': 24, 'n_features': 132, 'epochs': 100, 'patience': 10, 'n_layers': 1, 'd_model': 1024, 'conv_kernel': [4, 8], 'dropout': 0.2, 'lr': 0.0006913468614473439},
 'MRNN': {'n_steps': 24, 'n_features': 132, 'patience': 10, 'epochs': 100, 'rnn_hidden_size': 64, 'lr': 0.009213752279771266},
 'FiLM': {'n_steps': 24, 'n_features': 132, 'epochs': 100, 'patience': 10, 'window_size': [2], 'multiscale': [1, 2], 'modes1': 64, 'dropout': 0.4, 'mode_type': 2, 'd_model': 1024, 'lr': 0.009202830676100575},
 'Pyraformer': {'n_steps': 24, 'n_features': 132, 'epochs': 100, 'patience': 10, 'n_layers': 2, 'd_model': 512, 'd_ffn': 256, 'n_heads': 2, 'window_size': [4, 4], 'inner_size': 3, 'dropout': 0.1, 'attn_dropout': 0.5, 'lr': 0.0014313550487184197},
 'SCINet': {'n_steps': 24, 'n_features': 132, 'epochs': 100, 'patience': 10, 'n_stacks': 1, 'n_levels': 1, 'n_groups': 1, 'n_decoder_layers': 1, 'd_hidden': 64, 'dropout': 0.1, 'lr': 0.001305955655419499},
 'NonstationaryTransformer': {'n_steps': 24, 'n_features': 132, 'epochs': 100, 'patience': 10, 'n_layers': 1, 'd_model': 1024, 'n_heads': 2, 'd_ffn': 1024, 'n_projector_hidden_layers': 2, 'd_projector_hidden': [256, 256], 'dropout': 0.3, 'lr': 0.0001533470275031822},
 'SAITS': {'n_steps': 24, 'n_features': 132, 'epochs': 100, 'patience': 10, 'n_layers': 1, 'd_model': 64, 'd_ffn': 256, 'n_heads': 4, 'd_k': 256, 'd_v': 64, 'dropout': 0.1, 'attn_dropout': 0.1, 'lr': 0.00125059883141597},
 'BRITS': {'n_steps': 24, 'n_features': 132, 'patience': 10, 'epochs': 100, 'rnn_hidden_size': 512, 'lr': 0.006242554068503864},
 'ETSformer': {'n_steps': 24, 'n_features': 132, 'epochs': 100, 'patience': 10, 'n_e_layers': 3, 'n_d_layers': 2, 'd_model': 1024, 'd_ffn': 64, 'n_heads': 2, 'top_k': 5, 'dropout': 0, 'lr': 0.00017972369587604037},
 'FreTS': {'n_steps': 24, 'n_features': 132, 'epochs': 100, 'patience': 10, 'embed_size': 128, 'hidden_size': 256, 'channel_independence': False, 'lr': 0.0013412816847317107},
 'USGAN': {'n_steps': 24, 'n_features': 132, 'patience': 10, 'epochs': 100, 'lr': 0.0005217674009036597, 'rnn_hidden_size': 512, 'dropout': 0.1},
 'Transformer': {'n_steps': 24, 'n_features': 132, 'epochs': 100, 'patience': 10, 'n_layers': 4, 'd_model': 128, 'd_ffn': 1024, 'n_heads': 8, 'd_k': 512, 'd_v': 128, 'dropout': 0, 'attn_dropout': 0.2, 'lr': 7.091472731808204e-05},
 'Autoformer': {'n_steps': 24, 'n_features': 132, 'epochs': 100, 'patience': 10, 'n_layers': 1, 'd_model': 1024, 'd_ffn': 1024, 'n_heads': 8, 'factor': 3, 'moving_avg_window_size': 5, 'dropout': 0.3, 'lr': 6.864468906534477e-05},
 'Crossformer': {'n_steps': 24, 'n_features': 132, 'epochs': 100, 'patience': 10, 'n_layers': 3, 'd_model': 1024, 'd_ffn': 512, 'n_heads': 8, 'factor': 3, 'seg_len': 6, 'win_size': 2, 'dropout': 0.1, 'lr': 0.00015265551449293065},
 'Informer': {'n_steps': 24, 'n_features': 132, 'epochs': 100, 'patience': 10, 'n_layers': 1, 'd_model': 1024, 'd_ffn': 1024, 'n_heads': 1, 'factor': 5, 'dropout': 0, 'lr': 0.0003920360344910405},
 'DLinear': {'n_steps': 24, 'n_features': 132, 'epochs': 100, 'patience': 10, 'moving_avg_window_size': 5, 'd_model': 256, 'lr': 0.007545619490239286},
 'GRUD': {'n_steps': 24, 'n_features': 132, 'epochs': 100, 'patience': 10, 'rnn_hidden_size': 1024, 'lr': 0.00024042971963822373},
 'StemGNN': {'n_steps': 24, 'n_features': 132, 'epochs': 100, 'patience': 10, 'n_layers': 3, 'n_stacks': 2, 'd_model': 512, 'dropout': 0, 'lr': 0.002346769078880226},
 'iTransformer': {'n_steps': 24, 'n_features': 132, 'epochs': 100, 'patience': 10, 'n_layers': 3, 'd_model': 256, 'd_ffn': 4096, 'n_heads': 8, 'd_k': 32, 'd_v': 128, 'dropout': 0, 'attn_dropout': 0, 'lr': 0.0004619505533241738},
 'Koopa': {'n_steps': 24, 'n_features': 132, 'epochs': 100, 'patience': 10, 'n_seg_steps': 12, 'd_dynamic': 128, 'd_hidden': 128, 'n_hidden_layers': 2, 'n_blocks': 3, 'lr': 0.0023963331309816895},
 'TimesNet': {'n_steps': 24, 'n_features': 132, 'patience': 10, 'epochs': 100, 'n_layers': 2, 'top_k': 1, 'd_model': 1024, 'd_ffn': 128, 'n_kernels': 5, 'dropout': 0.1, 'lr': 0.000737615359931056},
 'GPVAE': {'n_steps': 24, 'n_features': 132, 'latent_size': 7, 'patience': 10, 'epochs': 100, 'lr': 0.0011055049750454135, 'beta': 0.2, 'sigma': 1.005, 'length_scale': 7, 'encoder_sizes': [512, 512], 'decoder_sizes': [512, 512], 'window_size': 6},
 'PatchTST': {'n_steps': 24, 'n_features': 132, 'epochs': 100, 'patience': 10, 'patch_len': 24, 'stride': 4, 'n_layers': 2, 'd_model': 64, 'd_ffn': 512, 'n_heads': 8, 'd_k': 256, 'd_v': 128, 'dropout': 0, 'attn_dropout': 0.4, 'lr': 0.00017627593670808844},
 'CSDI': {'n_steps': 24, 'n_features': 132, 'patience': 10, 'epochs': 100, 'n_layers': 6, 'n_heads': 8, 'n_channels': 32, 'd_time_embedding': 256, 'd_feature_embedding': 16, 'd_diffusion_embedding': 32, 'lr': 0.0036662098229766093},}

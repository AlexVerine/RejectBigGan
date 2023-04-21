# #!/bin/bash
python launch_jeanzay.py \
--dataset I128_hdf5 --parallel --shuffle  --num_workers 8 --batch_size 256 --load_in_mem  \
--num_G_accumulations 8 --num_D_accumulations 8 \
--num_D_steps 1 --G_lr 1e-4 --D_lr 4e-4 --D_B2 0.999 --G_B2 0.999 \
--G_attn 64 --D_attn 64 \
--G_nl inplace_relu --D_nl inplace_relu \
--SN_eps 1e-6 --BN_eps 1e-5 --adam_eps 1e-6 \
--G_ortho 0.0 \
--G_shared \
--G_init ortho --D_init ortho \
--hier --dim_z 120 --shared_dim 128 \
--G_eval_mode \
--G_ch 96 --D_ch 96 \
--ema --use_ema --ema_start 20000 \
--test_every 2000 --save_every 5000 --num_best_copies 10 --num_save_copies 10 --seed 0 \
--use_multiepoch_sampler --num_epochs 200  --resume  --partition gpu_p5  --which_best R
#
python launch_jeanzay.py \
--dataset I128_hdf5 --which_loss PR --which_div Chi2  --lambda 0.5 --parallel --shuffle  --num_workers 8 --batch_size 256 --load_in_mem  \
--num_G_accumulations 8 --num_D_accumulations 8 \
--num_D_steps 1 --G_lr 1e-4 --D_lr 4e-4 --D_B2 0.999 --G_B2 0.999 \
--G_attn 64 --D_attn 64 \
--G_nl inplace_relu --D_nl inplace_relu \
--SN_eps 1e-6 --BN_eps 1e-5 --adam_eps 1e-6 \
--G_ortho 0.0 \
--G_shared \
--G_init ortho --D_init ortho \
--hier --dim_z 120 --shared_dim 128 \
--G_eval_mode \
--G_ch 96 --D_ch 96 \
--ema --use_ema --ema_start 20000 \
--test_every 2000 --save_every 5000 --num_best_copies 10 --num_save_copies 10 --seed 0 \
--use_multiepoch_sampler --num_epochs 200 --resume  --partition gpu_p5 --which_best R


python launch_jeanzay.py \
--dataset I128_hdf5 --which_loss PR --which_div Chi2  --lambda 1 --parallel --shuffle  --num_workers 8 --batch_size 256 --load_in_mem  \
--num_G_accumulations 8 --num_D_accumulations 8 \
--num_D_steps 1 --G_lr 1e-4 --D_lr 4e-4 --D_B2 0.999 --G_B2 0.999 \
--G_attn 64 --D_attn 64 \
--G_nl inplace_relu --D_nl inplace_relu \
--SN_eps 1e-6 --BN_eps 1e-5 --adam_eps 1e-6 \
--G_ortho 0.0 \
--G_shared \
--G_init ortho --D_init ortho \
--hier --dim_z 120 --shared_dim 128 \
--G_eval_mode \
--G_ch 96 --D_ch 96 \
--ema --use_ema --ema_start 20000 \
--test_every 2000 --save_every 5000 --num_best_copies 10 --num_save_copies 10 --seed 0 \
--use_multiepoch_sampler --num_epochs 200 --resume --partition gpu_p5 --which_best R


python launch_jeanzay.py \
--dataset I128_hdf5 --which_loss PR --which_div Chi2  --lambda 2 --parallel --shuffle  --num_workers 8 --batch_size 256 --load_in_mem  \
--num_G_accumulations 8 --num_D_accumulations 8 \
--num_D_steps 1 --G_lr 1e-4 --D_lr 4e-4 --D_B2 0.999 --G_B2 0.999 \
--G_attn 64 --D_attn 64 \
--G_nl inplace_relu --D_nl inplace_relu \
--SN_eps 1e-6 --BN_eps 1e-5 --adam_eps 1e-6 \
--G_ortho 0.0 \
--G_shared \
--G_init ortho --D_init ortho \
--hier --dim_z 120 --shared_dim 128 \
--G_eval_mode \
--G_ch 96 --D_ch 96 \
--ema --use_ema --ema_start 20000 \
--test_every 2000 --save_every 5000 --num_best_copies 10 --num_save_copies 10  --seed 0 \
--use_multiepoch_sampler --num_epochs 200 --resume  --partition gpu_p5 --which_best R


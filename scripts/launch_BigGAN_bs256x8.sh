# #!/bin/bash
# python launch_jeanzay.py \
# --dataset I128_hdf5 --parallel --shuffle  --num_workers 4 --batch_size 128  \
# --num_G_accumulations 16  --num_D_accumulations 16 \
# --num_D_steps 2 --G_lr 5e-5 --D_lr 2e-4 --D_B2 0.999 --G_B2 0.999 \
# --G_attn 64 --D_attn 64 \
# --G_nl inplace_relu --D_nl inplace_relu \
# --SN_eps 1e-6 --BN_eps 1e-5 --adam_eps 1e-6 \
# --G_ortho 0.0 \
# --G_shared \
# --G_init ortho --D_init ortho \
# --hier --dim_z 120 --shared_dim 128 \
# --G_eval_mode \
# --G_ch 96 --D_ch 96 \
# --ema --use_ema --ema_start 20000 \
# --test_every 500 --save_every 500 --num_best_copies 5 --num_save_copies 2 --seed 0 \
# --use_multiepoch_sampler --resume_no_optim --num_epochs 160 --mem_constraint 32  \
# --which_best R
#
# python launch_jeanzay.py \
# --dataset I128_hdf5 --parallel --shuffle  --num_workers 4 --batch_size 128  \
# --num_G_accumulations 16  --num_D_accumulations 16 \
# --num_D_steps 2 --G_lr 1e-5 --D_lr 1e-5 --D_B2 0.999 --G_B2 0.999 \
# --G_attn 64 --D_attn 64 \
# --G_nl inplace_relu --D_nl inplace_relu \
# --SN_eps 1e-6 --BN_eps 1e-5 --adam_eps 1e-6 \
# --G_ortho 0.0 \
# --G_shared \
# --G_init ortho --D_init ortho \
# --hier --dim_z 120 --shared_dim 128 \
# --G_eval_mode \
# --G_ch 96 --D_ch 96 \
# --ema --use_ema --ema_start 20000 \
# --test_every 500 --save_every 500 --num_best_copies 5 --num_save_copies 2 --seed 0 \
# --resume_no_optim --num_epochs 210   \
# --which_best R  --name_suffix full
#

python launch_jeanzay.py \
--dataset I128_hdf5 --parallel --shuffle  --num_workers 4 --batch_size 128  \
--num_G_accumulations 16  --num_D_accumulations 16 \
--num_D_steps 2 --G_lr 1e-5 --D_lr 1e-5 --D_B2 0.999 --G_B2 0.999 \
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
--test_every 500 --save_every 500 --num_best_copies 5 --num_save_copies 5 --seed 0 \
--resume_no_optim --num_epochs 245 --mem_constraint 32  \
--which_best R --which_loss PR --lambda 0.5 --name_suffix full
 

python launch_jeanzay.py \
--dataset I128_hdf5 --parallel --shuffle  --num_workers 4 --batch_size 128  \
--num_G_accumulations 16  --num_D_accumulations 16 \
--num_D_steps 2 --G_lr 1e-5 --D_lr 1e-5 --D_B2 0.999 --G_B2 0.999 \
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
--test_every 500 --save_every 500 --num_best_copies 5 --num_save_copies 5 --seed 0 \
--resume_no_optim --num_epochs 245 --mem_constraint 32  \
--which_best P --which_loss PR --lambda 1  --name_suffix full


python launch_jeanzay.py \
--dataset I128_hdf5 --parallel --shuffle  --num_workers 4 --batch_size 128  \
--num_G_accumulations 16  --num_D_accumulations 16 \
--num_D_steps 2 --G_lr 1e-5 --D_lr 1e-5 --D_B2 0.999 --G_B2 0.999 \
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
--test_every 500 --save_every 500 --num_best_copies 5 --num_save_copies 5 --seed 0 \
--resume_no_optim --num_epochs 245 --mem_constraint 32  \
--which_best P --which_loss PR --lambda 2 --name_suffix full


python launch_jeanzay.py \
--dataset I128_hdf5 --parallel --shuffle  --num_workers 4 --batch_size 128  \
--num_G_accumulations 16  --num_D_accumulations 16 \
--num_D_steps 2 --G_lr 1e-5 --D_lr 1e-5 --D_B2 0.999 --G_B2 0.999 \
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
--test_every 500 --save_every 500 --num_best_copies 5 --num_save_copies 5 --seed 0 \
--resume_no_optim --num_epochs 245 --mem_constraint 32  \
--which_best P --which_loss PR --lambda 5 --name_suffix full



python launch_jeanzay.py \
--dataset I128_hdf5 --parallel --shuffle  --num_workers 4 --batch_size 128  \
--num_G_accumulations 16  --num_D_accumulations 16 \
--num_D_steps 2 --G_lr 1e-5 --D_lr 1e-5 --D_B2 0.999 --G_B2 0.999 \
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
--test_every 500 --save_every 500 --num_best_copies 5 --num_save_copies 5 --seed 0 \
--resume_no_optim --num_epochs 245 --mem_constraint 32  \
--which_best R --which_loss PR --lambda 0.2 --name_suffix full

# python launch_jeanzay.py \
# --dataset I128_hdf5 --parallel --shuffle  --num_workers 4 --batch_size 128  \
# --num_G_accumulations 16  --num_D_accumulations 16 \
# --num_D_steps 2 --G_lr 1e-6 --D_lr 1e-6 --D_B2 0.999 --G_B2 0.999 \
# --G_attn 64 --D_attn 64 \
# --G_nl inplace_relu --D_nl inplace_relu \
# --SN_eps 1e-6 --BN_eps 1e-5 --adam_eps 1e-6 \
# --G_ortho 0.0 \
# --G_shared \
# --G_init ortho --D_init ortho \
# --hier --dim_z 120 --shared_dim 128 \
# --G_eval_mode \
# --G_ch 96 --D_ch 96 \
# --ema --use_ema --ema_start 20000 \
# --test_every 500 --save_every 500 --num_best_copies 5 --num_save_copies 2 --seed 0 \
# --use_multiepoch_sampler --resume_no_optim --num_epochs 154 --mem_constraint 32  \
# --which_best R
#
# python launch_jeanzay.py \
# --dataset I128_hdf5 --parallel --shuffle  --num_workers 4 --batch_size 128 --load_in_mem  \
# --num_G_accumulations 16  --num_D_accumulations 16 \
# --num_D_steps 1 --G_lr 1e-4 --D_lr 4e-4 --D_B2 0.999 --G_B2 0.999 \
# --G_attn 64 --D_attn 64 \
# --G_nl inplace_relu --D_nl inplace_relu \
# --SN_eps 1e-6 --BN_eps 1e-5 --adam_eps 1e-6 \
# --G_ortho 0.0 \
# --G_shared \
# --G_init ortho --D_init ortho \
# --hier --dim_z 120 --shared_dim 128 \
# --G_eval_mode \
# --G_ch 96 --D_ch 96 \
# --ema --use_ema --ema_start 20000 \
# --test_every 1000 --save_every 1000 --num_best_copies 5 --num_save_copies 2 --seed 0 \
# --use_multiepoch_sampler --resume --num_epochs 200  --mem_constraint 32 \
#
# python launch_jeanzay.py \
# --dataset I128_hdf5 --parallel --shuffle  --num_workers 4 --batch_size 128 --load_in_mem  \
# --num_G_accumulations 1 --num_D_accumulations 1 \
# --num_D_steps 1 --G_lr 1e-5  --D_lr 4e-5 --D_B2 0.999 --G_B2 0.999 \
# --G_attn 64 --D_attn 64 \
# --G_nl inplace_relu --D_nl inplace_relu \
# --SN_eps 1e-6 --BN_eps 1e-5 --adam_eps 1e-6 \
# --G_ortho 0.0 \
# --G_shared \
# --G_init ortho --D_init ortho \
# --hier --dim_z 120 --shared_dim 128 \
# --G_eval_mode \
# --G_ch 96 --D_ch 96 \
# --ema --use_ema --ema_start 20000 \
# --test_every 1000 --save_every 1000 --num_best_copies 5 --num_save_copies 2 --seed 0 \
# --use_multiepoch_sampler --resume  --num_epochs 200  --mem_constraint 32 \

# python launch_jeanzay.py \
# --dataset I128_hdf5 --which_loss PR --which_div Chi2  --lambda 0.5 --parallel --shuffle  --num_workers 8 --batch_size 256 --load_in_mem  \
# --num_G_accumulations 8 --num_D_accumulations 8 \
# --num_D_steps 1 --G_lr 1e-4 --D_lr 4e-4 --D_B2 0.999 --G_B2 0.999 \
# --G_attn 64 --D_attn 64 \
# --G_nl inplace_relu --D_nl inplace_relu \
# --SN_eps 1e-6 --BN_eps 1e-5 --adam_eps 1e-6 \
# --G_ortho 0.0 \
# --G_shared \
# --G_init ortho --D_init ortho \
# --hier --dim_z 120 --shared_dim 128 \
# --G_eval_mode \
# --G_ch 96 --D_ch 96 \
# --ema --use_ema --ema_start 20000 \
# --test_every 2000 --save_every 5000 --num_best_copies 10 --num_save_copies 10 --seed 0 \
# --use_multiepoch_sampler --num_epochs 200 --resume  --partition gpu_p5 --which_best R
#
#
# python launch_jeanzay.py \
# --dataset I128_hdf5 --which_loss PR --which_div Chi2  --lambda 1 --parallel --shuffle  --num_workers 8 --batch_size 256 --load_in_mem  \
# --num_G_accumulations 8 --num_D_accumulations 8 \
# --num_D_steps 1 --G_lr 1e-4 --D_lr 4e-4 --D_B2 0.999 --G_B2 0.999 \
# --G_attn 64 --D_attn 64 \
# --G_nl inplace_relu --D_nl inplace_relu \
# --SN_eps 1e-6 --BN_eps 1e-5 --adam_eps 1e-6 \
# --G_ortho 0.0 \
# --G_shared \
# --G_init ortho --D_init ortho \
# --hier --dim_z 120 --shared_dim 128 \
# --G_eval_mode \
# --G_ch 96 --D_ch 96 \
# --ema --use_ema --ema_start 20000 \
# --test_every 2000 --save_every 5000 --num_best_copies 10 --num_save_copies 10 --seed 0 \
# --use_multiepoch_sampler --num_epochs 200 --resume --partition gpu_p5 --which_best R


# python launch_jeanzay.py \
# --dataset I128_hdf5 --which_loss PR --which_div Chi2  --lambda 2 --parallel --shuffle  --num_workers 8 --batch_size 256 --load_in_mem  \
# --num_G_accumulations 8 --num_D_accumulations 8 \
# --num_D_steps 1 --G_lr 1e-4 --D_lr 4e-4 --D_B2 0.999 --G_B2 0.999 \
# --G_attn 64 --D_attn 64 \
# --G_nl inplace_relu --D_nl inplace_relu \
# --SN_eps 1e-6 --BN_eps 1e-5 --adam_eps 1e-6 \
# --G_ortho 0.0 \
# --G_shared \
# --G_init ortho --D_init ortho \
# --hier --dim_z 120 --shared_dim 128 \
# --G_eval_mode \
# --G_ch 96 --D_ch 96 \
# --ema --use_ema --ema_start 20000 \
# --test_every 2000 --save_every 5000 --num_best_copies 10 --num_save_copies 10  --seed 0 \
# --use_multiepoch_sampler --num_epochs 200 --resume  --partition gpu_p5 --which_best R
#

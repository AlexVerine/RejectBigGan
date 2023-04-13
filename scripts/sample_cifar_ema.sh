#!/bin/bash
CUDA_VISIBLE_DEVICES=0,1 python sample.py \
--which_loss PR --which_div Chi2  --lambda 0.1 \
--shuffle --batch_size 128 --parallel --which_best R \
--num_G_accumulations 1 --num_D_accumulations 1 --num_epochs 500 \
--num_D_steps 4 --G_lr 2e-4 --D_lr 2e-4 \
--dataset C10 \
--G_ortho 0.0 \
--G_attn 0 --D_attn 0 \
--G_init N02 --D_init N02 \
--ema  --ema_start 1000 --G_eval_mode --sample_random \
--test_every 5000 --save_every 2000 --num_best_copies 5 --num_save_copies 2 --seed 0 \
--sample_inception_metrics --load_weights best0 \


# python sample.py \
# --which_loss PR --which_div Chi2  --lambda 0.2 \
# --shuffle --batch_size 128 --parallel --which_best R \
# --num_G_accumulations 1 --num_D_accumulations 1 --num_epochs 500 \
# --num_D_steps 4 --G_lr 2e-4 --D_lr 2e-4 \
# --dataset C10 \
# --G_ortho 0.0 \
# --G_attn 0 --D_attn 0 \
# --G_init N02 --D_init N02 \
# --ema  --ema_start 1000 --G_eval_mode --sample_random \
# --test_every 5000 --save_every 2000 --num_best_copies 5 --num_save_copies 2 --seed 0 \
# --sample_inception_metrics --load_weights best0 

# python sample.py \
# --which_loss PR --which_div Chi2  --lambda 0.5 \
# --shuffle --batch_size 128 --parallel --which_best R \
# --num_G_accumulations 1 --num_D_accumulations 1 --num_epochs 500 \
# --num_D_steps 4 --G_lr 2e-4 --D_lr 2e-4 \
# --dataset C10 \
# --G_ortho 0.0 \
# --G_attn 0 --D_attn 0 \
# --G_init N02 --D_init N02 \
# --ema  --ema_start 1000 --G_eval_mode --sample_random \
# --test_every 5000 --save_every 2000 --num_best_copies 5 --num_save_copies 2 --seed 0 \
# --sample_inception_metrics --load_weights best0 


# python sample.py \
# --which_loss PR --which_div Chi2  --lambda 1 \
# --shuffle --batch_size 128 --parallel --which_best R \
# --num_G_accumulations 1 --num_D_accumulations 1 --num_epochs 500 \
# --num_D_steps 4 --G_lr 2e-4 --D_lr 2e-4 \
# --dataset C10 \
# --G_ortho 0.0 \
# --G_attn 0 --D_attn 0 \
# --G_init N02 --D_init N02 \
# --ema  --ema_start 1000 --G_eval_mode --sample_random \
# --test_every 5000 --save_every 2000 --num_best_copies 5 --num_save_copies 2 --seed 0 \
# --sample_inception_metrics --load_weights best0 

# python sample.py \
# --which_loss PR --which_div Chi2  --lambda 2 \
# --shuffle --batch_size 128 --parallel --which_best R \
# --num_G_accumulations 1 --num_D_accumulations 1 --num_epochs 500 \
# --num_D_steps 4 --G_lr 2e-4 --D_lr 2e-4 \
# --dataset C10 \
# --G_ortho 0.0 \
# --G_attn 0 --D_attn 0 \
# --G_init N02 --D_init N02 \
# --ema  --ema_start 1000 --G_eval_mode --sample_random \
# --test_every 5000 --save_every 2000 --num_best_copies 5 --num_save_copies 2 --seed 0 \
# --sample_inception_metrics --load_weights best0 

# python sample.py \
# --which_loss PR --which_div Chi2  --lambda 5 \
# --shuffle --batch_size 128 --parallel --which_best R \
# --num_G_accumulations 1 --num_D_accumulations 1 --num_epochs 500 \
# --num_D_steps 4 --G_lr 2e-4 --D_lr 2e-4 \
# --dataset C10 \
# --G_ortho 0.0 \
# --G_attn 0 --D_attn 0 \
# --G_init N02 --D_init N02 \
# --ema  --ema_start 1000 --G_eval_mode --sample_random \
# --test_every 5000 --save_every 2000 --num_best_copies 5 --num_save_copies 2 --seed 0 \
# --sample_inception_metrics --load_weights best0 

# python sample.py \
# --which_loss PR --which_div Chi2  --lambda 10 \
# --shuffle --batch_size 128 --parallel --which_best R \
# --num_G_accumulations 1 --num_D_accumulations 1 --num_epochs 500 \
# --num_D_steps 4 --G_lr 2e-4 --D_lr 2e-4 \
# --dataset C10 \
# --G_ortho 0.0 \
# --G_attn 0 --D_attn 0 \
# --G_init N02 --D_init N02 \
# --ema  --ema_start 1000 --G_eval_mode --sample_random \
# --test_every 5000 --save_every 2000 --num_best_copies 5 --num_save_copies 2 --seed 0 \
# --sample_inception_metrics --load_weights best0 
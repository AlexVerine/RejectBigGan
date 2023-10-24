# python launch_jeanzay.py --mode train \
# --which_loss reject --which_div gan \ # --dataset CA64_hdf5 --parallel --shuffle  --num_workers 8 --batch_size 128  \
# --num_G_accumulations 1 --num_D_accumulations 1 \
# --num_D_steps 1 --G_lr 1e-5 --D_lr 4e-4 --D_B2 0.999 --G_B2 0.999 \
# --G_attn 32 --D_attn 32 \
# --G_nl relu --D_nl relu \
# --SN_eps 1e-8 --BN_eps 1e-5 --adam_eps 1e-8 \
# --G_ortho 0.0 \
# --G_init xavier --D_init xavier \
# --G_eval_mode \
# --G_ch 32 --D_ch 32 \
# --ema --use_ema --ema_start 2000 \
# --test_every 10000 --save_every 15000 --num_best_copies 5 --num_save_copies 2 --seed 0   --num_epochs 500
# #
# python launch_jeanzay.py --mode train \
# --which_loss reject --which_div gan \
# --dataset CA64_hdf5 --parallel --shuffle  --num_workers 8 --batch_size 128  \
# --num_G_accumulations 1 --num_D_accumulations 1 \
# --num_D_steps 1 --G_lr 5e-5 --D_lr 4e-4 --D_B2 0.999 --G_B2 0.999 \
# --G_attn 32 --D_attn 32 \
# --G_nl relu --D_nl relu \
# --SN_eps 1e-8 --BN_eps 1e-5 --adam_eps 1e-8 \
# --G_ortho 0.0 \
# --G_init xavier --D_init xavier \
# --G_eval_mode \
# --G_ch 32 --D_ch 32 \
# --ema --use_ema --ema_start 2000 \
# --test_every 10000 --save_every 15000 --num_best_copies 5 --num_save_copies 2 --seed 0   --num_epochs 500

# python launch_jeanzay.py --mode train --TOBRS --update_every 500 \
# --which_loss reject --which_div gan \
# --dataset CA64_hdf5 --parallel --shuffle  --num_workers 8 --batch_size 128  \
# --num_G_accumulations 1 --num_D_accumulations 1 \
# --num_D_steps 1 --G_lr 1e-4 --D_lr 4e-4 --D_B2 0.999 --G_B2 0.999 \
# --G_attn 32 --D_attn 32 \
# --G_nl relu --D_nl relu \
# --SN_eps 1e-8 --BN_eps 1e-5 --adam_eps 1e-8 \
# --G_ortho 0.0 \
# --G_init xavier --D_init xavier \
# --G_eval_mode \
# --G_ch 32 --D_ch 32 \
# --ema --use_ema --ema_start 2000 \
# --test_every 10000 --save_every 15000 --num_best_copies 5 --num_save_copies 2 --seed 1   --num_epochs 1000 
#
# python launch_jeanzay.py --mode train --TOBRS  --update_every 1000 \
# --which_loss reject --which_div gan \
# --dataset CA64_hdf5 --parallel --shuffle  --num_workers 8 --batch_size 128  \
# --num_G_accumulations 1 --num_D_accumulations 1 \
# --num_D_steps 1 --G_lr 1e-4 --D_lr 4e-4 --D_B2 0.999 --G_B2 0.999 \
# --G_attn 32 --D_attn 32 \
# --G_nl relu --D_nl relu \
# --SN_eps 1e-8 --BN_eps 1e-5 --adam_eps 1e-8 \
# --G_ortho 0.0 \
# --G_init xavier --D_init xavier \
# --G_eval_mode \
# --G_ch 32 --D_ch 32 \
# --ema --use_ema --ema_start 2000 \
# --test_every 10000 --save_every 15000 --num_best_copies 5 --num_save_copies 2 --seed 1   --num_epochs 1000 

# python launch_jeanzay.py --mode train \
# --which_loss reject --which_div gan --TOBRS \
# --dataset CA64_hdf5 --parallel --shuffle  --num_workers 8 --batch_size 128  \
# --num_G_accumulations 1 --num_D_accumulations 1 \
# --num_D_steps 1 --G_lr 1e-4 --D_lr 1e-4 --D_B2 0.999 --G_B2 0.999 \
# --G_attn 32 --D_attn 32 \
# --G_nl relu --D_nl relu \
# --SN_eps 1e-8 --BN_eps 1e-5 --adam_eps 1e-8 \
# --G_ortho 0.0 \
# --G_init xavier --D_init xavier \
# --G_eval_mode \
# --G_ch 32 --D_ch 32 \
# --ema --use_ema --ema_start 2000 \
# --test_every 10000 --save_every 15000 --num_best_copies 5 --num_save_copies 2 --seed 0   --num_epochs 1000 
#
# python launch_jeanzay.py --mode train \
# --which_loss reject --which_div gan  \
# --dataset CA64_hdf5 --parallel --shuffle  --num_workers 8 --batch_size 128  \
# --num_G_accumulations 1 --num_D_accumulations 1 \
# --num_D_steps 1 --G_lr 1e-4 --D_lr 4e-4 --D_B2 0.999 --G_B2 0.999 \
# --G_attn 32 --D_attn 32 \
# --G_nl relu --D_nl relu \
# --SN_eps 1e-8 --BN_eps 1e-5 --adam_eps 1e-8 \
# --G_ortho 0.0 \
# --G_init xavier --D_init xavier \
# --G_eval_mode \
# --G_ch 32 --D_ch 32 \
# --ema --use_ema --ema_start 2000 \
# --test_every 5000 --save_every 5000 --num_best_copies 5 --num_save_copies 2 --seed 0   --num_epochs 1500  --name_suffix finetune

# python launch_jeanzay.py --mode train \
# --which_loss reject --which_div gan --TOBRS \
# --dataset CA64_hdf5 --parallel --shuffle  --num_workers 8 --batch_size 128  \
# --num_G_accumulations 1 --num_D_accumulations 1 \
# --num_D_steps 1 --G_lr 1e-4 --D_lr 4e-4 --D_B2 0.999 --G_B2 0.999 \
# --G_attn 32 --D_attn 32 \
# --G_nl relu --D_nl relu \
# --SN_eps 1e-8 --BN_eps 1e-5 --adam_eps 1e-8 \
# --G_ortho 0.0 \
# --G_init xavier --D_init xavier \
# --G_eval_mode \
# --G_ch 32 --D_ch 32 \
# --ema --use_ema --ema_start 2000 \
# --test_every 5000 --save_every 5000 --num_best_copies 5 --num_save_copies 2 --seed 0   --num_epochs 200 --resume --name_suffix finetune_real

# python launch_jeanzay.py --mode train \
# --which_loss reject --which_div gan  \
# --dataset CA64_hdf5 --parallel --shuffle  --num_workers 8 --batch_size 128  \
# --num_G_accumulations 1 --num_D_accumulations 1 \
# --num_D_steps 1 --G_lr 1e-4 --D_lr 4e-4 --D_B2 0.999 --G_B2 0.999 \
# --G_attn 32 --D_attn 32 \
# --G_nl relu --D_nl relu \
# --SN_eps 1e-8 --BN_eps 1e-5 --adam_eps 1e-8 \
# --G_ortho 0.0 \
# --G_init xavier --D_init xavier \
# --G_eval_mode \
# --G_ch 32 --D_ch 32 \
# --ema --use_ema --ema_start 2000 \
# --test_every 5000 --save_every 5000 --num_best_copies 5 --num_save_copies 2 --seed 0   --num_epochs 200 --resume --name_suffix finetune_real
#
# python launch_jeanzay.py --mode train \
# --which_loss reject --which_div gan  \
# --dataset CA64_hdf5 --parallel --shuffle  --num_workers 8 --batch_size 128  \
# --num_G_accumulations 1 --num_D_accumulations 1 \
# --num_D_steps 1 --G_lr 1e-6 --D_lr 4e-6 --D_B2 0.999 --G_B2 0.999 \
# --G_attn 32 --D_attn 32 \
# --G_nl relu --D_nl relu \
# --SN_eps 1e-8 --BN_eps 1e-5 --adam_eps 1e-8 \
# --G_ortho 0.0 \
# --G_init xavier --D_init xavier \
# --G_eval_mode \
# --G_ch 32 --D_ch 32 \
# --ema --use_ema --ema_start 2000 \
# --test_every 5000 --save_every 5000 --num_best_copies 5 --num_save_copies 2 --seed 0   --num_epochs 200 --resume --name_suffix finetune_real

python3 launch_jeanzay.py --mode train \
--which_loss reject --which_div gan \
--dataset CA64_hdf5 --parallel --shuffle  --num_workers 8 --batch_size 128  \
--num_G_accumulations 1 --num_D_accumulations 1 \
--num_D_steps 1 --G_lr 1e-50 --D_lr 4e-5 --D_B2 0.999 --G_B2 0.999 \
--G_attn 32 --D_attn 32 \
--G_nl relu --D_nl relu \
--SN_eps 1e-8 --BN_eps 1e-5 --adam_eps 1e-8 \
--G_ortho 0.0 \
--G_init xavier --D_init xavier \
--G_eval_mode \
--G_ch 32 --D_ch 32 \
--ema --use_ema --ema_start 2000 \
--test_every 2000 --save_every 5000 --num_best_copies 5 --num_save_copies 5 --seed 0   --num_epochs 150 --resume_no_optim --name_suffix Donly 

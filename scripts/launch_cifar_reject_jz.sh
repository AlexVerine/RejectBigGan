python launch_jeanzay.py --mode train \
--which_loss reject --which_div gan  --update_every 100 \
--shuffle --batch_size 50 --parallel \
--num_G_accumulations 1 --num_D_accumulations 1 --num_epochs 800   \
--num_D_steps 4 --G_lr 2e-4 --D_lr 2e-5 \
--dataset C10_hdf5 \
--G_ortho 0.0 \
--G_attn 0 --D_attn 0 \
--G_init N02 --D_init N02 \
--ema --use_ema --ema_start 1000 \
--test_every 10000 --save_every 10000 --num_best_copies 5 --num_save_copies 2 --seed 0 \
 --load_in_mem --num_workers 4  --G_eval_mode 

python launch_jeanzay.py --mode train \
--which_loss reject --which_div gan  --update_every 10 \
--shuffle --batch_size 50 --parallel \
--num_G_accumulations 1 --num_D_accumulations 1 --num_epochs 800   \
--num_D_steps 4 --G_lr 2e-4 --D_lr 2e-5 \
--dataset C10_hdf5 \
--G_ortho 0.0 \
--G_attn 0 --D_attn 0 \
--G_init N02 --D_init N02 \
--ema --use_ema --ema_start 1000 \
--test_every 10000 --save_every 10000 --num_best_copies 5 --num_save_copies 2 --seed 0 \
 --load_in_mem --num_workers 4  --G_eval_mode 


python launch_jeanzay.py --mode train \
--which_loss reject --which_div gan  --update_every 1 \
--shuffle --batch_size 50 --parallel \
--num_G_accumulations 1 --num_D_accumulations 1 --num_epochs 800   \
--num_D_steps 4 --G_lr 2e-4 --D_lr 2e-5 \
--dataset C10_hdf5 \
--G_ortho 0.0 \
--G_attn 0 --D_attn 0 \
--G_init N02 --D_init N02 \
--ema --use_ema --ema_start 1000 \
--test_every 10000 --save_every 10000 --num_best_copies 5 --num_save_copies 2 --seed 0 \
 --load_in_mem --num_workers 4  --G_eval_mode 

# python launch_jeanzay.py --mode train \
# --which_loss reject --which_div gan  --update_every 1000 --TOBRS \
# --shuffle --batch_size 50 --parallel \
# --num_G_accumulations 1 --num_D_accumulations 1 --num_epochs 800   \
# --num_D_steps 4 --G_lr 2e-4 --D_lr 2e-5 \
# --dataset C10_hdf5 \
# --G_ortho 0.0 \
# --G_attn 0 --D_attn 0 \
# --G_init N02 --D_init N02 \
# --ema --use_ema --ema_start 1000 \
# --test_every 10000 --save_every 10000 --num_best_copies 5 --num_save_copies 2 --seed 0 \
#  --load_in_mem --num_workers 4  --G_eval_mode 
#
# python launch_jeanzay.py --mode train \
# --which_loss reject --which_div gan  --update_every 100 --TOBRS \
# --shuffle --batch_size 50 --parallel \
# --num_G_accumulations 1 --num_D_accumulations 1 --num_epochs 800   \
# --num_D_steps 4 --G_lr 2e-5 --D_lr 2e-5 \
# --dataset C10_hdf5 \
# --G_ortho 0.0 \
# --G_attn 0 --D_attn 0 \
# --G_init N02 --D_init N02 \
# --ema --use_ema --ema_start 1000 \
# --test_every 10000 --save_every 10000 --num_best_copies 5 --num_save_copies 2 --seed 0 \
#  --load_in_mem --num_workers 4  --G_eval_mode 
#
# python launch_jeanzay.py --mode train \
# --which_loss reject --which_div hinge  --update_every 1000 \
# --shuffle --batch_size 50 --parallel \
# --num_G_accumulations 1 --num_D_accumulations 1 --num_epochs 800   \
# --num_D_steps 4 --G_lr 2e-5 --D_lr 2e-5 \
# --dataset C10_hdf5 \
# --G_ortho 0.0 \
# --G_attn 0 --D_attn 0 \
# --G_init N02 --D_init N02 \
# --ema --use_ema --ema_start 1000 \
# --test_every 10000 --save_every 10000 --num_best_copies 5 --num_save_copies 2 --seed 0 \
#  --load_in_mem --num_workers 4  --G_eval_mode 
#
# python launch_jeanzay.py --mode train \
# --which_loss reject --which_div hinge  --update_every 1000 --TOBRS \
# --shuffle --batch_size 50 --parallel \
# --num_G_accumulations 1 --num_D_accumulations 1 --num_epochs 800   \
# --num_D_steps 4 --G_lr 2e-5 --D_lr 2e-5 \
# --dataset C10_hdf5 \
# --G_ortho 0.0 \
# --G_attn 0 --D_attn 0 \
# --G_init N02 --D_init N02 \
# --ema --use_ema --ema_start 1000 \
# --test_every 10000 --save_every 10000 --num_best_copies 5 --num_save_copies 2 --seed 0 \
#  --load_in_mem --num_workers 4  --G_eval_mode 


# python launch_jeanzay.py --mode train \
# --which_loss reject --which_div gan  --update_every 1000 \
# --shuffle --batch_size 50 --parallel \
# --num_G_accumulations 1 --num_D_accumulations 1 --num_epochs 800   \
# --num_D_steps 4 --G_lr 2e-6 --D_lr 2e-6 \
# --dataset C10_hdf5 \
# --G_ortho 0.0 \
# --G_attn 0 --D_attn 0 \
# --G_init N02 --D_init N02 \
# --ema --use_ema --ema_start 1000 \
# --test_every 1000 --save_every 10000 --num_best_copies 5 --num_save_copies 2 --seed 0 \
#  --load_in_mem --num_workers 4  --G_eval_mode --resume_no_optim --name_suffix finetune

# python launch_jeanzay.py --mode train  --TOBRS \
# --which_loss reject --which_div gan --update_every 1000 \
# --shuffle --batch_size 50 --parallel \
# --num_G_accumulations 1 --num_D_accumulations 1 --num_epochs 800   \
# --num_D_steps 4 --G_lr 2e-6 --D_lr 2e-6 \
# --dataset C10_hdf5 \
# --G_ortho 0.0 \
# --G_attn 0 --D_attn 0 \
# --G_init N02 --D_init N02 \
# --ema --use_ema --ema_start 1000 \
# --test_every 5000 --save_every 10000 --num_best_copies 5 --num_save_copies 2 --seed 0 \
#  --load_in_mem --num_workers 4  --G_eval_mode --resume_no_optim --name_suffix finetune
#
# python launch_jeanzay.py --mode train  --TOBRS \
# --which_loss reject --which_div gan --update_every 10 \
# --shuffle --batch_size 50 --parallel \
# --num_G_accumulations 1 --num_D_accumulations 1 --num_epochs 800   \
# --num_D_steps 4 --G_lr 2e-6 --D_lr 2e-6 \
# --dataset C10_hdf5 \
# --G_ortho 0.0 \
# --G_attn 0 --D_attn 0 \
# --G_init N02 --D_init N02 \
# --ema --use_ema --ema_start 1000 \
# --test_every 5000 --save_every 10000 --num_best_copies 5 --num_save_copies 2 --seed 0 \
#  --load_in_mem --num_workers 4  --G_eval_mode --resume_no_optim --name_suffix finetune



# python launch_jeanzay.py --mode train \
# --which_loss reject --which_div gan \
# --shuffle --batch_size 50 --parallel \
# --num_G_accumulations 1 --num_D_accumulations 1 --num_epochs 505   \
# --num_D_steps 4 --G_lr 2e-10 --D_lr 2e-5 \
# --dataset C10_hdf5 \
# --G_ortho 0.0 \
# --G_attn 0 --D_attn 0 \
# --G_init N02 --D_init N02 \
# --ema --use_ema --ema_start 1000 \
# --test_every 100000 --save_every 5000 --num_best_copies 5 --num_save_copies 2 --seed 0 \
# --load_in_mem --num_workers 4  --G_eval_mode --name_suffix Donly --resume_no_optim

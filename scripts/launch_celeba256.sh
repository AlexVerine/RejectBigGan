#
# python launch_jeanzay.py \
# --which_loss PR --which_div Chi2 --lambda 1.0  \
# --shuffle --batch_size 128 --parallel --which_best P+R \
# --num_G_accumulations 1 --num_D_accumulations 1 --num_epochs 250 \
# --num_D_steps 4 --G_lr 5e-5 --D_lr 5e-5 \
# --dataset CA256_hdf5 --load_in_mem --num_workers 8 \
# --G_ortho 0.0 \
# --G_attn 0 --D_attn 0 \
# --G_init N02 --D_init N02 \
# --ema  --ema_start 5000 --G_eval_mode \
# --test_every 2000 --save_every 2000 --num_best_copies 2 --num_save_copies 5 --seed 0 --partition gpu_p5
#
# python launch_jeanzay.py \
# --which_loss PR --which_div Chi2 --lambda 5.0  \
# --shuffle --batch_size 128 --parallel --which_best P \
# --num_G_accumulations 1 --num_D_accumulations 1 --num_epochs 250 \
# --num_D_steps 4 --G_lr 5e-5 --D_lr 5e-5 \
# --dataset CA256_hdf5 --load_in_mem --num_workers 8 \
# --G_ortho 0.0 \
# --G_attn 0 --D_attn 0 \
# --G_init N02 --D_init N02 \
# --ema  --ema_start 5000 --G_eval_mode \
# --test_every 2000 --save_every 2000 --num_best_copies 2 --num_save_copies 5 --seed 0 --partition gpu_p5
#
# python launch_jeanzay.py \
# --which_loss PR --which_div Chi2 --lambda 0.2  \
# --shuffle --batch_size 128 --parallel --which_best R \
# --num_G_accumulations 1 --num_D_accumulations 1 --num_epochs 250 \
# --num_D_steps 4 --G_lr 5e-5 --D_lr 5e-5 \
# --dataset CA256_hdf5 --load_in_mem --num_workers 8 \
# --G_ortho 0.0 \
# --G_attn 0 --D_attn 0 \
# --G_init N02 --D_init N02 \
# --ema  --ema_start 5000 --G_eval_mode \
# --test_every 2000 --save_every 2000 --num_best_copies 2 --num_save_copies 5 --seed 0 --partition gpu_p5
#
# python launch_jeanzay.py \
# --which_loss PR --which_div Chi2 --lambda 1.0  \
# --shuffle --batch_size 128 --parallel --which_best P+R \
# --num_G_accumulations 2 --num_D_accumulations 2 --num_epochs 400 \
# --num_D_steps 4 --G_lr 1e-5 --D_lr 1e-5 \
# --dataset CA256_hdf5 --load_in_mem --num_workers 8 \
# --G_ortho 0.0 \
# --G_attn 0 --D_attn 0 \
# --G_init N02 --D_init N02 \
# --ema  --ema_start 5000 --G_eval_mode \
# --test_every 2000 --save_every 2000 --num_best_copies 2 --num_save_copies 5 --seed 0  --partition gpu_p5 --resume_no_optim --name_suffix finetune_noopt
#
# python launch_jeanzay.py  \
# --which_loss PR --which_div Chi2 --lambda 2.0  \
# --shuffle --batch_size 128 --parallel --which_best P \
# --num_G_accumulations 2 --num_D_accumulations 2 --num_epochs 400 \
# --num_D_steps 4 --G_lr 1e-5 --D_lr 1e-5 \
# --dataset CA256_hdf5 --load_in_mem --num_workers 8 \
# --G_ortho 0.0 \
# --G_attn 0 --D_attn 0 \
# --G_init N02 --D_init N02 \
# --ema  --ema_start 5000 --G_eval_mode \
# --test_every 2000 --save_every 2000 --num_best_copies 2 --num_save_copies 5 --seed 0  --partition gpu_p5 --resume_no_optim --name_suffix finetune_noopt
#
#
# python launch_jeanzay.py  \
# --which_loss PR --which_div Chi2 --lambda 0.5  \
# --shuffle --batch_size 128 --parallel --which_best R \
# --num_G_accumulations 2 --num_D_accumulations 2 --num_epochs 400 \
# --num_D_steps 4 --G_lr 1e-5 --D_lr 1e-5 \
# --dataset CA256_hdf5 --load_in_mem --num_workers 8 \
# --G_ortho 0.0 \
# --G_attn 0 --D_attn 0 \
# --G_init N02 --D_init N02 \
# --ema  --ema_start 5000 --G_eval_mode \
# --test_every 2000 --save_every 2000 --num_best_copies 2 --num_save_copies 5 --seed 0  --partition gpu_p5 --resume_no_optim --name_suffix finetune_noopt
#
#
# python launch_jeanzay.py \
# --which_loss PR --which_div Chi2 --lambda 1.0  \
# --shuffle --batch_size 128 --parallel --which_best P+R \
# --num_G_accumulations 2 --num_D_accumulations 2 --num_epochs 400 \
# --num_D_steps 4 --G_lr 1e-6 --D_lr 1e-6 \
# --dataset CA256_hdf5 --load_in_mem --num_workers 8 \
# --G_ortho 0.0 \
# --G_attn 0 --D_attn 0 \
# --G_init N02 --D_init N02 \
# --ema  --ema_start 5000 --G_eval_mode \
# --test_every 2000 --save_every 2000 --num_best_copies 2 --num_save_copies 5 --seed 0  --partition gpu_p5 --resume_no_optim --name_suffix finetune_noopt
#
python launch_jeanzay.py  \
--which_loss PR --which_div Chi2 --lambda 5.0  \
--shuffle --batch_size 128 --parallel --which_best P \
--num_G_accumulations 2 --num_D_accumulations 2 --num_epochs 400 \
--num_D_steps 4 --G_lr 1e-6 --D_lr 1e-6 \
--dataset CA256_hdf5 --load_in_mem --num_workers 8 \
--G_ortho 0.0 \
--G_attn 0 --D_attn 0 \
--G_init N02 --D_init N02 \
--ema  --ema_start 5000 --G_eval_mode \
--test_every 2000 --save_every 2000 --num_best_copies 2 --num_save_copies 5 --seed 0  --partition gpu_p5 --resume_no_optim --name_suffix finetune_noopt


python launch_jeanzay.py   \
--which_loss PR --which_div Chi2 --lambda 0.2  \
--shuffle --batch_size 128 --parallel --which_best R \
--num_G_accumulations 2 --num_D_accumulations 2 --num_epochs 400 \
--num_D_steps 4 --G_lr 1e-6 --D_lr 1e-6 \
--dataset CA256_hdf5 --load_in_mem --num_workers 8 \
--G_ortho 0.0 \
--G_attn 0 --D_attn 0 \
--G_init N02 --D_init N02 \
--ema  --ema_start 5000 --G_eval_mode \
--test_every 2000 --save_every 2000 --num_best_copies 2 --num_save_copies 5 --seed 0  --partition gpu_p5 --resume_no_optim --name_suffix finetune_noopt

# python launch_jeanzay.py \
# --which_loss PR --which_div Chi2 --lambda 1.0  \
# --shuffle --batch_size 128 --parallel --which_best P+R \
# --num_G_accumulations 2 --num_D_accumulations 2 --num_epochs 400 \
# --num_D_steps 4 --G_lr 1e-5 --D_lr 1e-5 \
# --dataset CA256_hdf5 --load_in_mem --num_workers 4 \
# --G_ortho 0.0 \
# --G_attn 0 --D_attn 0 \
# --G_init N02 --D_init N02 \
# --ema  --ema_start 5000 --G_eval_mode \
# --test_every 2000 --save_every 2000 --num_best_copies 2 --num_save_copies 5 --seed 0 --mem_constraint 32  --resume
#
# python launch_jeanzay.py \
# --which_loss PR --which_div Chi2 --lambda 2.0  \
# --shuffle --batch_size 128 --parallel --which_best P \
# --num_G_accumulations 2 --num_D_accumulations 2 --num_epochs 400 \
# --num_D_steps 4 --G_lr 1e-5 --D_lr 1e-5 \
# --dataset CA256_hdf5 --load_in_mem --num_workers 4 \
# --G_ortho 0.0 \
# --G_attn 0 --D_attn 0 \
# --G_init N02 --D_init N02 \
# --ema  --ema_start 5000 --G_eval_mode \
# --test_every 2000 --save_every 2000 --num_best_copies 2 --num_save_copies 5 --seed 0 --mem_constraint 32 --resume
#
#
#
# python launch_jeanzay.py \
# --which_loss PR --which_div Chi2 --lambda 0.5  \
# --shuffle --batch_size 128 --parallel --which_best R \
# --num_G_accumulations 2 --num_D_accumulations 2 --num_epochs 400 \
# --num_D_steps 4 --G_lr 1e-5 --D_lr 1e-5 \
# --dataset CA256_hdf5 --load_in_mem --num_workers 4 \
# --G_ortho 0.0 \
# --G_attn 0 --D_attn 0 \
# --G_init N02 --D_init N02 \
# --ema  --ema_start 5000 --G_eval_mode \
# --test_every 2000 --save_every 2000 --num_best_copies 2 --num_save_copies 5 --seed 0 --mem_constraint 32 --resume
#

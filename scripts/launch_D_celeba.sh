python launch_jeanzay.py --D_only \
--which_loss div --which_div Chi2 \
--shuffle --batch_size 128 --parallel\
--num_G_accumulations 1 --num_D_accumulations 1 --num_epochs 100 \
--num_D_steps 4 --D_lr 2e-4 \
--dataset CA64_denseflow_hdf5 --load_in_mem --num_workers 4 \
--G_ortho 0.0 \
--G_attn 0 --D_attn 0 \
--G_init N02 --D_init N02 \
--test_every 2000 --save_every 2000 --num_best_copies 1 --num_save_copies 2 --seed 0 --resume_no_optim
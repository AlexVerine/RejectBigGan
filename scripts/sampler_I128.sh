python launch_jeanzay.py --mode sampler --sample_random --sample_inception_metrics \
--dataset I128_hdf5 --parallel --shuffle  --num_workers 4 --batch_size 128  \
--num_G_accumulations 16  --num_D_accumulations 16 \
--num_D_steps 2 --G_lr 1e-10 --D_lr 1e-5 --D_B2 0.999 --G_B2 0.999 \
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
--name_suffix Donly --which_loss reject --which_div gan --partition gpu_p5 

for rate in 0.1  0.3  0.5  0.7  0.9
do 
	python launch_jeanzay.py --mode sampler --sample_random --sample_inception_metrics \
	--dataset I128_hdf5 --parallel --shuffle  --num_workers 4 --batch_size 128  \
	--num_G_accumulations 16  --num_D_accumulations 16 \
	--num_D_steps 2 --G_lr 1e-10 --D_lr 1e-5 --D_B2 0.999 --G_B2 0.999 \
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
	--name_suffix Donly --which_loss reject --which_div gan --partition gpu_p5 \
	--sampling OBRS --budget_test $rate 
done


for rate in -10 -5 0
do 
	python launch_jeanzay.py --mode sampler --sample_random --sample_inception_metrics \
	--dataset I128_hdf5 --parallel --shuffle  --num_workers 4 --batch_size 128  \
	--num_G_accumulations 16  --num_D_accumulations 16 \
	--num_D_steps 2 --G_lr 1e-10 --D_lr 1e-5 --D_B2 0.999 --G_B2 0.999 \
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
	--name_suffix Donly --which_loss reject --which_div gan --partition gpu_p5 \
	--sampling DRS --gamma_drs $rate --partition gpu_p5
done

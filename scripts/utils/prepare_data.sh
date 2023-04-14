#!/bin/bash
# python make_hdf5.py --dataset I128 --batch_size 256  --data_dir $WORK/data
python calculate_inception_moments.py --dataset I128_hdf5 --data_root $WORK/data --batch_size 256	
python calculate_vgg_features.py --dataset I128_hdf5 --data_root $WORK/data
python calculate_inception_features.py --dataset I128_hdf5 --data_root $WORK/data

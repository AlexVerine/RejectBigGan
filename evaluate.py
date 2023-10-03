
''' Evaluate
   This script loads a pretrained net and a weightsfile and sample '''
import functools
import math
import numpy as np
from tqdm import tqdm, trange
import logging
import os
import sys
import torch
import torch.nn as nn
from torch.nn import init
import torch.optim as optim
import torch.nn.functional as F
from torch.nn import Parameter as P
import torchvision

# Import my stuff
import inception_utils
import precision_recall_kyn_utils
import precision_recall_simon_utils
import utils
import losses



def run_eval(config):
  # Prepare state dict, which holds things like epoch # and itr #
  state_dict = {'itr': 0, 'epoch': 0, 'save_num': 0, 'save_best_num': 0,
                'best_IS': 0, 'best_FID': 999999, 'config': config}

  experiment_name = (config['experiment_name'] if config['experiment_name']
                       else utils.name_from_config(config))
  print(experiment_name)

  config['experiment_name'] = experiment_name
  # Optionally, get the configuration from the state dict. This allows for
  # recovery of the config provided only a state dict and experiment name,
  # and can be convenient for writing less verbose sample shell scripts.
  # Seed RNG
  utils.seed_rng(config['seed'])
  utils.setup_logging(config)

  # Setup cudnn.benchmark for free speed
  torch.backends.cudnn.benchmark = True
  
  # Import the model--this line allows us to dynamically select different files.
  logging.info('Experiment name is %s' % experiment_name)
  
  # Get Inception Score and FID
  get_inception_metrics = inception_utils.prepare_inception_metrics(config['dataset'], config['parallel'], config['no_fid'])
  # Prepare vgg metrics: Precision and Recall
  get_pr_metric = precision_recall_kyn_utils.prepare_pr_metrics(config)
  get_pr_curve = precision_recall_simon_utils.prepare_pr_curve(config)

  # Prepare a simple function get metrics that we use for trunc curves
  def get_metrics():
    sample = os.path.join(config['samples_root'], config['experiment_name'],'samples.npz')

    print(sample)
    # Psi, Rsi, Psa, Rsa = get_pr_curve(sample, "eval") 
    Psi, Rsi, Psa, Rsa = 0,0,0,0
    print("Sajj + Simon ")
    IS_mean, IS_std, FID = get_inception_metrics(sample, config['num_inception_images'], 
                                                 num_splits=10, 
                                                 prints=False,
                                                 use_torch=False)
    print("FID + IS")
    P, R, D, C = get_pr_metric(sample)
    print("Naaem + Kyn")
    # Prepare output string
    outstring = 'Using generated samples '
    outstring += '\nItr %d: PYTORCH UNOFFICIAL Inception Score is %3.3f +/- %3.3f, PYTORCH UNOFFICIAL FID is %5.4f' % (state_dict['itr'], IS_mean, IS_std, FID)
    outstring += '\nItr %d: Kynkäänniemi Precision is %2.3f, Kynkäänniemi Recall is %2.3f' % (state_dict['itr'], P*100, R*100)
    outstring += '\nItr %d: Sajjadi Precision is %2.3f, Sajjadi Recall is %2.3f' % (state_dict['itr'], Psa*100, Rsa*100)
    outstring += '\nItr %d: Simon Precision is %2.3f, Simon Recall is %2.3f' % (state_dict['itr'], Psi*100, Rsi*100)
    outstring += '\nItr %d: Naeem Density is %2.3f, Naeem Coverage is %2.3f' % (state_dict['itr'], D, C)

    utils.write_evaldata(config['logs_root'], experiment_name, config, 
                         {'itr': state_dict['itr'], 'IS_mean':IS_mean, 'IS_std' : IS_std, 'FID':FID, 'P':P, 'R':R,'D':D, 'C':C, 'Psa':Psa, 'Rsa':Rsa,  'Psi':Psi, 'Rsi':Rsi, })

    # utils.write_evaldata(config['logs_root'], experiment_name, config, 
    #                      {'P':P, 'R':R, 'D':D, 'C':C})
    logging.info(outstring)
    # logging.info('Calculating Inception metrics...')
  get_metrics()
  batch_sample = 100
  # logging.info('Preparing random sample sheet...')
  sample = os.path.join(config['samples_root'], config['experiment_name'],'samples.npz')

  images = torch.Tensor(np.load(sample)['imgs'])
  images = images[:batch_sample]
  torchvision.utils.save_image(images.float(),
                                '%s/%s/random_samples.jpg' % (config['samples_root'], experiment_name),
                                nrow=int(batch_sample**0.5),
                                normalize=True)


def main():
  # parse command line and run    
  parser = utils.prepare_parser()
  parser = utils.add_sample_parser(parser)
  config = vars(parser.parse_args())
  config['mode'] = 'eval'
  if config['data_root'] is None:
      config['data_root'] = os.environ.get('DATADIR', None)
  if config['data_root'] is None:
      ValueError("the following arguments are required: --data_dir")
  if config['data_root'] is None:
    config['data_root'] = os.environ.get('DATADIR', None)


  print(config)
  run_eval(config)
  
if __name__ == '__main__':    
  main()

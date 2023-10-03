''' Sample
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
from utils_rejection import get_sampler_function, get_sampling_function, test_sampling



def run_sampler(config):
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
  if config['config_from_name']:
    utils.load_weights(None, None, state_dict, config['weights_root'], 
                       config['experiment_name'], config['load_weights'], None,
                       strict=False, load_optim=False)
    # Ignore items which we might want to overwrite from the command line
    for item in state_dict['config']:
      if item not in ['z_var', 'base_root', 'batch_size', 'G_batch_size', 'use_ema', 'G_eval_mode']:
        config[item] = state_dict['config'][item]
  
  # update config (see train.py for explanation)
  config['resolution'] = utils.imsize_dict[config['dataset']]
  config['n_classes'] = utils.nclass_dict[config['dataset']]
  config['G_activation'] = utils.activation_dict[config['G_nl']]
  config['D_activation'] = utils.activation_dict[config['D_nl']]
  config = utils.update_config_roots(config)
  config['skip_init'] = True
  config['no_optim'] = True
  device = 'cuda'
  
  # Seed RNG
  utils.seed_rng(config['seed'])
  utils.setup_logging(config)

  # Setup cudnn.benchmark for free speed
  torch.backends.cudnn.benchmark = True
  
  # Import the model--this line allows us to dynamically select different files.
  model = __import__(config['model'])
  experiment_name = (config['experiment_name'] if config['experiment_name']
                       else utils.name_from_config(config))
  logging.info('Experiment name is %s' % experiment_name)
  utils.prepare_root(config)

  G = model.Generator(**config).cuda()
  D = model.Discriminator(**config).to(device)

  config['batch_size'] = config['batch_size_eval']
  # loaders = utils.get_data_loaders(**config)
  # Load weights
  logging.info('Loading weights...')
  # Here is where we deal with the ema--load ema weights or load normal weights
  utils.load_weights(G if not (config['use_ema']) else None, D, state_dict, 
                     config['weights_root'], experiment_name, config['load_weights'],
                     G if config['ema'] and config['use_ema'] else None,
                     strict=True, load_optim=False)
  # Update batch size setting used for G
  G_batch_size = max(config['G_batch_size'], config['batch_size']) 
  z_, y_ = utils.prepare_z_y(G_batch_size, G.dim_z, config['n_classes'],
                             device=device, fp16=config['G_fp16'], 
                             z_var=config['z_var'], trunc=config['trunc'])
  
  if config['G_eval_mode']:
    # logging.info('Putting G in eval mode..')
    G.eval()
    D.eval()
  else:
    logging.info('G is in %s mode...' % ('training' if G.training else 'eval'))
    
  #Sample function
  # sample = functools.partial(utils.sample, G=G, z_=z_, y_=y_, config=config)  

  sample = functools.partial(utils.sample, G=G, z_=z_, y_=y_, config=config)  
  Sampling, M = get_sampling_function(config, sample, D)
  Sampling.eval()
  # loaders = utils.get_data_loaders(**{**config, 'original':False})

  #load datasets: loaders[0]=G, loaders[1]=Dataset 
  sampler = get_sampler_function(config, sampling=Sampling, D=D, sample=sample)
  sample = functools.partial(sampler, test=False)
  print(f'Rate : {1/M:.3f}')


  if config['sampling'] is not None:
    if config['sampling'] == 'DRS':
      name = config['sampling'] + '_' +  str(config['gamma_drs'])
      config['sampling_params'] = config['gamma_drs']
    else:
      name = config['sampling'] + '_' +  str(config['budget']) 
      config['sampling_params'] = config['budget']
  else:
    name = "NoRS"
    config['sampling_params'] = 1

  batch_sample = 100
  # logging.info('Preparing random sample sheet...')
  # loaders = utils.get_data_loaders(**{**config, 'batch_size': config['batch_size'] ,
  #                                   'start_itr': state_dict['itr']})
  utils.seed_rng(config['seed'])

  images, labels = sample()    

  images = images[:batch_sample]

  new_images = torch.from_numpy(images.float().cpu().numpy())
  torchvision.utils.save_image(new_images.float(),
                                '%s/%s/%s.jpg' % (config['samples_root'], experiment_name, name),
                                nrow=int(batch_sample**0.5),
                                normalize=True)
  
  utils.seed_rng(config['seed'])
  N, Na = test_sampling(functools.partial(sampler, test=True))

  # Get Inception Score and FID
  get_inception_metrics = inception_utils.prepare_inception_metrics(config['dataset'], config['parallel'], config['no_fid'])
  # Prepare vgg metrics: Precision and Recall
  get_pr_metric = precision_recall_kyn_utils.prepare_pr_metrics(config)
  get_pr_curve = precision_recall_simon_utils.prepare_pr_curve(config)
  # Prepare a simple function get metrics that we use for trunc curves
  def get_metrics():
    utils.seed_rng(config['seed'])

    Psi, Rsi, Psa, Rsa = get_pr_curve(sample, "eval")
    # Psi, Rsi, Psa, Rsa = 0, 0, 0, 0 
    utils.seed_rng(config['seed'])

    IS_mean, IS_std, FID = get_inception_metrics(sample, config['num_inception_images'], 
                                                 num_splits=10, 
                                                 prints=False,
                                                 use_torch=False)
    # IS_mean, IS_std, FID = 0, 0, 0
    utils.seed_rng(config['seed'])

    P, R, D, C = get_pr_metric(sample)
    # P, R, D, C = 0, 0, 0, 0
    
    # Prepare output string
    outstring = 'Using %s weights ' % ('ema' if config['use_ema'] else 'non-ema')
    outstring += 'in %s mode, ' % ('eval' if config['G_eval_mode'] else 'training')
    outstring += 'with noise variance %3.3f, ' % z_.var
    outstring += 'over %d images, ' % config['num_inception_images']
    if config['accumulate_stats'] or not config['G_eval_mode']:
      outstring += 'with batch size %d, ' % G_batch_size
    if config['accumulate_stats']:
      outstring += 'using %d standing stat accumulations, ' % config['num_standing_accumulations']
    outstring += '\nItr %d: Estimated Acceptance Rate: %2.3f Vs Actual Acceptance Rate %2.3f' % (state_dict['itr'], 100/M, 100*Na/N)

    outstring += '\nItr %d: PYTORCH UNOFFICIAL Inception Score is %3.3f +/- %3.3f, PYTORCH UNOFFICIAL FID is %5.4f' % (state_dict['itr'], IS_mean, IS_std, FID)
    outstring += '\nItr %d: Kynkäänniemi Precision is %2.3f, Kynkäänniemi Recall is %2.3f' % (state_dict['itr'], P*100, R*100)
    outstring += '\nItr %d: Sajjadi Precision is %2.3f, Sajjadi Recall is %2.3f' % (state_dict['itr'], Psa*100, Rsa*100)
    outstring += '\nItr %d: Simon Precision is %2.3f, Simon Recall is %2.3f' % (state_dict['itr'], Psi*100, Rsi*100)
    outstring += '\nItr %d: Naeem Density is %2.3f, Naeem Coverage is %2.3f' % (state_dict['itr'], D, C)

    utils.write_evaldata(config['logs_root'], experiment_name, config, 
                         {'itr': state_dict['itr'], 'IS_mean':IS_mean, 'IS_std' : IS_std, 
                          'FID':FID, 'P':P, 'R':R, 
                          'Psa':Psa, 'Rsa':Rsa,  'Psi':Psi, 'Rsi':Rsi,
                          "D":D, "C":C,  
                          "sampling": config['sampling'],'params': config['sampling_params'] , 'rate':1/M})

    logging.info(outstring)
  if config['sample_inception_metrics']: 
    # logging.info('Calculating Inception metrics...')
    get_metrics()
    
  # # Sample truncation curve stuff. This is basically the same as the inception metrics code
  # if config['sample_trunc_curves']:
  #   start, step, end = [float(item) for item in config['sample_trunc_curves'].split('_')]
  #   logging.info('Getting truncation values for variance in range (%3.3f:%3.3f:%3.3f)...' % (start, step, end))
  #   for var in np.arange(start, end + step, step):     
  #     z_.var = var
  #     # Optionally comment this out if you want to run with standing stats
  #     # accumulated at one z variance setting
  #     if config['accumulate_stats']:
  #       utils.accumulate_standing_stats(G, z_, y_, config['n_classes'],
  #                                   config['num_standing_accumulations'])
  #     get_metrics()

def main():
  # parse command line and run    
  parser = utils.prepare_parser()
  parser = utils.add_sample_parser(parser)
  config = vars(parser.parse_args())
  config['mode'] = 'sample'
  if config['data_root'] is None:
      config['data_root'] = os.environ.get('DATADIR', None)
  if config['data_root'] is None:
      ValueError("the following arguments are required: --data_dir")
  if config['data_root'] is None:
    config['data_root'] = os.environ.get('DATADIR', None)


  print(config)
  run_sampler(config)
  
if __name__ == '__main__':    
  main()

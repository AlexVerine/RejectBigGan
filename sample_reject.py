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
import train_fns
from utils_rejection import get_sampler_function, get_sampling_function, test_sampling


def run_samplerD(config):
  # Prepare state dict, which holds things like epoch # and itr #
  state_dict = {'itr': 0, 'epoch': 0, 'save_num': 0, 'save_best_num': 0,
                'best_IS': 0, 'best_FID': 999999, 'config': config}

  experiment_name = (config['experiment_name'] if config['experiment_name']
                       else utils.name_from_config(config))
  print(experiment_name)

  config['experiment_name'] = experiment_name


  
  # update config (see train.py for explanation)
  config['resolution'] = utils.imsize_dict[config['dataset']]
  config['n_classes'] = 1
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


  config['batch_size'] = config['batch_size_eval']
  if "hdf5" in config['dataset']:
    config['dataset'] = config['dataset'].replace("_hdf5", "_eval_hdf5")
  else:
    config['dataset'] = config['dataset'] + "_eval"
  # Next, build the model
  D = model.Discriminator(**config).to(device)
  D.eval()
    # Load weights
  logging.info('Loading weights...')
  # Here is where we deal with the ema--load ema weights or load normal weights
  utils.load_weights(None, D, None,
                    config['weights_root'], experiment_name, 
                    config['load_weights'] if config['load_weights'] else None,
                    None,
                    load_optim=False)

  # If parallel, parallelize the D module
  if config['parallel']:
    D = nn.DataParallel(D)




  if config["sampling"] is None:
   
    loaders = utils.get_data_loaders(**{**config, 'original':True, 'load_in_mem':False, "mode":'train'})
    print(len(loaders[0].dataset))
    test = train_fns.D_evaluating_function(D, loaders, config, mode="eval")
    logging.info("Testing on trained data:")
    test()
    
    loaders = utils.get_data_loaders(**{**config, 'original':True, 'load_in_mem':False, "mode":'sample'})
    print(len(loaders[0].dataset))
    test = train_fns.D_evaluating_function(D, loaders, config, mode="eval")
    logging.info("Testing on eval data:")
    test()

  loaders = utils.get_data_loaders(**{**config, 'original':False})
  print(len(loaders[0].dataset))

  iterative_loader = iter(loaders[0])
  Sampling, M = get_sampling_function(config, loaders[0], D)
  Sampling.eval()
  #load datasets: loaders[0]=G, loaders[1]=Dataset 
  sampler = get_sampler_function(config, sampling=Sampling, D=D)
  sample = functools.partial(sampler, loader=iterative_loader, test=False)


  if config['sampling'] is not None:
    if config['sampling'] == 'DRS':
      name = config['sampling'] + '_' +  str(config['gamma_drs'])
    else:
      name = config['sampling'] + '_' +  str(config['budget']) 
  else:
    name = "NoRS"

  # Get Inception Score and FID
  get_inception_metrics = inception_utils.prepare_inception_metrics(config['dataset'].split('_')[0], config['parallel'], config['no_fid'])
  # Prepare vgg metrics: Precision and Recall
  get_pr_metric = precision_recall_kyn_utils.prepare_pr_metrics(config)
  get_pr_curve = precision_recall_simon_utils.prepare_pr_curve(config)
    # Prepare a simple function get metrics that we use for trunc curves

  if config['sample_inception_metrics']: 
    logging.info('Calculating metrics...')
    with torch.no_grad():
      x = torch.Tensor().cuda()
      while x.shape[0]<config['num_inception_images'] or x.shape[0]<config['num_pr_images']:
        try:
          x  = torch.cat([x, sample()[0]], dim=0)
        except StopIteration:
          logging.info(f"Not enough samples in Dataset Eval. Stopped at {x.shape[0]}.")
          iterative_loader = iter(loaders[0])
          sample = functools.partial(sampler, loader=iterative_loader, test=True)
        
    if x.shape[0]>= config['num_pr_images']:
      P, R, D, C = get_pr_metric(x[:config['num_pr_images']])
      Ps, Rs = get_pr_curve(x[:config['num_pr_images']], name)
    else:
      P, R, Ps, Rs, D, C = 0, 0, 0, 0
    if x.shape[0]>= config['num_inception_images']:
      IS_mean, IS_std, FID = get_inception_metrics(x[:config['num_inception_images']], config['num_inception_images'], 
                                                 num_splits=10, 
                                                 prints=False,
                                                 use_torch=False)
    else:
      IS_mean, IS_std, FID = 0, 0, 0
    
    iterative_loader = iter(loaders[0])
    sample = functools.partial(sampler, loader=iterative_loader, test=True)
    N, Na = test_sampling(sample)


    # Prepare output string
    outstring = 'On %s ' % (config['dataset'])
    outstring += 'using %s weights ' % (config['load_weights'])

    outstring += 'with %s, ' % (config['sampling'])
    if config['sampling'] == 'DRS':
      outstring += f'using Gamma={config["gamma_drs"]:.2f} and eps={config["eps_drs"]:.3e}.'
      outstring += f'\nEstimated Optimal Rate -  {100/M:.2f}%.'

    elif config['sampling'] == 'OBRS':
      outstring += f'using Budget={config["budget"]:.2f}.'
      outstring += f'\nEstimated Optimal Rate -  {100/M:.2f}%.'

    outstring += f'\nTest on rate - Sampled {N}, Accepted: {Na}, Rate: {100*Na/N:.2f}%.'
    outstring += '\nItr %d: PYTORCH UNOFFICIAL Inception Score is %3.3f +/- %3.3f, PYTORCH UNOFFICIAL FID is %5.4f' % (state_dict['itr'], IS_mean, IS_std, FID)
    outstring += '\nItr %d: Kynkäänniemi Precision is %2.3f, Kynkäänniemi Recall is %2.3f' % (state_dict['itr'], P*100, R*100)
    outstring += '\nItr %d: Naeem Density is %2.3f, Naeem Coverage is %2.3f' % (state_dict['itr'], D*100, C*100)
    outstring += '\nItr %d: Simon Precision is %2.3f, Simon Recall is %2.3f' % (state_dict['itr'], Ps*100, Rs*100)
    logging.info(outstring)

    utils.write_evaldata(config['logs_root'], experiment_name, config, 
                         {'itr': state_dict['itr'], 'IS_mean':IS_mean, 'IS_std' : IS_std, 'FID':FID, 'P':P, 'R':R, 'Rs':Rs, 'Ps':Ps, 'D':D, 'C':C,  'rate':Na/N,
                          'sampling': config['sampling'],'gamma': config['gamma_drs'], 'budget': config['budget'] })


  if config['sample_random']:
    iterative_loader = iter(loaders[0])
    sample = functools.partial(sampler, loader=iterative_loader, test=False)

    batch_sample = 100
    images, labels = sample()    
    images = images[:batch_sample]
    if not os.path.isdir('%s/%s' % (config['samples_root'], experiment_name)):
      os.mkdir('%s/%s' % (config['samples_root'], experiment_name))
    if config['sampling'] is not None:
      if config['sampling'] == 'DRS':
        name = config['sampling'] + '_' +  str(config['gamma_drs'])
      else:
        name = config['sampling'] + '_' +  str(config['budget']) 
    else:
      name = "NoRS"
    torchvision.utils.save_image(images.float(),
                                 '%s/%s/random_samples_%s.jpg' % (config['samples_root'], experiment_name, name),
                                 nrow=int(batch_sample**0.5),
                                 normalize=True)

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
  run_samplerD(config)
  
if __name__ == '__main__':    
  main()

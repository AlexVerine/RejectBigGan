""" BigGAN: The Authorized Unofficial PyTorch release
    Code by A. Brock and A. Andonian
    This code is an unofficial reimplementation of
    "Large-Scale GAN Training for High Fidelity Natural Image Synthesis,"
    by A. Brock, J. Donahue, and K. Simonyan (arXiv 1809.11096).

    Let's go.
"""

import os
import functools
import math
import numpy as np
from tqdm import tqdm, trange

import time
import torch
import torch.nn as nn
from torch.nn import init
import torch.optim as optim
import torch.nn.functional as F 
from torch.nn import Parameter as P
import torchvision
import logging
# Import my stuff
# import inception_utils
# import precision_recall_kyn_utils
# import precision_recall_simon_utils

import utils
import losses
import train_fns
from sync_batchnorm import patch_replication_callback

# The main training file. Config is a dictionary specifying the configuration
# of this training run.
def runD(config):

  # Update the config dict as necessary
  # This is for convenience, to add settings derived from the user-specified
  # configuration into the config-dict (e.g. inferring the number of classes
  # and size of the images from the dataset, passing in a pytorch object
  # for the activation specified as a string)
  config['resolution'] = utils.imsize_dict[config['dataset']]
  config['n_classes'] = 1
  # config['G_activation'] = utils.activation_dict[config['G_nl']]
  config['D_activation'] = utils.activation_dict[config['D_nl']]
  # By default, skip init if resuming training.
  if config['resume']:
    logging.info('Skipping initialization for training resumption...')
    config['skip_init'] = True
  config = utils.update_config_roots(config)
  device = 'cuda'
  # Prepare root folders if necessary
  utils.prepare_root(config)
  # Seed RNG
  utils.seed_rng(config['seed'])

  # Setup cudnn.benchmark for free speed
  torch.backends.cudnn.benchmark = True

  # Import the model--this line allows us to dynamically select different files.
  model = __import__(config['model'])
  experiment_name = (config['experiment_name'] if config['experiment_name']
                       else utils.name_from_config(config))
  config['experiment_name'] = experiment_name
  print(experiment_name)



  utils.setup_logging(config)
  logging.info('Experiment name is %s' % experiment_name)
  config['resume'] += config['resume_no_optim']

  # Next, build the model
  D = model.Discriminator(**config).to(device)
  
  
  # FP16?
  if config['D_fp16']:
    logging.info('Casting D to fp16...')
    D = D.half()
    # Consider automatically reducing SN_eps?
  logging.info(D)
  logging.info('Number of params in D: {}'.format(
    *[sum([p.data.nelement() for p in net.parameters()]) for net in [D]]))
  # Prepare state dict, which holds things like epoch # and itr #
  state_dict = {'itr': 0, 'epoch': 0, 'save_num': 0, 'save_best_num': 0,
                'best_Acc':0, 'config': config}

  # If loading from a pre-trained model, load weights
  if config['resume']:
    logging.info('Loading weights...')
    utils.load_weights(None, D, None,
                       config['weights_root'], experiment_name, 
                       config['load_weights'] if config['load_weights'] else None,
                       None, 
                      load_optim=not config['resume_no_optim'])
    logging.info(f"Resume training with lrD: {D.optim.param_groups[0]['lr']:e}")

  # If parallel, parallelize the D module
  if config['parallel']:
    D = nn.DataParallel(D)
    if config['cross_replica']:
      patch_replication_callback(D)

  # Prepare loggers for stats; metrics holds test metrics,
  # lmetrics holds any desired training metrics.
  test_metrics_fname = '%s/%s_log.jsonl' % (config['logs_root'],
                                            experiment_name)
  train_metrics_fname = '%s/%s' % (config['logs_root'], experiment_name)
  logging.info('Inception Metrics will be saved to {}'.format(test_metrics_fname))
  test_log = utils.MetricsLogger(test_metrics_fname, 
                                 reinitialize=(not config['resume']))
  logging.info('Training Metrics will be saved to {}'.format(train_metrics_fname))
  train_log = utils.MyLogger(train_metrics_fname, 
                             reinitialize=(not config['resume']),
                             logstyle=config['logstyle'])
  # Write metadata
  utils.write_metadata(config['logs_root'], experiment_name, config, state_dict)
  # Prepare data; the Discriminator's batch size is all that needs to be passed
  # to the dataloader, as G doesn't require dataloading.
  # Note that at every loader iteration we pass in enough data to complete
  # a full D iteration (regardless of number of D steps and accumulations)
  D_batch_size = (config['batch_size'] * config['num_D_steps']
                  * config['num_D_accumulations'])
  loaders = utils.get_data_loaders(**{**config, 'batch_size': D_batch_size,
                                      'start_itr': state_dict['itr']})
  if config['use_multiepoch_sampler']:
    size_loader = int(np.ceil(loaders[1].sampler.num_samples/D_batch_size))
  else:
    size_loader = len(loaders[1])
  config['total_itr'] = int((size_loader)*(config['num_epochs']-state_dict['epoch']))

  if (size_loader//5)<500:
    config['log_itr'] = [i*(size_loader//5) for i in range(5)]+[size_loader-1]
  else:
    config['log_itr'] =  [size_loader-1]
  print(config['log_itr'])
  

  # Loaders are loaded, prepare the training function
  if config['parallel']:
    train = train_fns.D_training_function(D, D.module.optim, config)
  else:
    train = train_fns.D_training_function(D, D.optim, config)
  test = train_fns.D_evaluating_function(D, loaders, config)
 
  logging.info(f'Beginning training at epoch {state_dict["epoch"]} for {config["total_itr"]} iterations.')
  t_init = time.time()
  # Train for specified number of epochs, although we mostly track G iterations.
  for epoch in range(state_dict['epoch'], config['num_epochs']):    
    # Which progressbar to use? TQDM or my own
    t0 = time.time()
    dataloader_iterator = iter(loaders[0])
  
    for i, (x_real, _) in enumerate(loaders[1]):
      try:
          x_fake, y_fake = next(dataloader_iterator)
      except StopIteration:
          dataloader_iterator = iter(loaders[0])
          x_fake, y_fake = next(dataloader_iterator)

      if i % size_loader  == 0 :
        t0 = time.time()

      # Increment the iteration counter
      state_dict['itr'] += 1
      # Make sure G and D are in training mode, just in case they got set to eval
      D.train()

      if config['D_fp16']:
        x_real, x_fake = x_real.to(device).half(), x_fake.to(device).half()
        y_fake = y_fake.to(device).half()
      else:
        x_real, x_fake = x_real.to(device), x_fake.to(device)
        y_fake = y_fake.to(device)

      metrics = train(x_real, x_fake, y_fake)
      train_log.log(itr=int(state_dict['itr']), **metrics)
      
      # Every sv_log_interval, log singular values
      if (config['sv_log_interval'] > 0) and (not (state_dict['itr'] % config['sv_log_interval'])):
        train_log.log(itr=int(state_dict['itr']), 
                      **{**utils.get_SVs(D, 'D')})

      # If using my progbar, print metrics.
      if i in config["log_itr"]:
        e = 1+ i//size_loader if config['use_multiepoch_sampler'] else epoch

        logging.info(f'[{e:d}/{config["num_epochs"]:d}]({i+1}/{size_loader-1})({int(time.time()-t0):d}s/{int((size_loader -i%size_loader)*(time.time()-t0)/(i%size_loader+1)):d}s) : {state_dict["itr"] }     '+ 'Mem used (Go) {:.2f}/{:.2f}'.format(torch.cuda.mem_get_info(0)[1]/1024**3
                       -torch.cuda.mem_get_info(0)[0]/1024**3, torch.cuda.mem_get_info(0)[1]/1024**3))

        logging.info('\t'+', '.join(['%s : %+4.3f' % (key, metrics[key])
                           for key in metrics]))
          # logging.info()
          # logging.info(', '.join(['itr: %d' % state_dict['itr']] 
          #                  + ['%s : %+4.3f' % (key, m etrics[key])
          #                  for key in metrics]))
      if not (state_dict['itr'] % config['test_every']) or state_dict["itr"] == config["total_itr"]:
        with torch.no_grad():
          test(state_dict, test_log, experiment_name)
        logging.info(f'\tEstimated time: {((time.time()-t_init)*config["total_itr"]/state_dict["itr"] - (time.time()-t_init))// 86400:.0f} days, '
              + f'{ ( (( time.time()-t_init)*config["total_itr"]/state_dict["itr"] - (time.time()-t_init))% 86400) // 3600:2.1f} hours and ' 
                      + f'{ (((( time.time()-t_init)*config["total_itr"]/state_dict["itr"] - (time.time()-t_init))% 86400) % 3600) //60 :2.1f} minutes.')

    # Increment epoch counter at end of epoch
    state_dict['epoch'] += 1

def main():
  # parse command line and run
  parser = utils.prepare_parser()
  config = vars(parser.parse_args())
  config['mode'] = 'train'

  if config['data_root'] is None:
      config['data_root'] = os.environ.get('DATADIR', None)
  if config['data_root'] is None:
      ValueError("the following arguments are required: --data_dir")
  if config['data_root'] is None:
    config['data_root'] = os.environ.get('DATADIR', None)


  print(config)
  runD(config)

if __name__ == '__main__':
  main()

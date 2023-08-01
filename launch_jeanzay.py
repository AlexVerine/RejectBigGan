import os
import sys
import submitit
from train import run 
from sample import run_sampler
from train_D import runD
from utils import prepare_parser, add_sample_parser, add_cluster_parser, prepare_root, name_from_config

def main():
  # parse command line and run
    parser = prepare_parser()
    parser = add_sample_parser(parser)
    parser = add_cluster_parser(parser)

    config = vars(parser.parse_args())
    if config['debug']:
        experiment_name ='folder_debug'
        config['experiment_name'] = experiment_name
    else:
        experiment_name = (config['experiment_name'] if config['experiment_name']
            else name_from_config(config))
        config['experiment_name'] = experiment_name
    print('Experiment name is %s' % experiment_name)
    
    prepare_root(config)
    config['train_dir'] = os.path.join('logs',config['experiment_name']) 
    #   print(config)
    
    config['cmd'] = f"python3 {' '.join(sys.argv)}"
    
    # cluster gpu memory constraint
    if config['mem_constraint'] is not None:
        config['mem_constraint'] = f"v100-{config['mem_constraint']}g"
        print(f'Mem Constraint : {config["mem_constraint"]}')

    if config['partition'] == "gpu_p5":
        config['slurm_account'] = "esq@a100"
        config['mem_constraint'] = 'a100'
    else:
        config['slurm_account'] = "esq@v100"
    if config['ngpus'] is None:
        if config['partition'] == "gpu_p5" or config['partition'] == "gpu_p2":
            config['ngpus'] = 8
        else:
            config['ngpus'] = 4

    config['nnodes'] = 1
    config['tasks_per_node'] = 1
    if config['partition'] == 'gpu_p5':
        config['cpus_per_task'] = 64
    elif config['partition'] == "gpu_p13":
        config['cpus_per_task'] = 40
    elif config['partition'] == "gpu_p2":
        config['cpus_per_task'] = 24
    elif config['partition'] == "gpu_p4":
        config['cpus_per_task'] = 48
    # get the path of datsets
    
    if config['data_root'] is None:
        config['data_root'] = os.environ.get('DATADIR', None)
    if config['data_root'] is None:
        ValueError("the following arguments are required: --data_dir")
    
    cluster = 'slurm' if not config['local'] else 'local'
    executor = submitit.AutoExecutor(folder=config['train_dir'], cluster=cluster)
    executor.update_parameters(
        gpus_per_node=config['ngpus'],
        nodes=config['nnodes'],
        tasks_per_node=config['tasks_per_node'],
        cpus_per_task=config['cpus_per_task'],
        stderr_to_stdout=True,
        slurm_account=config['slurm_account'],
        slurm_job_name=f"{config['dataset']}{config['which_loss'][:2]}",
        slurm_partition=config['partition'],
        slurm_qos=config['qos'],
        slurm_constraint=config['mem_constraint'],
        slurm_signal_delay_s=0,
        timeout_min=config['timeout'],
        slurm_exclude=None
    )

    if config['mode'] ==  'train':
        if config['D_only']:
            job = executor.submit(runD, config)
        else:
            job = executor.submit(run, config)
    else:
        job = executor.submit(run_sampler, config)
    
if __name__ == '__main__':
  main()

#!/gpfslocalsup/pub/anaconda-py3/2021.05/envs/pytorch-gpu-1.9.0+py3.9/bin/python
import os
import sys
import argparse
import subprocess


def run_interactive(args):

  cmd = ["salloc", "--exclusive", "--nodes=1", f"--time={args.timeout}",
         f"--partition={args.partition}", f"--account={args.account}"]
  if args.constraint:
    cmd += [f"--constraint={args.constraint}"]
  if not args.cpu:
    cmd += [f"--gres=gpu:{args.ngpus}"]
  if not args.no_srun:
    cmd += ["srun --pty bash"]
  cmd = ' '.join(cmd)
  print(cmd)
  os.system(cmd)


if __name__ == '__main__':

  parser = argparse.ArgumentParser()
  parser.add_argument("--account", type=str, default='yxj@gpu',
                      help="Slurm account to use.")
  parser.add_argument("--ngpus", type=int, default=4, 
                      help="Number of GPUs to use.")
  parser.add_argument("--partition", type=str, default=None,
                      help="Partition to use for Slurm.")
  parser.add_argument("--constraint", type=int, default=None, choices=[16, 32],
                      help="Add constraint for choice of GPUs: 16 or 32")
  parser.add_argument("--timeout", type=int, default=2*60,
                      help="Time of the Slurm job in minutes for training.")
  parser.add_argument("--no-srun", action='store_true', default=False,
                      help="Run without srun.")
  parser.add_argument("--cpu", action='store_true', default=False,
                      help="Run interactive job on CPU.")
  parser.add_argument("--wait", action='store_true')
  args = parser.parse_args()

  if args.constraint is not None:
    args.constraint = f"v100-{args.constraint}g"

  if args.partition is None and not args.cpu:
    # get first available partition
    for partition in ['gpu_p13', 'gpu_p2', 'gpu_p4']:
      cmd = f"sinfo --partition={partition} --states=idle --noheader"
      proc = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE)
      output = proc.stdout.read().decode('UTF-8')
      n_nodes_avail = int(output.split()[3])
      if n_nodes_avail > 0:
        args.partition = partition
        print(f"Partition '{args.partition}' available")
        break
    else:
      if args.wait:
        args.partition = 'gpu_p13'
      else:
        print('No partition available')
        sys.exit(0)

  elif args.partition == 'gpu_p5':
    args.account = 'yxj@a100'
    args.constraint = 'a100'
    args.ngpus = 8

  elif args.cpu:
    print('run on cpu')
    args.account = args.account.replace('gpu', 'cpu')
    args.partition = 'cpu_p1'

  run_interactive(args)

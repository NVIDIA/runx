import os
import yaml
from .collections import AttrDict

__C = AttrDict()
cfg = __C

__C.CLUSTER = None

# Current dir
__C.RUNROOT = None

# Environment variables to set for the runs
__C.ENV = None

# Batch submission command
__C.BATCH_CMD = None

# Arguments to supply to batch submission command
__C.BATCH_ARGS = None

# Ignore these patterns of files/dirs when copying code
__C.CODE_IGNORE_PATTERNS = None

# By default, this points at the code (from the current directory) but you can override this
# in case you need to point at other stuff.
__C.PYTHONPATH = None

# The initial portion of the training command
__C.CMD = None

# Experiment name
__C.EXP_NAME = None



def read_config_file(args=None):
    """
    Read in the .runx file and return it in the form of a dict
    """

    local_config_fn = './.runx'
    home = os.path.expanduser('~')
    global_config_fn = '{}/.config/runx.yml'.format(home)

    if args is not None and hasattr(args, 'config_file') and \
       args.config_file is not None and \
       os.path.isfile(args.config_file):
        config_fn = args.config_file
    elif os.path.isfile(local_config_fn):
        config_fn = local_config_fn
    elif os.path.exists(global_config_fn):
        config_fn = global_config_fn
    else:
        raise('can\'t find file ./.runx or ~/.config/runx.yml config files')

    if 'FullLoader' in dir(yaml):
        global_config = yaml.load(open(config_fn), Loader=yaml.SafeLoader)
    else:
        global_config = yaml.safe_load(open(config_fn))
    return global_config


def read_cluster(args):
    """
    Determine which cluster we're using.
    """
    if args.cluster is not None:
        cluster = args.cluster
    elif 'CLUSTER' in os.environ:
        cluster = os.environ['CLUSTER']
    else:
        cluster = 'NO_CLUSTER'
    return cluster


def read_config_and_experiment(args):
    """
    Read in .runx file and load it into global cfg
    """
    
    # determine which cluster (if any) we are working with
    cfg.CLUSTER = read_cluster(args)

    # capture the current dir
    cfg.RUNROOT = os.getcwd()

    # Read .runx file and pull out per-cluster information
    runx_config = read_config_file(args)

    assert cfg.CLUSTER in runx_config, \
        f'Looked for cluster {cfg.CLUSTER} in .runx file, but didn\'t find it'
    cluster_config = runx_config[cfg.CLUSTER]
    cfg.ENV = cluster_config['ENV']
    assert 'LOG_ROOT' in cfg.ENV, f'Must define LOG_ROOT in ENV'
    
    cfg.BATCH_CMD = cluster_config['BATCH_CMD']
    cfg.BATCH_ARGS = cluster_config['BATCH_ARGS']
    
    if 'CODE_IGNORE_PATTERNS' in runx_config:
        code_ignore_patterns = runx_config['CODE_IGNORE_PATTERNS']
    else:
        # use these by default:
        code_ignore_patterns = '.git,*.pyc,docs*,test*'
    code_ignore_patterns = code_ignore_patterns.split(',')
    # make double sure we don't copy checkpoints
    code_ignore_patterns.append('*.pth') 
    cfg.CODE_IGNORE_PATTERNS = code_ignore_patterns

    # Read in experiment yaml
    if hasattr(args, 'exp_yml') and args.exp_yml is not None:
        if not os.path.exists(args.exp_yml):
            print('couldn\'t find experiment file {}'.format(args.exp_yml))
            sys.exit()

        merge_experiment(args)
        
        
def merge_experiment(args):
    """
    Read in experiment yaml and merge it into global cfg.
    Determine experiment name

    Required fields:
      CMD
      HPARAMS

    Optional fields:
      PYTHONPATH
      BATCH_ARGS - can override global settings
    """  
    exp_config = yaml.load(open(args.exp_yml), Loader=yaml.SafeLoader)

    # Check on required fields
    assert 'CMD' in exp_config, f'couldn\'t find CMD in {args.exp_yml}'
    assert 'HPARAMS' in exp_config, f'couldn\'t find HPARAMS in {args.exp_yml}'

    cfg.CMD = exp_config['CMD']
    cfg.HPARAMS = exp_config['HPARAMS']

    if 'PYTHONPATH' in exp_config:
        cfg.PYTHONPATH = exp_config['PYTHONPATH']

    # Merge in experiment's BATCH_ARGS
    if 'BATCH_ARGS' in exp_config:
        cfg.BATCH_ARGS.update(exp_config['BATCH_ARGS'])

    # Normally, the experiment name comes from the experiment yaml basename,
    # but there are some ways to override it ...
    if args.exp_name is not None:
        cfg.EXP_NAME = args.exp_name
    elif args.exp_yml is None:
        cfg.EXP_NAME = 'none'
    else:
        cfg.EXP_NAME = os.path.splitext(os.path.basename(args.exp_yml))[0]


        
    

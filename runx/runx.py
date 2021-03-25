"""
Copyright 2020 Nvidia Corporation

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors
   may be used to endorse or promote products derived from this software
   without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
"""
from __future__ import print_function
from collections import OrderedDict
from coolname import generate_slug
from datetime import datetime
from shutil import copytree, ignore_patterns

import argparse
import itertools
import os
import math
import random
import re
import sys
import subprocess
from typing import Iterable, List
import yaml

from .config import cfg
from .distributions import enumerate_hparams
from .farm import build_farm_cmd, upload_to_ngc
from .utils import read_config, save_hparams, exec_cmd


parser = argparse.ArgumentParser(description='Experiment runner')
parser.add_argument('exp_yml', type=str,
                    help='experiment yaml file')
parser.add_argument('--tag', type=str, default=None,
                    help='tag label for run')
parser.add_argument('--no_run', '-n', action='store_true',
                    help='don\'t run')
parser.add_argument('--interactive', '-i', action='store_true',
                    help='run interactively instead of submitting to farm')
parser.add_argument('--no_cooldir', action='store_true',
                    help='no coolname, no datestring')
parser.add_argument('--farm', type=str, default=None,
                    help='Select farm for workstation submission')
parser.add_argument('--yml_params', action='store_true',
                    help='Hyperparameters are specified via a config yaml as opposed to through the command line.')
args = parser.parse_args()


def expand_hparams(hparams):
    """
    Construct the training script args from the hparams
    """
    cmd = ''
    for field, val in hparams.items():
        if type(val) is bool:
            if val is True:
                cmd += '--{} '.format(field)
        elif val != 'None':
            cmd += '--{} {} '.format(field, val)
    return cmd


def construct_cmd(cmd, hparams, logdir):
    """
    Build training command by starting with user-supplied 'CMD'
    and then adding in hyperparameters, which came from expanding the
    cross-product of all permutations from the experiment yaml file.

    We always copy the code to the target logdir and run from there.

    :cmd: farm submission command
    :hparams: hyperparams for training command
    """
    # Only add to the command line if YML_PARAMS isn't specified
    if not cfg.YML_PARAMS:
        # First, add hyperparameters
        cmd += ' ' + expand_hparams(hparams)

    # Expand PYTHONPATH, if necessary
    if cfg.PYTHONPATH is not None:
        pythonpath = cfg.PYTHONPATH
        pythonpath = pythonpath.replace('LOGDIR', logdir)
    else:
        pythonpath = f'{logdir}/code'

    # For signalling reasons, we have to insert the exec here when using submit_job.
    # Nvidia-internal thing.
    exec_str = ''
    if 'submit_job' in cfg.SUBMIT_CMD:
        exec_str = 'exec'

    cmd = f'cd {logdir}/code; PYTHONPATH={pythonpath} {exec_str} {cmd}'
    return cmd


def save_cmd(cmd, logdir):
    """
    Record the submit command
    """
    fp = open(os.path.join(logdir, 'submit_cmd.sh'), 'w')
    fp.write(cmd)
    fp.write('\n')
    fp.close()


def islist(elem):
    return type(elem) is list or type(elem) is tuple


def get_field(adict, f, required=True):
    if required:
        assert f in adict, 'expected {} to be defined in experiment'.format(f)
    return adict[f] if f in adict else None


def do_keyword_expansion(alist, pairs):
    """
    Substitute a string in place of certain keywords
    """
    if isinstance(alist, (list, tuple)):
        for i, v in enumerate(alist):
            alist[i] = do_keyword_expansion(v, pairs)
        return alist
    elif isinstance(alist, dict):
        for a_k, a_v in alist.items():
            alist[a_k] = do_keyword_expansion(a_v, pairs)
        return alist
    elif isinstance(alist, str):
        ret = alist
        for k, v in pairs:
            ret = ret.replace(k, v)
        return ret
    else:
        return alist


def make_cool_names():
    tagname = args.tag + '_' if args.tag else ''
    datestr = datetime.now().strftime("_%Y.%m.%d_%H.%M")
    if args.no_cooldir:
        coolname = tagname
    else:
        coolname = tagname + generate_slug(2) + datestr

    # Experiment directory is the parent of N runs
    expdir = os.path.join(cfg.LOGROOT, cfg.EXP_NAME)

    # Each run has a logdir
    logdir_name = coolname
    logdir = os.path.join(expdir, logdir_name)

    # Jobname is a unique name for the batch job
    job_name = '{}_{}'.format(cfg.EXP_NAME, coolname)
    return job_name, logdir, logdir_name, expdir


def copy_code(logdir, runroot, code_ignore_patterns):
    """
    Copy sourcecode to logdir's code directory
    """
    print('Copying codebase to {} ...'.format(logdir))
    tgt_code_dir = os.path.join(logdir, 'code')
    if code_ignore_patterns is not None:
        code_ignore_patterns = ignore_patterns(*code_ignore_patterns)
    copytree(runroot, tgt_code_dir, ignore=code_ignore_patterns)


def write_yml_params(logdir, hparams):
    with open(os.path.join(logdir, 'code', 'hyper_parameters.yml'), 'w') as fd:
        yaml.dump(hparams, fd)


def hacky_substitutions(cmd, hparams, resource_copy, logdir, runroot, replica):
    replace_list = [('LOGDIR', logdir), ('REPLICA', str(replica))]
    # Substitute the true logdir in for the magic variable LOGDIR
    hparams = do_keyword_expansion(hparams, replace_list)
    resource_copy = do_keyword_expansion(resource_copy, replace_list)
    cmd = do_keyword_expansion(cmd, replace_list)

    # Build hparams to save out after LOGDIR but before deleting
    # the key 'SUBMIT_JOB.NODES', so that it is part of the hparams saved
    # This is done so that we can see the node count in sumx.
    hparams_out = hparams.copy()

    # SUBMIT_JOB.NODES is a hyperparmeter that sets the node count
    # This is actually a resource, so when we find this arg, we delete
    # it from the list of hyperparams that the training script sees.
    if 'SUBMIT_JOB.NODES' in hparams:
        resource_copy['nodes'] = hparams['SUBMIT_JOB.NODES']
        del hparams['SUBMIT_JOB.NODES']
    if 'SUBMIT_JOB.PARTITION' in hparams:
        resource_copy['partition'] = hparams['SUBMIT_JOB.PARTITION']
        del hparams['SUBMIT_JOB.PARTITION']

    # Record the directory from whence the experiments were launched
    hparams_out['srcdir'] = runroot

    return cmd, hparams_out


def get_tag(hparams):
    # Pull tag from hparams and then remove it
    # Also can do variable substitution into tag
    if 'RUNX.TAG' in hparams:
        tag_val = hparams['RUNX.TAG']

        # do variable expansion:
        for sub_key, sub_val in hparams.items():
            search_str = '{' + sub_key + '}'
            tag_val = re.sub(search_str, str(sub_val), tag_val)
        hparams['RUNX.TAG'] = tag_val
        args.tag = tag_val
        del hparams['RUNX.TAG']


def skip_run(hparams):
    return 'RUNX.SKIP' in hparams and hparams['RUNX.SKIP']


def get_code_ignore_patterns(experiment):
    if 'CODE_IGNORE_PATTERNS' in experiment:
        code_ignore_patterns = experiment['CODE_IGNORE_PATTERNS']
    else:
        code_ignore_patterns = '.git,*.pyc,docs*,test*'

    code_ignore_patterns += ',*.pth' # don't copy checkpoints
    code_ignore_patterns = code_ignore_patterns.split(',')
    return code_ignore_patterns


def run_yaml(experiment, runroot):
    """
    Run an experiment, expand hparams
    """
    resources = get_field(experiment, 'RESOURCES')
    code_ignore_patterns = get_code_ignore_patterns(experiment)
    ngc_batch = 'ngc' in cfg.FARM and not args.interactive
    experiment_cmd = experiment['CMD']

    # Build the args that the submit_cmd will see
    yaml_hparams = OrderedDict()

    # Add yaml_hparams
    for k, v in experiment['HPARAMS'].items():
        yaml_hparams[k] = v

    num_trials = experiment.get('NUM_TRIALS', 0)
    num_replications = experiment.get('NUM_REPLICAS', 1)

    # Calculate cross-product of hyperparams
    expanded_hparams = enumerate_hparams(yaml_hparams, num_trials)
    num_cases = len(expanded_hparams)

    # Run each permutation
    for hparam_vals in expanded_hparams:
        g_job_name, g_logdir, coolname, expdir = make_cool_names()

        for replica in range(num_replications):
            if num_replications > 1:
                job_name = f'{g_job_name}_run_{replica}'
                logdir = f'{g_logdir}/run_{replica}'
            else:
                job_name = g_job_name
                logdir = g_logdir

            hparam_vals = list(hparam_vals)
            hparam_keys = list(yaml_hparams.keys())

            # hparams to use for experiment
            hparams = {k: v for k, v in zip(hparam_keys, hparam_vals)}
            if skip_run(hparams):
                continue
            get_tag(hparams)

            resource_copy = resources.copy()
            """
            A few different modes of operation:
            1. interactive runs
            a. copy local code to logdir under LOGROOT
            b. cd to logdir, execute cmd

            2. farm submission: non-NGC
            In this regime, the LOGROOT is expected to be visible to the farm's compute nodes
            a. copy local code to logdir under LOGROOT
            b. call cmd, which should invoke whatever you have specified for SUBMIT_JOB

            3. farm submission: NGC
            a. copy local code to logdir under LOGROOT
            b. ngc workspace upload the logdir to NGC_WORKSPACE
            c. call cmd, which should invoke SUBMIT_JOB==`ngc batch run`
            """
            if ngc_batch:
                ngc_logdir = logdir.replace(cfg.LOGROOT, cfg.NGC_LOGROOT)
                cmd, hparams_out = hacky_substitutions(
                    experiment_cmd, hparams, resource_copy, ngc_logdir, runroot, replica)
                cmd = construct_cmd(cmd, hparams, ngc_logdir)
            else:
                cmd, hparams_out = hacky_substitutions(
                    experiment_cmd, hparams, resource_copy, logdir, runroot, replica)
                cmd = construct_cmd(cmd, hparams, logdir)

            if not args.interactive:
                cmd = build_farm_cmd(cmd, job_name, resource_copy, logdir)

            if args.no_run:
                print(cmd)
                continue

            # copy code to NFS-mounted share
            copy_code(logdir, runroot, code_ignore_patterns)
            if cfg.YML_PARAMS:
                write_yml_params(logdir, hparams)

            # save some meta-data from run
            save_cmd(cmd, logdir)

            # upload to remote farm
            if ngc_batch:
                upload_to_ngc(logdir)

            subprocess.call(['chmod', '-R', 'a+rw', expdir])
            os.chdir(logdir)

            if args.interactive:
                print('Running job {}'.format(job_name))
            else:
                print('Submitting job {}'.format(job_name))
            exec_cmd(cmd)


def run_experiment(exp_fn):
    """
    Run an experiment, given a global config file + an experiment file.
    The global config sets defaults that are inherited by the experiment.
    """
    experiment = read_config(args.farm, args.exp_yml, args.yml_params)

    assert 'HPARAMS' in experiment, 'experiment file is missing hparams'

    # Iterate over hparams if it's a list
    runroot = os.getcwd()
    if isinstance(experiment['HPARAMS'], (list, tuple)):
        # Support inheritance from the first hparams item in list
        first_hparams = experiment['HPARAMS'][0].copy()

        for hparams_set in experiment['HPARAMS']:
            hparams = first_hparams.copy()
            # Inheritance = first hparam set, updated with current hparam set
            hparams.update(hparams_set)

            # create a clean copy of the experiment and fill in hparams
            experiment_copy = experiment.copy()
            experiment_copy['HPARAMS'] = hparams

            run_yaml(experiment_copy, runroot)
    else:
        run_yaml(experiment, runroot)


def main():
    if os.path.exists(args.exp_yml):
        run_experiment(args.exp_yml)
    else:
        print('couldn\'t find experiment file {}'.format(args.exp_yml))
        sys.exit()


if __name__ == '__main__':
    main()

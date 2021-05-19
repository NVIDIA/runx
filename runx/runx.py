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

import os
import re
import sys
import subprocess
import argparse
import itertools

from .utils import read_config, exec_cmd, get_cfg
from .farm import build_farm_cmd, upload_to_ngc


parser = argparse.ArgumentParser(description='Experiment runner')
parser.add_argument('exp_yml', type=str,
                    help='Experiment yaml file')
parser.add_argument('--exp_name', type=str, default=None,
                    help=('Override the *experiment* name, which normally is '
                          'taken from the experiment yaml filename.'))
parser.add_argument('--tag', type=str, default=None,
                    help=('Add a string to the generated *run* name for '
                          ' identification.'))
parser.add_argument('--no_cooldir', action='store_true',
                    help=('For the *run* name, don\'t auto-generate a '
                          'coolname or datestring, only use the tag'))
parser.add_argument('--no_run', '-n', action='store_true',
                    help='Don\'t run, just display the command.')
parser.add_argument('--interactive', '-i', action='store_true',
                    help='Run interactively instead of submitting to farm.')
parser.add_argument('--farm', type=str, default=None,
                    help='Select farm for workstation submission')
parser.add_argument('--config_file', '-c', type=str, default=None,
                    help='Use this file instead of .runx')
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
    # First, add hyperparameters
    cmd += ' ' + expand_hparams(hparams)

    # Expand PYTHONPATH, if necessary
    if get_cfg('PYTHONPATH') is not None:
        pythonpath = get_cfg('PYTHONPATH')
        pythonpath = pythonpath.replace('LOGDIR', logdir)
    else:
        pythonpath = f'{logdir}/code'

    # For signalling reasons, we have to insert the exec here when using submit_job.
    # Nvidia-internal thing.
    exec_str = ''
    if 'submit_job' in get_cfg('SUBMIT_CMD'):
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


def cross_product_hparams(hparams):
    """
    This function takes in just the hyperparameters for the target script,
    such as your main.py.

    inputs:
      hparams is a dict, where each key is the name of a commandline arg and
      the value is the target value of the arg.

      However any arg can also be a list and so this function will calculate
      the cross product for all combinations of all args.

    output:
      The return value is a sequence of lists. Each list is one of the
      permutations of argument values.
    """
    hparam_values = []

    # turn every hyperparam into a list, to prep for itertools.product
    for elem in hparams.values():
        if islist(elem):
            hparam_values.append(elem)
        else:
            hparam_values.append([elem])

    expanded_hparams = itertools.product(*hparam_values)

    # have to do this in order to know length
    expanded_hparams, dup_expanded = itertools.tee(expanded_hparams, 2)
    expanded_hparams = list(expanded_hparams)
    num_cases = len(list(dup_expanded))

    return expanded_hparams, num_cases


def get_field(adict, f, required=True):
    if required:
        assert f in adict, 'expected {} to be defined in experiment'.format(f)
    return adict[f] if f in adict else None


def do_keyword_expansion(alist, pairs):
    """
    Substitute a string in place of certain keywords
    """
    if type(alist) is list or type(alist) is tuple:
        for i, v in enumerate(alist):
            if type(v) == str:
                for k, v in pairs:
                    alist[i] = alist[i].replace(k, v)
    elif type(alist) is dict:
        for a_k, a_v in alist.items():
            if type(a_v) == str:
                for k, v in pairs:
                    alist[a_k] = alist[a_k].replace(k, v)
    else:
        raise


def make_cool_names():
    tagname = args.tag + '_' if args.tag else ''
    datestr = datetime.now().strftime("_%Y.%m.%d_%H.%M")
    if args.no_cooldir:
        coolname = tagname
    else:
        coolname = tagname + generate_slug(2) + datestr

    # Experiment directory is the parent of N runs
    expdir = os.path.join(get_cfg('LOGROOT'), get_cfg('EXP_NAME'))

    # Each run has a logdir
    logdir_name = coolname
    logdir = os.path.join(expdir, logdir_name)

    # Jobname is a unique name for the batch job
    job_name = '{}_{}'.format(get_cfg('EXP_NAME'), coolname)
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


def hacky_substitutions(hparams, resource_copy, logdir, runroot):
    # Substitute the true logdir in for the magic variable LOGDIR
    do_keyword_expansion(hparams, [('LOGDIR', logdir)])
    do_keyword_expansion(resource_copy, [('LOGDIR', logdir)])

    # Build hparams to save out after LOGDIR but before deleting
    # the key 'SUBMIT_JOB.NODES', so that it is part of the hparams saved
    # This is done so that we can see the node count in sumx.
    # hparams_out = hparams.copy()

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
    # hparams_out['srcdir'] = runroot

    # return hparams_out


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
    ngc_batch = 'ngc' in get_cfg('FARM') and not args.interactive
    experiment_cmd = experiment['CMD']

    # Build the args that the submit_cmd will see
    yaml_hparams = OrderedDict()

    # Add yaml_hparams
    for k, v in experiment['HPARAMS'].items():
        yaml_hparams[k] = v

    # Calculate cross-product of hyperparams
    expanded_hparams, num_cases = cross_product_hparams(yaml_hparams)

    # Run each permutation
    for i, hparam_vals in enumerate(expanded_hparams):
        hparam_vals = list(hparam_vals)
        hparam_keys = list(yaml_hparams.keys())

        # hparams to use for experiment
        hparams = {k: v for k, v in zip(hparam_keys, hparam_vals)}
        if skip_run(hparams):
            continue
        get_tag(hparams)

        job_name, logdir, coolname, expdir = make_cool_names()
        resource_copy = resources.copy()

        """
        A few different modes of operation:
        1. interactive runs
           a. copy local code to logdir under LOGROOT
           b. cd to logdir, execute cmd

        2. farm submission: non-NGC
           In this regime, the LOGROOT is expected to be visible to the farm's
           compute nodes
           a. copy local code to logdir under LOGROOT
           b. call cmd, which should invoke whatever you have specified for
              SUBMIT_JOB

        3. farm submission: NGC
           a. copy local code to logdir under LOGROOT
           b. ngc workspace upload the logdir to NGC_WORKSPACE
           c. call cmd, which should invoke SUBMIT_JOB==`ngc batch run`
        """
        if ngc_batch:
            ngc_logdir = logdir.replace(get_cfg('LOGROOT'),
                                        get_cfg('NGC_LOGROOT'))
            hacky_substitutions(
                hparams, resource_copy, ngc_logdir, runroot)
            cmd = construct_cmd(experiment_cmd, hparams, ngc_logdir)
        else:
            hacky_substitutions(
                hparams, resource_copy, logdir, runroot)
            cmd = construct_cmd(experiment_cmd, hparams, logdir)

        if not args.interactive:
            cmd = build_farm_cmd(cmd, job_name, resource_copy, logdir)

        if args.no_run:
            print(cmd)
            continue

        # copy code to NFS-mounted share
        copy_code(logdir, runroot, code_ignore_patterns)

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
    experiment = read_config(args)

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

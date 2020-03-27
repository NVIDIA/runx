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
import yaml
import argparse
import itertools

from .utils import read_config, save_hparams


parser = argparse.ArgumentParser(description='Experiment runner')
parser.add_argument('exp_yml', type=str, help='experiment yaml file')
parser.add_argument('--tag', type=str, default=None, help='tag label for run')
parser.add_argument('--no_run', '-n', action='store_true', help='don\'t run')
parser.add_argument('--no_cooldir', action='store_true',
                    help='no coolname, no datestring')
parser.add_argument('--farm', type=str, default=None, help=(
    'Select farm for workstation submission'))
args = parser.parse_args()


def expand_resources(resources):
    """
    Construct the submit_job arguments from the resource dict
    """
    cmd = ''
    for field, val in resources.items():
        if type(val) is bool:
            if val is True:
                cmd += '--{} '.format(field)
        elif type(val) is list or type(val) is tuple:
            for mp in val:
                cmd += '--{} {} '.format(field, mp)
        else:
            cmd += '--{} {} '.format(field, val)
    return cmd


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
    cmd += '\''
    return cmd


def exec_cmd(cmd):
    """
    Execute a command and print stderr/stdout to the console
    """
    result = subprocess.run(cmd, stderr=subprocess.PIPE, shell=True)
    if result.stderr:
        message = result.stderr.decode("utf-8")
        print(message)


def construct_cmd(cmd, hparams, resources, job_name, logdir):
    """
    Expand the hyperparams into a commandline
    """
    ######################################################################
    # You may wish to customize this function for your own needs
    ######################################################################
    if os.environ['NVIDIA_INTERNAL']:
        cmd += '--name {} '.format(job_name)
        if 'submit_job' in cmd:
            cmd += '--cd_to_logdir '
            cmd += '--logdir {}/logs '.format(logdir)

    cmd += expand_resources(resources)
    cmd += expand_hparams(hparams)

    if args.no_run:
        print(cmd)

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


def make_cool_names(exp_name, logroot):
    tagname = args.tag + '_' if args.tag else ''
    datestr = datetime.now().strftime("_%Y.%m.%d_%H.%M")
    if args.no_cooldir:
        coolname = tagname
    else:
        coolname = tagname + generate_slug(2) + datestr

    # Experiment directory is the parent of N runs
    expdir = os.path.join(logroot, exp_name)

    # Each run has a logdir
    logdir_name = coolname
    logdir = os.path.join(expdir, logdir_name)

    # Jobname is a unique name for the batch job
    job_name = '{}_{}'.format(exp_name, coolname)
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
    hparams_out = hparams.copy()

    # SUBMIT_JOB.NODES is a hyperparmeter that sets the node count
    # This is actually a resource, so when we find this arg, we delete
    # it from the list of hyperparams that the training script sees.
    if 'SUBMIT_JOB.NODES' in hparams:
        resource_copy['nodes'] = hparams['SUBMIT_JOB.NODES']
        del hparams['SUBMIT_JOB.NODES']

    # Record the directory from whence the experiments were launched
    hparams_out['srcdir'] = runroot

    return hparams_out


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


def get_code_ignore_patterns(experiment):
    if 'CODE_IGNORE_PATTERNS' in experiment:
        code_ignore_patterns = experiment['CODE_IGNORE_PATTERNS']
    else:
        return ['.git', '*.pyc', 'docs*', 'test*']

    code_ignore_patterns = code_ignore_patterns.split(',')
    return code_ignore_patterns


def run_yaml(experiment, exp_name, runroot):
    """
    Run an experiment, expand hparams
    """
    resources = get_field(experiment, 'RESOURCES')
    submit_cmd = get_field(experiment, 'SUBMIT_CMD') + ' '
    logroot = get_field(experiment, 'LOGROOT')
    code_ignore_patterns = get_code_ignore_patterns(experiment)

    # Build the args that the submit_cmd will see
    yaml_hparams = OrderedDict()
    yaml_hparams['command'] = '\'{}'.format(experiment['CMD'])

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
        get_tag(hparams)

        job_name, logdir, coolname, expdir = make_cool_names(exp_name, logroot)
        resource_copy = resources.copy()
        hparams_out = hacky_substitutions(hparams, resource_copy, logdir,
                                          runroot)
        cmd = construct_cmd(submit_cmd, hparams,
                            resource_copy, job_name, logdir)

        if not args.no_run:
            # copy code to NFS-mounted share
            copy_code(logdir, runroot, code_ignore_patterns)

            # save some meta-data from run
            save_cmd(cmd, logdir)
            save_hparams(hparams_out, logdir)

            subprocess.call(['chmod', '-R', 'a+rw', expdir])
            os.chdir(logdir)

            print('Submitting job {}'.format(job_name))
            exec_cmd(cmd)


def run_experiment(exp_fn):
    """
    Run an experiment, given a global config file + an experiment file.
    The global config sets defaults that are inherited by the experiment.
    """
    global_config = read_config(args.farm)
    exp_config = yaml.load(open(args.exp_yml), Loader=yaml.FullLoader)

    # Inherit global config into experiment config:
    experiment = global_config
    for k, v in exp_config.items():
        experiment[k] = v

    exp_name = os.path.splitext(os.path.basename(args.exp_yml))[0]

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

            run_yaml(experiment_copy, exp_name, runroot)
    else:
        run_yaml(experiment, exp_name, runroot)


def main():
    if os.path.exists(args.exp_yml):
        run_experiment(args.exp_yml)
    else:
        print('couldn\'t find experiment file {}'.format(args.exp_yml))
        sys.exit()


if __name__ == '__main__':
    main()

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
import os
import yaml
import shlex
import json
from warnings import warn

import subprocess
from subprocess import call, getoutput, DEVNULL
from .config import cfg


def exec_cmd(cmd):
    """
    Execute a command and print stderr/stdout to the console
    """
    print(cmd)
    result = subprocess.run(cmd, stderr=subprocess.PIPE, shell=True)
    if result.stderr:
        message = result.stderr.decode("utf-8")
        print(message)


trn_names = ('trn', 'train', 'training')
val_names = ('val', 'validate', 'validation', 'test')


def get_cfg(key):
    if key in cfg:
        return cfg[key]
    else:
        warn(f'Please define {key} in .runx or your experiment file')
        return None


def read_config_item(config, key, optional=True):
    if key in config:
        return config[key]
    elif optional:
        return None
    else:
        raise f'can\'t find {key} in config'


def read_config_file(args=None):
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


def read_config(args):
    '''
    Merge the global and experiment config files into a single config

    Need to support the case where there is no FARM defined, which should
    be fine for interactive jobs.
    '''
    global_config = read_config_file(args)

    if hasattr(args, 'farm') and args.farm is not None:
        global_config['FARM'] = args.farm

    # Support the case of no farm
    if 'FARM' not in global_config or \
       global_config['FARM'] not in global_config:
        no_farm = True
    else:
        no_farm = False
        farm_name = read_config_item(global_config, 'FARM')
        # Dereference the farm config items
        for k, v in global_config[farm_name].items():
            global_config[k] = v

    # Inherit global config into experiment config:
    experiment = global_config

    # Merge experiment settings into the global configuration.
    # This allows an experiment yaml to override the settings in the .runx
    if hasattr(args, 'exp_yml'):
        exp_config = yaml.load(open(args.exp_yml), Loader=yaml.SafeLoader)

        for k, v in exp_config.items():
            if k in experiment:
                experiment.update(v)
            else:
                experiment[k] = v

        if args.exp_name is not None:
            cfg.EXP_NAME = args.exp_name
        elif args.exp_yml is None:
            cfg.EXP_NAME = 'none'
        else:
            cfg.EXP_NAME = os.path.splitext(os.path.basename(args.exp_yml))[0]

    if no_farm:
        cfg.FARM = 'NOFARM'
        cfg.LOGROOT = read_config_item(experiment, 'LOGROOT')
        cfg.SUBMIT_CMD = 'echo'
        cfg.PYTHONPATH = ''
        # have to define resources because runx expects it
        global_config['RESOURCES'] = {}
    else:
        cfg.FARM = read_config_item(experiment, 'FARM')
        cfg.LOGROOT = read_config_item(experiment, 'LOGROOT')
        cfg.SUBMIT_CMD = read_config_item(experiment, 'SUBMIT_CMD')
        cfg.PYTHONPATH = read_config_item(experiment, 'PYTHONPATH')
        if 'ngc' in cfg.FARM:
            cfg.NGC_LOGROOT = read_config_item(experiment, 'NGC_LOGROOT')
            cfg.WORKSPACE = read_config_item(experiment, 'WORKSPACE')

    return experiment


def get_logroot():
    global_config = read_config_file()
    return read_config_item(global_config, 'LOGROOT')


def get_bigfiles(root):
    output = getoutput('find {} -size +100k'.format(root))
    if len(output):
        bigfiles = output.split('\n')
        return bigfiles
    else:
        return []


def save_code(logdir, coderoot):
    zip_outfile = os.path.join(logdir, 'code.tgz')

    # skip over non-sourcecode items
    exclude_list = ['*.pth', '*.jpg', '*.jpeg', '*.pyc', '*.so', '*.o',
                    '*.git', '__pycache__', '*~']
    bigfiles = get_bigfiles(coderoot)
    exclude_str = ''
    for ex in exclude_list + bigfiles:
        exclude_str += ' --exclude=\'{}\''.format(ex)

    cmd = 'tar -czvf {} {} {}'.format(zip_outfile, exclude_str, coderoot)
    call(shlex.split(cmd), stdout=DEVNULL, stderr=DEVNULL)


def save_hparams(hparams, logdir):
    """
    Save hyperparameters into a json file
    """
    json_fn = os.path.join(logdir, 'hparams.json')

    if os.path.isfile(json_fn):
        return

    with open(json_fn, 'w') as outfile:
        json.dump(hparams, outfile, indent=4)


class _CallableProxy:
    def __init__(self, real_callable, post_hook=None):
        self.real_callable = real_callable
        self.post_hook = post_hook

    def __call__(self, *args, **kwargs):
        ret_val = self.real_callable(*args, **kwargs)

        if self.post_hook is not None:
            self.post_hook()

        return ret_val


class ConditionalProxy:
    """
    This object can be used to serve as a proxy on an object where we want to
    forward all function calls along to the dependent object, but only when
    some condition is true. For example, the primary use case for this object
    is to deal with that fact that in a distributed training job, we only want
    to manage artifacts (checkpoints, logs, TB summaries) on the rank-0
    process.

    So, let's say that we have this class:
    ```
    class Foo:
        def bar(self, val):
            pass

        def baz(self, val1, val2):
            pass
    ```

    and we wrap it with this object:
    ```
    proxy = ConditionalProxy(Foo(), rank == 0)
    proxy.bar(42)
    proxy.baz(10, 20)
    proxy.some_other_function('darn it')  # Throws an exception because `Foo`
                                          # doesn't have an implementation for
                                          # this.
    ```

    In this case, if `rank == 0`, then we will end up calling `Foo.bar` and
    `Foo.baz`.
    If `rank != 0`, then the calls will be ignored.

    In addition to the basic usage, you can also add a `post_hook` to the
    proxy, which is a callable that takes no arguments. The proxy will call
    that function after each function call made through the proxy, but only
    when `condition == True`.
    """

    def __init__(self, real_object, condition, post_hook=None):
        self.real_object = real_object
        self.condition = condition
        self.post_hook = post_hook

    @staticmethod
    def _throw_away(*args, **kwargs):
        pass

    def __getattr__(self, name):
        if not self.condition:
            # When `self.condition == False`, then we want to return a function
            # that can take any form of arguments, and does nothing. This works
            # under the assumption that the only API interface for the
            # dependent object is function, e.g. this would be awkward if the
            # caller was trying to access a member variable.
            return ConditionalProxy._throw_away

        real_fn = getattr(self.real_object, name)

        # Wrap the return function in a `_CallableProxy` so that we can
        # invoke the `self.post_hook`, if specified, after the real function
        # executes.
        return _CallableProxy(real_fn, self.post_hook)

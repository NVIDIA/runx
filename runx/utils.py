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
import socket

from subprocess import call, getoutput, DEVNULL


trn_names = ('trn', 'train', 'training')
val_names = ('val', 'validate', 'validation', 'test')


def read_item(config, key):
    assert key in config, 'couldn\'t find {} in config'.format(key)
    return config[key]


def read_global_config():
    home = os.path.expanduser('~')
    cwd_config_fn = './.runx'
    home_config_fn = '{}/.config/runx.yml'.format(home)
    if os.path.isfile(cwd_config_fn):
        config_fn = cwd_config_fn
    elif os.path.exists(home_config_fn):
        config_fn = home_config_fn
    else:
        raise('can\'t find file ./.runx or ~/.config/runx.yml config files')
    if 'FullLoader' in dir(yaml):
        global_config = yaml.load(open(config_fn), Loader=yaml.FullLoader)
    else:
        global_config = yaml.load(open(config_fn))
    return global_config


def read_config(args_farm):
    '''
    read the global config
    pull the farm portion and merge with global config
    '''
    global_config = read_global_config()

    merged_config = {}
    merged_config['LOGROOT'] = read_item(global_config, 'LOGROOT')
    if 'CODE_IGNORE_PATTERNS' in global_config:
        merged_config['CODE_IGNORE_PATTERNS'] = \
            global_config['CODE_IGNORE_PATTERNS']

    if args_farm is not None:
        farm_name = args_farm
    else:
        assert 'FARM' in global_config, \
            'FARM not defined in .runx'
        farm_name = global_config['FARM']

    assert farm_name in global_config, \
        f'{farm_name} not found in .runx'

    for k, v in global_config[farm_name].items():
        merged_config[k] = v

    return merged_config


def get_logroot():
    global_config = read_global_config()
    return read_item(global_config, 'LOGROOT')


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

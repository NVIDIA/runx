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


Routines to build farm submission commands.
"""
import os

from .config import cfg
from .utils import exec_cmd


def expand_resources(resources):
    """
    Construct the submit_job arguments from the resource dict.
    
    In general, a k,v from the dict turns into an argument '--k v'.
    If the value is a boolean, then the argument turns into a flag.
    If the value is a list/tuple, then multiple '--k v' are presented,
    one for each list item.

    :resources: a dict of arguments for the farm submission command.
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


def build_draco(train_cmd, job_name, resources, logdir):
    """
    For using Draco, only for NVIDIA-ADLR folks

    See build_farm_cmd for arg description
    """
    assert 'submit_job' in cfg.SUBMIT_CMD, \
        'Expected \'submit_job\' as SUBMIT_CMD. Exiting ...'
    submit_cmd = cfg.SUBMIT_CMD + ' '
    submit_cmd += expand_resources(resources)
    submit_cmd += f' --name {job_name}'
    submit_cmd += f' --command \' {train_cmd} \''
    submit_cmd += f' --logdir {logdir}/gcf_log'
    return submit_cmd


def build_ngc_generic(train_cmd, job_name, resources, logdir):
    """
    Compose the farm submission command for generic NGC users, folks
    both inside and outside of Nvidia.

    The SUBMIT_CMD should be 'ngc batch run'.

    See build_farm_cmd for arg description
    """
    assert cfg.SUBMIT_CMD == 'ngc batch run', \
        'Expected SUBMIT_CMD to be \'ngc batch run\'. Exiting ...'
    submit_cmd = cfg.SUBMIT_CMD + ' '
    submit_cmd += expand_resources(resources)
    submit_cmd += f' --name {job_name}'
    submit_cmd += f' --commandline \' {train_cmd} \''
    submit_cmd += f' --workspace {cfg.WORKSPACE}:{cfg.NGC_LOGROOT}:RW'
    return submit_cmd


def build_ngc(train_cmd, job_name, resources, logdir):
    """
    For using NGC with submit_job, only for NVIDIA-ADLR folks.

    See build_farm_cmd for arg description
    """
    if 'submit_job' in cfg.SUBMIT_CMD:
        ngc_logdir = logdir.replace(cfg.LOGROOT, cfg.NGC_LOGROOT)
        return build_draco(train_cmd, job_name, resources, ngc_logdir)
    else:
        return build_ngc_generic(train_cmd, job_name, resources, logdir)


def build_generic(train_cmd, job_name, resources, logdir):
    """
    Generic farm support

    See build_farm_cmd for arg description
    """
    if 'submit_job' in cfg.SUBMIT_CMD:
        ngc_logdir = logdir.replace(cfg.LOGROOT, cfg.NGC_LOGROOT)
        return build_draco(train_cmd, job_name, resources, ngc_logdir)
    else:
        return build_ngc_generic(train_cmd, job_name, resources, logdir)


def build_farm_cmd(train_cmd, job_name, resources, logdir):
    """
    This function builds a farm submission command.

    :train_cmd: full training command
    :job_name: unique job_name, to be used for tracking
    :resources: farm submission command args, pulled from .runx
    :logdir: target log directory
    """

    if 'ngc' in cfg.FARM:
        return build_ngc(train_cmd, job_name, resources, logdir)
    elif 'draco' in cfg.FARM:
        return build_draco(train_cmd, job_name, resources, logdir)
    else:
        raise f'Unsupported farm: {cfg.FARM}'


def upload_to_ngc(staging_logdir):
    """
    Upload single run's code to NGC workspace.

    Within the job, the workspace will be mounted at: <NGC_LOGROOT>.
    The full path of the logdir in the job is: <NGC_LOGROOT>/<exp_name>/<run_name>

    :staging_logdir: path to the staging logdir, on the client machine
    """
    fields = staging_logdir.split('/')
    exp_name = fields[-2]
    run_name = fields[-1]

    ngc_workspace = cfg.WORKSPACE
    target_dir = os.path.join(exp_name, run_name)
    msg = 'Uploading experiment to {} in workpace {} ...'
    print(msg.format(target_dir, ngc_workspace))
    cmd = ('ngc workspace upload --source {staging_logdir} '
           '--destination {target_dir} {workspace}')
    cmd = cmd.format(staging_logdir=staging_logdir, target_dir=target_dir,
                     workspace=ngc_workspace)
    exec_cmd(cmd)

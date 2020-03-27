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
from collections import defaultdict
from contextlib import contextmanager
from shutil import copyfile

import csv
import os
import re
import shlex
import subprocess
import time


try:
    from torch.utils.tensorboard import SummaryWriter
except ModuleNotFoundError:
    from tensorboardX import SummaryWriter

import torch

try:
    from .utils import (get_logroot, save_hparams, trn_names, val_names,
                        ConditionalProxy)
except ImportError:
    # This is to allow the unit tests to run properly
    from utils import (get_logroot, save_hparams, trn_names, val_names,
                       ConditionalProxy)


def is_list(x):
    return isinstance(x, (list, tuple))


def get_gpu_utilization_pct():
    '''
    Use nvidia-smi to capture the GPU utilization, which is reported as an
    integer in range 0-100.
    '''
    util = subprocess.check_output(
        shlex.split('nvidia-smi --query-gpu="utilization.gpu" '
                    '--format=csv,noheader,nounits -i 0'))
    util = util.decode('utf-8')
    util = util.replace('\n', '')
    return int(util)


class LogX(object):
    def __init__(self, rank=0):
        self.initialized = False

    def initialize(self, logdir=None, coolname=False, hparams=None,
                   tensorboard=False, no_timestamp=False, global_rank=0,
                   eager_flush=True):
        '''
        Initialize logx

        inputs
        - logdir - where to write logfiles
        - tensorboard - whether to write to tensorboard file
        - global_rank - must set this if using distributed training, so we only
          log from rank 0
        - coolname - generate a unique directory name underneath logdir, else
          use logdir as output directory
        - hparams - only use if not launching jobs with runx, which also saves
          the hparams.
        - eager_flush - call `flush` after every tensorboard write
        '''
        self.rank0 = (global_rank == 0)
        self.initialized = True

        if logdir is not None:
            self.logdir = logdir
        else:
            logroot = get_logroot()
            if coolname:
                from coolname import generate_slug
                self.logdir = os.path.join(logroot, generate_slug(2))
            else:
                self.logdir = os.path.join(logroot, 'default')

        # confirm target log directory exists
        if not os.path.isdir(self.logdir):
            os.makedirs(self.logdir, exist_ok=True)

        if hparams is not None:
            save_hparams(hparams, self.logdir)

        # Tensorboard file
        if self.rank0 and tensorboard:
            self.tb_writer = SummaryWriter(log_dir=self.logdir,
                                           flush_secs=1)
        else:
            self.tb_writer = None

        self.eager_flush = eager_flush

        # This allows us to use the tensorboard with automatic checking of both
        # the `tensorboard` condition, as well as ensuring writes only happen
        # on rank0. Any function supported by `SummaryWriter` is supported by
        # `ConditionalProxy`. Additionally, flush will be called after any call
        # to this.
        self.tensorboard = ConditionalProxy(
            self.tb_writer,
            tensorboard and self.rank0,
            post_hook=self._flush_tensorboard,
        )

        if not self.rank0:
            return

        # Metrics file
        metrics_fn = os.path.join(self.logdir, 'metrics.csv')
        self.metrics_fp = open(metrics_fn, mode='a+')
        self.metrics_writer = csv.writer(self.metrics_fp, delimiter=',')

        # Log file
        log_fn = os.path.join(self.logdir, 'logging.log')
        self.log_file = open(log_fn, mode='a+')

        # save metric
        self.save_metric = None
        self.best_metric = None
        self.save_ckpt_fn = ''
        # Find the existing best checkpoint, and update `best_metric`,
        # if available
        self.best_ckpt_fn = self.get_best_checkpoint() or ''
        if self.best_ckpt_fn:
            best_chk = torch.load(self.best_ckpt_fn, map_location='cpu')
            self.best_metric = best_chk.get('__metric', None)
        self.epoch = defaultdict(lambda: 0)
        self.no_timestamp = no_timestamp

        # Initial timestamp, so that epoch time calculation is correct
        phase = 'start'
        csv_line = [phase]

        # add epoch/iter
        csv_line.append('{}/step'.format(phase))
        csv_line.append(0)

        # add timestamp
        if not self.no_timestamp:
            # this feature is useful for testing
            csv_line.append('timestamp')
            csv_line.append(time.time())

        self.metrics_writer.writerow(csv_line)
        self.metrics_fp.flush()

    def __del__(self):
        if self.initialized and self.rank0:
            self.metrics_fp.close()
            self.log_file.close()

    def msg(self, msg):
        '''
        Print out message to std and to a logfile
        '''
        if not self.rank0:
            return

        print(msg)
        self.log_file.write(msg + '\n')
        self.log_file.flush()

    def add_image(self, path, img, step=None):
        '''
        Write an image to the tensorboard file
        '''
        self.tensorboard.add_image(path, img, step)

    def add_scalar(self, name, val, idx):
        '''
        Write a scalar to the tensorboard file
        '''
        self.tensorboard.add_scalar(name, val, idx)

    def _flush_tensorboard(self):
        if self.eager_flush and self.tb_writer is not None:
            self.tb_writer.flush()

    @contextmanager
    def suspend_flush(self, flush_at_end=True):
        prev_flush = self.eager_flush
        self.eager_flush = False
        yield
        self.eager_flush = prev_flush
        if flush_at_end:
            self._flush_tensorboard()

    def metric(self, phase, metrics, epoch=None):
        """Record train/val metrics. This serves the dual-purpose to write these
        metrics to both a tensorboard file and a csv file, for each parsing by
        sumx.

        Arguments:
            phase: 'train' or 'val'. sumx will only summarize val metrics.
            metrics: dictionary of metrics to record
            global_step: (optional) epoch or iteration number
        """
        if not self.rank0:
            return

        # define canonical phase
        if phase in trn_names:
            canonical_phase = 'train'
        elif phase in val_names:
            canonical_phase = 'val'
        else:
            raise('expected phase to be one of {} {}'.format(str(val_names,
                                                                 trn_names)))

        if epoch is not None:
            self.epoch[canonical_phase] = epoch

        # Record metrics to csv file
        csv_line = [canonical_phase]
        for k, v in metrics.items():
            csv_line.append(k)
            csv_line.append(v)

        # add epoch/iter
        csv_line.append('epoch')
        csv_line.append(self.epoch[canonical_phase])

        # add timestamp
        if not self.no_timestamp:
            # this feature is useful for testing
            csv_line.append('timestamp')
            csv_line.append(time.time())

        # To save a bit of disk space, only save validation metrics
        if canonical_phase == 'val':
            self.metrics_writer.writerow(csv_line)
            self.metrics_fp.flush()

        # Write updates to tensorboard file
        with self.suspend_flush():
            for k, v in metrics.items():
                self.add_scalar('{}/{}'.format(phase, k), v,
                                self.epoch[canonical_phase])

        # if no step, then keep track of it automatically
        if epoch is None:
            self.epoch[canonical_phase] += 1

    @staticmethod
    def is_better(save_metric, best_metric, higher_better):
        return best_metric is None or \
            higher_better and (save_metric > best_metric) or \
            not higher_better and (save_metric < best_metric)

    def save_model(self, save_dict, metric, epoch, higher_better=True,
                   delete_old=True):
        """Saves a model to disk. Keeps a separate copy of latest and best models.

        Arguments:
            save_dict: dictionary to save to checkpoint
            epoch: epoch number, used to name checkpoint
            metric: metric value to be used to evaluate whether this is the
                    best result
            higher_better: True if higher valued metric is better, False
                    otherwise
            delete_old: Delete prior 'lastest' checkpoints. By setting to
                    false, you'll get a checkpoint saved every time this
                    function is called.
        """
        if not self.rank0:
            return

        save_dict['__metric'] = metric

        if os.path.exists(self.save_ckpt_fn) and delete_old:
            os.remove(self.save_ckpt_fn)
        # Save out current model
        self.save_ckpt_fn = os.path.join(
            self.logdir, 'last_checkpoint_ep{}.pth'.format(epoch))
        torch.save(save_dict, self.save_ckpt_fn)
        self.save_metric = metric

        is_better = self.is_better(self.save_metric, self.best_metric,
                                   higher_better)
        if is_better:
            if os.path.exists(self.best_ckpt_fn):
                os.remove(self.best_ckpt_fn)
            self.best_ckpt_fn = os.path.join(
                self.logdir, 'best_checkpoint_ep{}.pth'.format(epoch))
            self.best_metric = self.save_metric
            copyfile(self.save_ckpt_fn, self.best_ckpt_fn)
        return is_better

    def get_best_checkpoint(self):
        """
        Finds the checkpoint in `self.logdir` that is considered best.

        If, for some reason, there are multiple best checkpoint files, then
        the one with the highest epoch will be preferred.

        Returns:
            None - If there is no best checkpoint file
            path (str) - The full path to the best checkpoint otherwise.
        """
        match_str = r'^best_checkpoint_ep([0-9]+).pth$'
        best_epoch = -1
        best_checkpoint = None
        for filename in os.listdir(self.logdir):
            match = re.fullmatch(match_str, filename)
            if match is not None:
                # Extract the epoch number
                epoch = int(match.group(1))
                if epoch > best_epoch:
                    best_epoch = epoch
                    best_checkpoint = filename

        if best_checkpoint is None:
            return None
        return os.path.join(self.logdir, best_checkpoint)

    def load_model(self, path):
        """Restore a model and return a dict with any meta data included in
        the snapshot
        """
        checkpoint = torch.load(path)
        state_dict = checkpoint['state_dict']
        meta = {k: v for k, v in checkpoint.items() if k != 'state_dict'}
        return state_dict, meta


# Importing logx gives you access to this shared object
logx = LogX()

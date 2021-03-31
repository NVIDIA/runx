import argparse
import os
from functools import partial
import subprocess
import signal
import sys

import numpy as np
from hyperopt import Trials, fmin, tpe
import hyperopt
import pickle as pkl
import datetime

import torch
import torch.multiprocessing as mp

from .utils import load_yaml

SPACE_FILE = 'hyperopt_space.pkl'
TRIAL_FILE = 'hyperopt_trial.pkl'

def main():
    parser = argparse.ArgumentParser(description='Wrapper script for invoking bayesian hyper-parameter search')
    parser.add_argument('--hopt_sweep_root', type=str, required=True,
                        help='Path to the root of the hyper-parameter search')
    parser.add_argument('--hopt_exp_root', type=str, required=True,
                        help='Path to the experiment root directory. Trials will be loaded from there.')
    parser.add_argument('--hopt_num_trials', type=int, required=True,
                        help='The maximum number of hyperopt iterations')

    args, rest = parser.parse_args()

    trials = []

    for dirpath, dirs, files in os.walk(args.hopt_sweep_root):
        # Find the experiment directories by looking for the metric file
        if TRIAL_FILE in files:
            with open(os.path.join(dirpath, TRIAL_FILE), 'rb') as fd:
                trial = pkl.load(fd)
                trials.append(trial)

    if len(trials) >= args.hopt_num_trials:
        return

    # Gather all of the disparate trials
    history = Trials()
    for t in trials:
        merge_trials(history, t)
    history.refresh()

    eval_fn = partial(evaluate, rest)

    with open(os.path.join(args.hopt_sweep_root, SPACE_FILE), 'rb') as fd:
        space = pkl.load(fd)

    # Run the next hyper step
    fmin(eval_fn, space=space,
         algo=tpe.suggest,
         trials=history,
         max_evals=len(trials) + 1,
    )

    last_trial = get_last_trial(history)

    with open(os.path.join(args.hopt_exp_root, TRIAL_FILE), 'wb') as fd:
        pkl.dump(last_trial, fd)

    if len(history) < args.hopt_num_trials:
        queue_next_job()


def merge_trials(trials_accum: Trials, other_trials: Trials):
    max_tid = max([trial['tid'] for trial in trials_accum.trials])

    for trial in other_trials:
        tid = trial['tid'] + max_tid + 1
        hyperopt_trial = Trials().new_trial_docs(
                tids=[None],
                specs=[None],
                results=[None],
                miscs=[None])
        hyperopt_trial[0] = trial
        hyperopt_trial[0]['tid'] = tid
        hyperopt_trial[0]['misc']['tid'] = tid
        for key in hyperopt_trial[0]['misc']['idxs'].keys():
            hyperopt_trial[0]['misc']['idxs'][key] = [tid]
        trials_accum.insert_trial_docs(hyperopt_trial)
        trials_accum.refresh()
    return trials_accum


def get_last_trial(history: Trials):
    max_tid = max(history.tids)

    for trial in history:
        if trial['tid'] == max_tid:
            ret = Trials()
            ret.insert_trial_doc(trial)
            return ret
    assert False, "Shouldn't reach this!"


class SignalHandler:
    def __init__(self, child_procs):
        self.child_procs = child_procs

    def __call__(self, incoming_signal, frame):
        print("Signal %d detected in process %d " % ( incoming_signal, os.getpid() ))
        print("Forwarding to children " )
        for child in self.child_procs:
            print("Will now pass the signal %d to child process %d" % ( incoming_signal, child.pid ) )
            os.kill( child.pid, incoming_signal)
        if incoming_signal in [ signal.SIGUSR1,signal.SIGUSR2 ]:
            # This is the most important part - we return silently and will be allowed to keep running.
            return
        else:
            sys.exit(1)


def _set_signal_handlers(child_procs):
    signal_handler = SignalHandler(child_procs)
    print("Setting signal handlers in process %d" % os.getpid())
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGUSR1, signal_handler)
    signal.signal(signal.SIGUSR2, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

def evaluate(command_args, hopt_args):
    pass


def queue_next_job():
    pass

if __name__ == '__main__':
    main()
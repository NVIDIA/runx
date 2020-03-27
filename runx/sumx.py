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
from tabulate import tabulate

import os
import copy
import argparse
import time
import json
import csv

from .utils import get_logroot


parser = argparse.ArgumentParser(
    description='Summarize run results',
    epilog=('Summarize results from training runs. Feed this script the name'
            'of the parent directory of a set of runs. sumx will automatically'
            'find the experiments by recursing downwards. You should only have'
            'to specify the parent because the overall root should come from'
            'logroot, contained in ~/.config/runx.yml'))
parser.add_argument('dirs', nargs='+', type=str)
parser.add_argument('--logroot', type=str, default=None)
parser.add_argument('--ignore', type=str, default=None,
                    help=('comma-separated list of hparams to ignore, default'
                          'logdir,command,result_dir,nbr_workers,mmap_cache'))
parser.add_argument('--sortwith', '-s', type=str, default=None,
                    help='sort based on this metrics field')
parser.add_argument('--csv', type=str, default=None,
                    help='Dump cvs file of results')

args = parser.parse_args()

if args.ignore:
    args.ignore = args.ignore.split(',')
else:
    args.ignore = []
args.ignore += ['logdir', 'command', 'result_dir', 'nbr_workers', 'paths',
                'val_paths']


def load_json(fname):
    with open(fname) as json_data:
        adict = json.load(json_data)
    return adict


def get_runs(parent_dir):
    '''
    Assemble list of full paths to runs underneath parent.
    Can be any depth of hierarchical tree
    Look for code.tgz file.
    '''
    runs = []
    for adir in os.listdir(parent_dir):
        run_dir = os.path.join(parent_dir, adir)
        hparams_fn = os.path.join(run_dir, 'hparams.json')
        if os.path.isfile(hparams_fn):
            runs += [run_dir]

    return runs


def get_hparams(runs):
    '''
    given a list of full paths to directories, read in all hparams
    '''
    hparams = {}
    for run in runs:
        json_fn = os.path.join(run, 'hparams.json')
        assert os.path.isfile(json_fn), \
            'hparams.json not found in {}'.format(run)
        hparams[run] = load_json(json_fn)
    return hparams


def load_csv(csv_fn):
    fp = open(csv_fn)
    csv_reader = csv.reader(fp, delimiter=',')
    return list(csv_reader)


def avg_time_util(metrics_fn):
    '''
    read in a metrics file
    calculate average: epoch time, gpu utilization
    '''

    metrics = load_csv(metrics_fn)
    val_lines = [l for l in metrics if 'val' in l]

    if not len(val_lines) or 'timestamp' not in val_lines[0]:
        return None

    if len(val_lines) == 1:
        return val_lines[0]['timestamp']

    for metric_line in metrics:
        phase = metric_line[0]
        metric_line = metric_line[1:]
        if phase == 'val':
            keys = metric_line[0::2]  # evens
            vals = metric_line[1::2]  # odds
            metric_dict = {k: v for k, v in zip(keys, vals)}
            return metric_dict


def extract_nontime_metrics(m):
    """
    Read latest metrics out of metrics file.

    if args.sortwith is defined, also capture the best value for the
    args.sortwith metric and add that into the dict returned
    """
    metrics = copy.deepcopy(m)
    metrics.reverse()

    skip_metrics = ('timestamp', 'gpu util')

    epochs = 0
    metric_dict = {}
    saw_final_metrics = False
    best_sortwith = None

    for metric_line in metrics:
        phase = metric_line[0]
        metric_line = metric_line[1:]
        if phase == 'val':
            keys = metric_line[0::2]  # evens
            vals = metric_line[1::2]  # odds
            this_line_metrics = dict(zip(keys, vals))

            # Capture the final validation metrics
            if not saw_final_metrics:
                saw_final_metrics = True
                for k, v in this_line_metrics.items():
                    if k not in skip_metrics:
                        metric_dict[k] = v
                    # make the assumption that validation step == epoch
                    if k == 'step' or k == 'epoch':
                        epochs = int(v)

            # Update the best value for sortwith
            if args.sortwith:
                assert args.sortwith in this_line_metrics

                if best_sortwith is None or \
                   best_sortwith < this_line_metrics[args.sortwith]:
                    best_sortwith = this_line_metrics[args.sortwith]
                    metric_dict[args.sortwith + '-best'] = best_sortwith

    return metric_dict, epochs


def get_epoch_time(metrics, epochs):
    first_time = 0
    last_time = 0

    # first line should always contain the beginning timestamp
    start_metric = metrics[0]
    val_metrics = [m for m in metrics if 'val' in m]
    # last val line should be time at last epoch
    last_metric = val_metrics[-1]

    assert 'start' in start_metric, \
        'expected start timestamp in first line of metrics file'
    if 'timestamp' not in start_metric or 'timestamp' not in last_metric:
        return ''

    timestamp_idx = start_metric.index('timestamp') + 1
    first_time = float(start_metric[timestamp_idx])
    timestamp_idx = last_metric.index('timestamp') + 1
    last_time = float(last_metric[timestamp_idx])
    elapsed_time = last_time - first_time

    if epochs == 0:
        epochs = 1
    epoch_time = time.strftime("%H:%M:%S", time.gmtime(elapsed_time / epochs))
    return epoch_time


def has_val(metrics):
    counts = [v[0] == 'val' for v in metrics]
    return sum(counts)


def get_final_metrics(metrics_fn):
    '''
    read in a metrics file

    return a dict of the final metrics for test/val
    also include epoch #
    and average minutes/epoch
    '''

    # Extract reported metrics
    metrics = load_csv(metrics_fn)
    if has_val(metrics):
        metric_dict, epochs = extract_nontime_metrics(metrics)
        metric_dict.update({'epoch time': get_epoch_time(metrics, epochs)})
        return metric_dict
    else:
        return None


def get_metrics(runs):
    '''
    Given the set of runs, pull out metrics

    input: run list
    output: metrics dict and metrics names
    '''
    metrics = {}
    for run in runs:
        metrics_fn = os.path.join(run, 'metrics.csv')
        if not os.path.isfile(metrics_fn):
            continue
        metrics_run = get_final_metrics(metrics_fn)
        if metrics_run is not None:
            metrics[run] = metrics_run

    return metrics


def any_different(alist):
    if len(alist) < 2:
        return False
    first = alist[0]
    total = sum([x != first for x in alist[1:]])
    if total:
        return True
    else:
        return False


def get_uncommon_hparam_names(all_runs):
    '''
    returns a list of uncommon hparam names

    input:
    - dict of hparams for each run
    '''
    # if 1 or fewer runs
    if len(all_runs) <= 1:
        return []

    # assemble all keys
    all_hparams = {}
    for run in all_runs.values():
        for p in run.keys():
            all_hparams[p] = 1

    # find all items that ever have different values
    uncommon_list = []
    for k in all_hparams:
        all_values = []
        for hparams in all_runs.values():
            if k in hparams:
                all_values.append(hparams[k])
            else:
                all_values.append(None)
        if any_different(all_values) and k not in args.ignore:
            uncommon_list.append(k)

    return uncommon_list


def summarize_experiment(parent_dir):
    '''
    Summarize an experiment, which can consist of many runs.
    '''
    assert os.path.exists(parent_dir), \
        'Couldn\'t find directory {}'.format(parent_dir)

    # assemble full paths to list of runs
    runs = get_runs(parent_dir)

    # dict of dicts of hparams
    hparams = get_hparams(runs)

    # dict of dicts of final test/val metrics
    metrics = get_metrics(runs)

    if not len(runs) or not len(metrics):
        print('No valid experiments found for {}'.format(parent_dir))
        return

    # a list of hparams to list out
    uncommon_hparams_names = get_uncommon_hparam_names(hparams)

    # create header for table
    header = ['run']
    header += uncommon_hparams_names
    first_valid_run = list(metrics.keys())[0]
    sorted_metric_keys = sorted(metrics[first_valid_run].keys())
    header += sorted_metric_keys

    # fill table values out
    tablebody = []

    # Only iterate through runs in the metrics dict, which is restricted to
    # runs for which there are results.
    for r in metrics:
        # start table with run name, derived from directory
        run_dir = r.replace('{}/'.format(parent_dir), '')
        entry = [run_dir]

        # add to table the uncommon hparams
        for v in uncommon_hparams_names:
            if v in hparams[r]:
                val = hparams[r][v]
                entry.append(val)
            else:
                entry.append(None)

        # add key metrics
        entry += [metrics[r][k] for k in sorted_metric_keys]

        # add entry to the table
        tablebody.append(entry)

    do_sort = False
    if args.sortwith is None:
        idx = 0
        # Find a field with 'loss' in the name, so we can sort with it.
        for h in header:
            if 'loss' in h:
                do_sort = True
                idx = header.index(h)
                break
    else:
        do_sort = True
        idx = header.index(args.sortwith + '-best')

    if do_sort:
        def get_key(entry):
            return entry[idx]

        try:
            tablebody = sorted(tablebody, key=get_key, reverse=True)
        except:
            print('Some data in table prevented sorting')
            pass

    if args.csv is not None:
        unf_table = [header] + tablebody
        f = open("{}.csv".format(args.csv), "w+")
        for row in unf_table:
            for column in row:
                f.write("{}, ".format(column))
            f.write("\n")

    # We chop long strings into multiple lines if they contain '.' or '_'
    # This helps keep the output table more compact
    header = [h.replace('.', '\n') for h in header]
    header = [h.replace('_', '\n') for h in header]

    table = [header] + tablebody
    print(tabulate(table, headers='firstrow', floatfmt='1.2e'))


def main():

    if args.logroot:
        logroot = args.logroot
    else:
        logroot = get_logroot()

    for adir in args.dirs:
        full_path = os.path.join(logroot, adir)
        summarize_experiment(full_path)


main()

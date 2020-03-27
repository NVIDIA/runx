# runx - An experiment management tool

runx helps to automate common tasks while doing research:
* hyperparameter sweeps
* logging, tensorboard, checkpoint management
* per-run unique directory creation
* experiment summarization
* code checkpointing
* easy integration

## Quick-start Installation

Install with source:
```
> git clone https://github.com/NVIDIA/runx
> cd runx
> python setup.py install --user
```

Install using pip
TBD

## Introduction: a simple example
Suppose you have an existing project that you call as follows:

```bash
> python train.py --lr 0.01 --solver sgd
```
To run a hyperparameter sweep, you'd normally have to code up a one-off script to generate the sweep.
But with runx, you would simply define a yaml that defines lists of hyperparams that you'd like to use.

Start by creating a yaml file called `sweep.yml`:
```yml
cmd: 'python train.py'

hparams:
  lr: [0.01, 0.02]
  solver: ['sgd', 'adam']
```

Now you can run the sweep with runx:

```bash
> python -m runx.runx sweep.yml

python train.py --lr 0.01 --solver sgd
python train.py --lr 0.01 --solver adam
python train.py --lr 0.02 --solver sgd
python train.py --lr 0.02 --solver adam
```
You can see that runx automatically computes the cross product of all hyperparameters, which in this
case results in 4 runs. It then builds commandlines by concatenating the hyperparameters with
the training command.

runx is intended to be used to launch batch jobs to a farm. Because running many training runs
interactively would take a long time! 
Farm support is simple. Create a .runx file that configures the farm:

```yaml
LOGROOT: /home/logs
FARM: bigfarm

bigfarm:
  SUBMIT_CMD: 'submit_job'
  RESOURCES:
     gpu: 2
     cpu: 16
     mem: 128
```
You've told it a few things here: where to put the assets for each run, under /home/logs, and
what command and what resources to use to submit each run to the farm.

Now when you call runx, because `FARM` is defined as `bigfarm`, it will use `bigfarm`'s
definition to perform submissions.

```bash
> python -m runx.runx sweep.yml

submit_job --gpu 2 --cpu 16 --mem 128 -c "python train.py --lr 0.01 --solver sgd"
submit_job --gpu 2 --cpu 16 --mem 128 -c "python train.py --lr 0.01 --solver adam"
submit_job --gpu 2 --cpu 16 --mem 128 -c "python train.py --lr 0.02 --solver sgd"
submit_job --gpu 2 --cpu 16 --mem 128 -c "python train.py --lr 0.02 --solver adam"
```
Here, submit_job is a placeholder for your farm submission command.

Finally, we want the results for each training run to go into a unique output/log directory.
We don't want things like tensorboard files or logfiles to write over each other.
runx solves this problem by automatically generating a unique output directory per run.
This directory is passed to your training script via a special field: `LOGDIR`. Your training
script must use this path and write it's output there.

```yml
CMD: 'python train.py'

HPARAMS:
  lr: [0.01, 0.02]
  solver: ['sgd', 'adam']
  logdir: LOGDIR
```

Now when we launch the jobs, runx automatically generates unique output directories and passes the paths to your training script:
```bash
> python -m runx.runx sweep.yml

submit_job --gpu 2 --cpu 16 --mem 128 -c "python train.py --lr 0.01 --solver sgd  --logdir /home/logs/athletic-wallaby_2020.02.06_14.19"
submit_job --gpu 2 --cpu 16 --mem 128 -c "python train.py --lr 0.01 --solver adam  --logdir /home/logs/industrious-chicken_2020.02.06_14.19"
submit_job --gpu 2 --cpu 16 --mem 128 -c "python train.py --lr 0.02 --solver sgd  --logdir /home/logs/arrogant-buffalo_2020.02.06_14.19"
submit_job --gpu 2 --cpu 16 --mem 128 -c "python train.py --lr 0.02 --solver adam  --logdir /home/logs/vengeful-jaguar_2020.02.06_14.19"
```

After you've run your experiment, you will likely want to summarize the results. 
Which training run was best?

You summarize your runs with `sumx`. sumx knows where your logs are kept: it's the concatenation of LOGROOT and your experiment name.
So all you need to do is tell sumx which experiment you want summarized:

```bash
> python -m runx.sumx sweep --sortwith acc

        lr    solver  acc   epoch  epoch_time
------	----  ------  ----  -----  ----------
run4    0.02  adam    99.1   10     5:11
run3    0.02  sgd     99.0   10     5:05
run1    0.01  sgd     98.2   10     5:15
run2    0.01  adam    98.1   10     5:12
```
sumx is part of the runx suite, and is able to summarize the different hyperparmeters used as well as the
metrics/results of your runs. Notice that we used the --sortwith feature of sumx, which sorts your results so
you can easily locate your best runs.

This is the basic idea.
The following sections will go into more details about all the various features.

## runx Architecture
runx consists of three main modules:

* **runx**
  * Launch sweeps of training runs using a concise yaml format that allows for multiple values for each hyperparameter
  * In particular, when you call runx:
    * Calculate cross product of all hyperparameters -> runs
	* For each run, create an output directory, copy your code there, and then launch the training command
* **logx**
  * Logging of metrics, messages, checkpoints, tensorboard
* **sumx**
  * Summarize the results of training runs, showing results and unique hyperparameters

These modules are intended to be used jointly, but if you just want to use runx, that's fine.
However using sumx requires that you've used logx to record metrics.



## Installation

The easiest way to install is via pip:

```bash
> pip install runx --extra-index-url=https://pypi.perflab.nvidia.com
```

Alternatively, you could make runx a git submodule of your project:

``` bash
cd <your repo>
git submodule add -b master ssh://git@gitlab-master.nvidia.com:runx.git
```

## Create a project-specific configuration file

Create a .runx file within your repo using the example below.
The .runx file defines whether and how to submit jobs to your compute cluster, and where to put the output.

* `LOGROOT` - the root directory where you want your logs placed. This is a path that any farm job can write to.
* `FARM` - if defined, jobs should be submitted to this farm, else run interactively
* `SUBMIT_CMD` - the cluster submission command
* `RESOURCES` - hyperparameters passed to `submit_cmd`
* `CODE_IGNORE_PATTERNS` - ignore these files patterns when copying code to output directory

Here's an example of such a file:

```yaml
LOGROOT: /home/logs
FARM: bigfarm
CODE_IGNORE_PATTERNS: '.git,*.pyc,docs*,test*'

# Farm resource needs
bigfarm:
    SUBMIT_CMD: 'submit_job'
    RESOURCES:
        image: mydocker-image-big:1.0
        gpu: 8
        cpu: 64
        mem: 450

smallfarm:
    SUBMIT_CMD: 'submit_small'
    RESOURCES:
        image: mydocker-image-small:1.2
        gpu: 4
        cpu: 32
        mem: 256
```

## Run directory, logfiles

runx has two level of experiment hierarchy: **experiments** and **runs**.
An experiment correponds to a single yaml file, which may contain many runs.

runx creates both a parent experiment directory and a unique output directory for each run.
The name of the experiment directory is `LOGROOT/<experiment name>`, so in the example of
sweep.yml, the experiment name is `sweep`, derived from the yaml filename.

```bash
/home/logs
  sweep/
     curious-rattlesnake_2020.02.06_14.19/
     ambitious-lobster_2020.02.06_14.19/
     ...
```

The individual run directories are named with a combination of `coolname` and date. The
use of `coolname` makes it much easier to refer to a given run than referring to a date code.

The names can be customized using the RUNX.TAG field in your experiment yaml.

Copying code: because runx actually copies your code to each run's output directory, runx is essentially checkpointing your code. This provides two services: (1) it allows you to change your sourcecode after you've launch some runs (2) it records exactly what code was used for a particular run

## Experiment yaml details

### Booleans
If you want to specify that a boolean flag should be on or off, this is done using `true` and `false` keywords:
  ```
  some_flag: [true, false]
  ```
This would result having one run with `--some_flag` and another run without that flag

If instead you want to pass an actual string, you could instad do the following:
```
  some_arg: ['True', 'False']
```
This would result in one run with `--some_arg True` and other run with `--some_arg False`

If you'd like an argument to not be passed into your script at all, you can set it to `None`
```
  some_arg: None
```

### Lists, Inheritance
Oftentimes, you might want to define separate lists of hyperparameters in your experiment.
For example:
 1. arch = alexnet with lr=[0.01, 0.02]
 2. arch = resnet50 with lr=[0.002, 0.005]

You can do this with hparams defined as follows:
```yaml
hparams: [
  {
   logdir: LOGDIR,
   adam: true,
   arch: alexnet,
   lr: [0.01, 0.02],
   epochs: 10,
   RUNX.TAG: 'alexnet',
  },
  {
   arch: resnet50,
   lr: [0.002, 0.005],
   RUNX.TAG: 'resnet50',
  }
]
```
You might observe that hparams is now a list of two dicts. 
The nice thing is that runx assumes inheritance from the first item in the list to all remaining dicts, so that you don't have to re-type all the redundant hyperparms.

When you pass this yaml to runx, you'll get the following out:

``` bash
submit_job ... --name alexnet_2020.02.06_6.32  -c "python train.py --logdir ... --lr 0.01 --adam --arch alexnet --epochs 10
submit_job ... --name alexnet_2020.02.06_6.40  -c "python train.py --logdir ... --lr 0.02 --adam --arch alexnet --epochs 10
submit_job ... --name resnet50_2020.02.06_6.45 -c "python train.py --logdir ... --lr 0.002 --adam --arch resnet50 --epochs 10
submit_job ... --name resnet50_2020.02.06_6.50 -c "python train.py --logdir ... --lr 0.005 --adam --arch resnet50 --epochs 10
```
Because of inheritance, adam, arch, and epochs params are set identically in each run.

This is also showing the use of the magic variable `RUNX.TAG`, which allows you to add a tag to a subset of your experiment. This is the same as if you'd used the --tag <tagname> option to runx.py, except that here you can specify the tag within the hparams data structure. The value of `RUNX.TAG` is not passed to your training script

## logx - logging, tensorboarding, checkpointing
In order to use sumx, you need to export metrics with logx.
logx helps to write metrics in a canonical way, so that sumx can summarize the results.

logx can also make it easy for you to output log information to a file (and stdout)
logx can also manage saving of checkpoints automatically, with the benefit being that logx will keep around only the latest and best checkpoints, saving much disk space.

The basic way you use logx is to modify your training code in the following ways:

At the top of your training script (or any module that calls logx functions:
```python
from runx.logx import logx
```

Before using logx, you must initialize it as follows:
```python
   logx.initialize(logdir=args.logdir, coolname=True, tensorboard=True)
```
Make sure that you're only calling logx from rank=0, in the event that you're using distributed data parallel.


Then, substitute the following logx calls into your code:

| From                | To                | What                      |
| ------------------- | ----------------- | ------------------------- |
| print()             | logx.msg()        | stdout messages           |
| writer.add_scalar() | logx.add_scalar() | tensorboard scalar writes |
| writer.add_image()  | logx.add_image()  | tensorboard image writes  |
|                     | logx.save_model() | save latest/best models   |

Finally, in order for sumx to be able to read the results of your run, you have to push your metrics to logx. You should definitely push the 'val' metrics, but can push 'train' metrics if you like (sumx doesn't consume them at the moment). 

```python
# define which metrics to record
metrics = {'loss': test_loss, 'accuracy': accuracy}
# push the metrics to logfile
logx.metric(phase='val', metrics=metrics, epoch=epoch)
```

Some important points of logx.metric():
* The `phase` argument describes whether the metric is a train or validation metric.
* You should set idx == epoch for validation metrics. And for training, idx is typically the iteration count.


Here's a final feature of logx: saving of the model. This feature helps save not only the latest but also the best model.

```python
save_dict = {'epoch': epoch + 1,
             'arch': args.arch,
             'state_dict': model.state_dict(),
             'best_acc1': best_acc1,
             'optimizer' : optimizer.state_dict()}
logx.save_model(save_dict, metric=accuracy, epoch=epoch, higher_better=True)
```

You do have to tell save_model whether the metric is better when it's higher or lower.

## sumx - summarizing your runs

sumx summarizes the results of your runs. It requires that you've logged your metrics with logx.metric().
We chose this behavior instead of reading Tensorboard files directly because that would be much slower.

``` bash
> python -m runx.sumx sweep
        lr    solver  acc   epoch  epoch_time
run4    0.02  adam    99.1  10     5:21
run3    0.02  sgd     99.0  10     5:02
run1    0.01  sgd     98.2  10     5:40
run2    0.01  adam    98.1  10     5:25
```

A few features worth knowing about:
* use `--sortwith` to sort the output by a particular field (like accuracy) that you care about most
* sumx tells you what epoch your run is current on
* sumx tells you the average epoch time, which is handy if you are monitoring training speed
* use the optional `--ignore` flag to limit what fields sumx prints out


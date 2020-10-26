# runx - An experiment management tool

`runx` helps to automate common tasks while doing research:
* hyperparameter sweeps
* logging, tensorboard, checkpoint management
* experiment summarization
* code checkpointing
* unique, per-run, directory creation

# Table of Contents
- [runx - An experiment management tool](#runx---an-experiment-management-tool)
- [Table of Contents](#table-of-contents)
  - [Quick-start Installation](#quick-start-installation)
  - [Introduction example](#introduction-example)
    - [runx is especially useful to launch batch jobs to a farm.](#runx-is-especially-useful-to-launch-batch-jobs-to-a-farm)
    - [Unique run directories](#unique-run-directories)
  - [Summarization with sumx](#summarization-with-sumx)
  - [runx Architecture](#runx-architecture)
  - [Create a project-specific configuration file](#create-a-project-specific-configuration-file)
  - [Run directory, logfiles](#run-directory-logfiles)
  - [Staging of code](#staging-of-code)
  - [Experiment yaml details](#experiment-yaml-details)
    - [Special variables](#special-variables)
    - [HPARAMS](#hparams)
      - [A simple example of HPARAMS is:](#a-simple-example-of-hparams-is)
    - [Booleans](#booleans)
    - [Lists, Inheritance](#lists-inheritance)
  - [logx - logging, tensorboarding, checkpointing](#logx---logging-tensorboarding-checkpointing)
  - [sumx - summarizing your runs](#sumx---summarizing-your-runs)
  - [NGC Support](#ngc-support)

## Quick-start Installation

Install with pip:
```
> pip install runx
```

Install with source:
```
> git clone https://github.com/NVIDIA/runx
> cd runx
> python setup.py .
```

## Introduction example
Suppose you have an existing project that you call as follows:

```bash
> python train.py --lr 0.01 --solver sgd
```
To run a hyperparameter sweep, you'd normally have to code up a one-off script to generate the sweep. But with runx, you would simply define a yaml that defines lists of hyperparams that you'd like to use.

Start by creating a yaml file called `sweep.yml`:
```yml
CMD: 'python train.py'

HPARAMS:
  lr: [0.01, 0.02]
  solver: ['sgd', 'adam']
```

Now you can run the sweep with runx:

```bash
 > python -m runx.runx sweep.yml -i

python train.py --lr 0.01 --solver sgd
python train.py --lr 0.01 --solver adam
python train.py --lr 0.02 --solver sgd
python train.py --lr 0.02 --solver adam
```
You can see that runx automatically computes the cross product of all hyperparameters, which in this case results in 4 runs. It then builds commandlines by concatenating the hyperparameters with the training command.

A few useful runx options:
```
-n     don't run, just print the command
-i     interactive mode (as opposed to submitting jobs to a farm)
```
### runx is especially useful to launch batch jobs to a farm.

Farm support is simple. First create a .runx file that configures the farm:
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
**LOGROOT**: this is where the output of runs should go

**FARM**: you can define multiple farm targets. This selects which one to use

**SUBMIT_CMD**: the script you use to launch jobs to a farm

**RESOURCES**: the arguments to present to SUBMIT_CMD

Now when you run runx, it will generate commands that will attempt to launch jobs to a farm using your SUBMIT_CMD. Notice that we left out the `-i` cmdline arg because now we want to submit jobs and not run them interactively.

```bash
> python -m runx.runx sweep.yml

submit_job --gpu 2 --cpu 16 --mem 128 -c "python train.py --lr 0.01 --solver sgd"
submit_job --gpu 2 --cpu 16 --mem 128 -c "python train.py --lr 0.01 --solver adam"
submit_job --gpu 2 --cpu 16 --mem 128 -c "python train.py --lr 0.02 --solver sgd"
submit_job --gpu 2 --cpu 16 --mem 128 -c "python train.py --lr 0.02 --solver adam"
```

### Unique run directories
We want the results for each training run to go into a unique output/log directory.  We don't want things like tensorboard files or logfiles to write over each other. `runx` solves this problem by automatically generating a unique output directory per run.

You have access to this unique directory name within your experiment yaml via the special variable: `LOGDIR`. Your training
script may use this path and write its output there.

```yml
CMD: 'python train.py'

HPARAMS:
  lr: [0.01, 0.02]
  solver: ['sgd', 'adam']
  logdir: LOGDIR
```

In the above experiment yaml, we have passed LOGDIR as an argument to your training script. When we launch the jobs, runx automatically generates unique output directories and passes the paths to your training script:

```bash
> python -m runx.runx sweep.yml

submit_job --gpu 2 --cpu 16 --mem 128 -c "python train.py --lr 0.01 --solver sgd  --logdir /home/logs/athletic-wallaby_2020.02.06_14.19"
submit_job --gpu 2 --cpu 16 --mem 128 -c "python train.py --lr 0.01 --solver adam  --logdir /home/logs/industrious-chicken_2020.02.06_14.19"
submit_job --gpu 2 --cpu 16 --mem 128 -c "python train.py --lr 0.02 --solver sgd  --logdir /home/logs/arrogant-buffalo_2020.02.06_14.19"
submit_job --gpu 2 --cpu 16 --mem 128 -c "python train.py --lr 0.02 --solver adam  --logdir /home/logs/vengeful-jaguar_2020.02.06_14.19"
```

## Summarization with sumx
After you've run your experiment, you will likely want to summarize the results.  You might want to know:
* Which training run was best?
* How long was an epoch
* What about other metrics?

You summarize your runs with on the commandline with `sumx`. All you need to do is tell `sumx` which experiment you want summarized. `sumx` knows what your LOGROOT (it'll get that from the .runx file) and so it looks within that directory for your experiment directory.

In the following example, we ask `sumx` to summarize the `sweep` experiment.

```bash
> python -m runx.sumx sweep --sortwith acc

        lr    solver  acc   epoch  epoch_time
------	----  ------  ----  -----  ----------
run4    0.02  adam    99.1   10     5:11
run3    0.02  sgd     99.0   10     5:05
run1    0.01  sgd     98.2   10     5:15
run2    0.01  adam    98.1   10     5:12
```
`sumx` is part of the runx suite, and is able to summarize the different hyperparmeters used as well as the metrics/results of your runs. Notice that we used the --sortwith feature of sumx, which sorts your results so you can easily locate your best runs.

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


## Create a project-specific configuration file
In order to use `runx`, you need to create a configuration file in the directory where you'll call the `runx` CLI.

The .runx file defines a number of critical fields:
* `LOGROOT` - the root directory where you want your logs placed. This is a path that any farm job can write to.
* `FARM` - if defined, jobs should be submitted to this farm, else run interactively
* For a given farm, these fields are required:
  + `SUBMIT_CMD` - the farm submission command
  + `RESOURCES` - hyperparameters passed to the `SUBMIT_CMD`. You can list any number of these items, the ones shown below are just examples.
* `CODE_IGNORE_PATTERNS` - ignore these files patterns when copying code to output directory

Here's an example of such a file:

```yaml
LOGROOT: /home/logs
CODE_IGNORE_PATTERNS: '.git,*.pyc,docs*,test*'
FARM: bigfarm

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

runx has two level of experiment hierarchy: **experiments** and **runs**. An `experiment` corresponds to a single yaml file, which may contain many `runs`.

`runx` creates both a parent experiment directory and a unique subdirectory for each run. The name of the experiment directory is `LOGROOT/<experiment name>`, so in the example of sweep.yml, the experiment name is `sweep`, derived from the yaml filename.

For example, this might be the directory structure for the sweep study:
```bash
/home/logs
  sweep/
     curious-rattlesnake_2020.02.06_14.19/
     ambitious-lobster_2020.02.06_14.19/
     ...
```

The individual run directories are named with a combination of `coolname` and date. The use of `coolname` makes it much easier to refer to a given run than referring to a date code.

If you include the RUNX.TAG field in your experiment yaml or if you supply the --tag argument to the `runx` CLI, the names will include that tag.

## Staging of code
`runx` actually makes a copy of your code within each run's log directory. This is done for a number of reasons:
* If you wish to continue modifying your code, while a training run is going on, you may do so without worry whether it will affect the running job(s)
* In case your job dies and you must restart it, the code and training environment is self-contained within the logdir of a run.
* This is also useful for documentation purposes: in case you ever want to know exactly the state of the code for a given run.


## Experiment yaml details

### Special variables
**CMD** - Your base training command. You typically don't include any args here.
**HPARAMS** - All hyperparmeters. This is a datastructure that may either be a simple dict of params or may be a list of dicts. Furthermore, each hyperparameter may be a scalar or list or boolean.
**PYTHONPATH** - This is field optional. For the purpose of altering the default PYTHONPATH which is simply LOGDIR/code. Can be a colon-separated list of paths. May include LOGDIR special variable.

### HPARAMS

#### A simple example of HPARAMS is:
```yaml
CMD: "python train.py"

HPARAMS:
  logdir: LOGDIR
  adam: true
  arch: alexnet
  lr: [0.01, 0.02]
  epochs: 10
  RUNX.TAG: 'alexnet'
```
Here, there will be 2 runs that will be created.

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
PYTHONPATH: LOGDIR/code:LOGDIR/code/lib
CMD: "python train.py"

HPARAMS: [
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
  },
  {
   RUNX.SKIP: true,
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

## NGC Support
NGC support is now standard. Your `.runx` file should look like the following.

```yaml

LOGROOT: /path/to/logroot

FARM: ngc

ngc:
    NGC_LOGROOT: /path/to/ngc_logroot
    WORKSPACE: <your ngc workspace>
    SUBMIT_CMD: 'ngc batch run'
    RESOURCES:
       image: nvidian/pytorch:19.10-py3
       instance: dgx1v.16g.1.norm
       ace: nv-us-west-2
       result: /result
```

Necessary steps:
* Fill out a path to LOGROOT, which is a client-side staging directory for the log directory
* Create a RW NGC workspace and fill in `WORKSPACE` with it
* Mount this workspace on your local machine and fill in `NGC_LOGROOT` with this path. When the
  job is launched, this is also the path used to mount the workspace on the running instance.
* Fill out any necessary fields under `RESOURCES`. Recall that these parameters are passed on
  to the `SUBMIT_CMD`, which must be `ngc batch run`.


You should be able to launch jobs to NGC using this configuration. When jobs write their results, you should also be able to see the results in the mounted workspace,
and then you should be able to run runx.sumx in order to summarize the results of those runs.

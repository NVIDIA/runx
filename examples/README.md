Examples of using runx.

In both these examples, we use the `-n` flag so that the commands don't actually execute, but are instead printed to show you what would be normally run.

```bash
> python -m runx.runx mnist.yml -n

submit_job --gpu 2 --cpu 16 --mem 128 --command 'cd /home/logs/mnist/imaginary-quoll_2020.03.27_10.39/code; PYTHONPATH=/home/logs/mnist/imaginary-quoll_2020.03.27_10.39/code exec python mnist.py --lr 0.01 --momentum 0.5 --logdir /home/logs/mnist/imaginary-quoll_2020.03.27_10.39 '
submit_job --gpu 2 --cpu 16 --mem 128 --command 'cd /home/logs/mnist/solid-viper_2020.03.27_10.39/code; PYTHONPATH=/home/logs/mnist/solid-viper_2020.03.27_10.39/code exec python mnist.py --lr 0.01 --momentum 0.25 --logdir /home/logs/mnist/solid-viper_2020.03.27_10.39 '
submit_job --gpu 2 --cpu 16 --mem 128 --command 'cd /home/logs/mnist/stereotyped-catfish_2020.03.27_10.39/code; PYTHONPATH=/home/logs/mnist/stereotyped-catfish_2020.03.27_10.39/code exec python mnist.py --lr 0.015 --momentum 0.5 --logdir /home/logs/mnist/stereotyped-catfish_2020.03.27_10.39 '
submit_job --gpu 2 --cpu 16 --mem 128 --command 'cd /home/logs/mnist/expert-okapi_2020.03.27_10.39/code; PYTHONPATH=/home/logs/mnist/expert-okapi_2020.03.27_10.39/code exec python mnist.py --lr 0.015 --momentum 0.25 --logdir /home/logs/mnist/expert-okapi_2020.03.27_10.39 '
submit_job --gpu 2 --cpu 16 --mem 128 --command 'cd /home/logs/mnist/shrewd-ostrich_2020.03.27_10.39/code; PYTHONPATH=/home/logs/mnist/shrewd-ostrich_2020.03.27_10.39/code exec python mnist.py --lr 0.02 --momentum 0.5 --logdir /home/logs/mnist/shrewd-ostrich_2020.03.27_10.39 '
submit_job --gpu 2 --cpu 16 --mem 128 --command 'cd /home/logs/mnist/umber-spoonbill_2020.03.27_10.39/code; PYTHONPATH=/home/logs/mnist/umber-spoonbill_2020.03.27_10.39/code exec python mnist.py --lr 0.02 --momentum 0.25 --logdir /home/logs/mnist/umber-spoonbill_2020.03.27_10.39 '
```

mnist_multi.yml
```bash
> python -m runx.runx mnist_multi.yml -n

submit_job --image hw-adlr-docker/atao/superslomo:v2 --partition volta-dcg-short --gpu 1 --cpu 8 --mem 64 --duration 1 --command 'cd /home/logs/mnist_multi/ubiquitous-fulmar_2020.03.27_10.41/code; PYTHONPATH=/home/logs/mnist_multi/ubiquitous-fulmar_2020.03.27_10.41/code exec python mnist.py --TAG_NAME foo --lr 0.01 --momentum 0.5 --logdir /home/logs/mnist_multi/ubiquitous-fulmar_2020.03.27_10.41 '
submit_job --image hw-adlr-docker/atao/superslomo:v2 --partition volta-dcg-short --gpu 1 --cpu 8 --mem 64 --duration 1 --command 'cd /home/logs/mnist_multi/psychedelic-albatross_2020.03.27_10.41/code; PYTHONPATH=/home/logs/mnist_multi/psychedelic-albatross_2020.03.27_10.41/code exec python mnist.py --TAG_NAME foo --lr 0.01 --momentum 0.25 --logdir /home/logs/mnist_multi/psychedelic-albatross_2020.03.27_10.41 '
submit_job --image hw-adlr-docker/atao/superslomo:v2 --partition volta-dcg-short --gpu 1 --cpu 8 --mem 64 --duration 1 --command 'cd /home/logs/mnist_multi/classic-viper_2020.03.27_10.41/code; PYTHONPATH=/home/logs/mnist_multi/classic-viper_2020.03.27_10.41/code exec python mnist.py --TAG_NAME bar --lr 0.02 --momentum 0.25 --logdir /home/logs/mnist_multi/classic-viper_2020.03.27_10.41 '
submit_job --image hw-adlr-docker/atao/superslomo:v2 --partition volta-dcg-short --gpu 1 --cpu 8 --mem 64 --duration 1 --command 'cd /home/logs/mnist_multi/prehistoric-malamute_2020.03.27_10.41/code; PYTHONPATH=/home/logs/mnist_multi/prehistoric-malamute_2020.03.27_10.41/code exec python mnist.py --TAG_NAME bar --lr 0.02 --momentum 0.12 --logdir /home/logs/mnist_multi/prehistoric-malamute_2020.03.27_10.41 '
```

```bash
> python -m runx.runx imgnet.yml -n

submit_job --gpu 2 --cpu 16 --mem 128 --command 'cd /home/logs/imgnet/famous-albatross_2020.03.27_10.38/code; PYTHONPATH=/home/logs/imgnet/famous-albatross_2020.03.27_10.38/code exec python imgnet.py /data/ImageNet --lr 0.1 --logdir /home/logs/imgnet/famous-albatross_2020.03.27_10.38 '
submit_job --gpu 2 --cpu 16 --mem 128 --command 'cd /home/logs/imgnet/piquant-ara_2020.03.27_10.38/code; PYTHONPATH=/home/logs/imgnet/piquant-ara_2020.03.27_10.38/code exec python imgnet.py /data/ImageNet --lr 0.05 --logdir /home/logs/imgnet/piquant-ara_2020.03.27_10.38 '
```


Examples of using runx.

# Interactive runs.

...These runs will use the .runx file. Please set the LOGROOT in there to a path where you want logs to go.

Dry run:
```bash
  > python -m runx.runx mnist.yml -i -n
```

Real run:
```bash
  > python -m runx.runx mnist.yml -i
```

Summarize the results:
```bash
  > python -m runx.sumx mnist
```

# Submissions to a compute farm
dry run:
```bash
  > python -m runx.runx mnist.yml -n -c .runx_ngc
```

real run:
```bash
  > python -m runx.runx mnist.yml -c .runx_ngc
```

Can also try imgnet.yml.

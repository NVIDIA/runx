Examples of using runx.
Please create a .runx file from .runx_example first.

Interactive:
  > python -m runx.runx mnist.yml -i -n   -> dry run
  > python -m runx.runx mnist.yml -i      -> real run

Batch runs:
  > python -m runx.runx mnist.yml -n      -> dry run
  > python -m runx.runx mnist.yml         -> real run

Can also try imgnet.yml.
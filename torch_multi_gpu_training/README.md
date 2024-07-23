## torch multi gpu training

a simple and dumb example, on how to do it. If you're an idiot like myself, I only recently learned how to properly do this. 

to run this, simply -
```
torchrun --nproc_per_node=<number_of_GPUs> train.py
```

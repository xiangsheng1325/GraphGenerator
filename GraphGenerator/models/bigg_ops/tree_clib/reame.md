# Installation on NVIDIA GeForce RTX 3090 and CUDA 11.1
Using default Makefile, there is no bug when installing BiGG in this environment.

# Installation on different devices and environments
The installation of BiGG requires one more step, i.e., check the computing capability of your gpu.

## check the computing capability
Visiting this website, we can query the corresponding computing capability: https://developer.nvidia.com/cuda-gpus

## choosing specific Makefile
According to the query result, choosing specific Makefile_xx as your Makefile.

For example, if the existing GPU device is GeForce RTX 2080, the `Makefile_75` or `Makefile_70` can be renamed
as `Makefile` because the computing capability of 2080 Ti is `7.5`.


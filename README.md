# Posterior Sampling Based on Gradient Flows of the MMD with Negative Distance Kernel

This code belongs to the paper 'Posterior Sampling Based on Gradient Flows of the MMD with Negative Distance Kernel' [1] available at https://arxiv.org/abs/2310.03054. Please cite the paper, if you use our code.

It contains implementations for the experiments of Section 4. 
For questions or bug reports, please contact Fabian Altekrüger (fabian.altekrueger@hu-berlin.de) or Johannes Hertrich (j.hertrich@math.tu-berlin.de).

## REQUIREMENTS

We tested the code with python 3.9.13 and the following package versions:

- pytorch 1.21.1
- matplotlib 3.6.1
- scikit-image 0.19.3
- dival 0.6.1
- odl 1.0.0

Usually, the code is also compatible with some other versions of the corresponding packages.

## USAGE 

You can start the training of the conditional MMD flow by calling the script 'run_{dataset}_{setting}.py' for the datasets MNIST, FashionMNIST, CIFAR10 and CelebA as well as a CT dataset and a material's microstructures dataset. For CT you have the choice between the limited angle setting, which is the default setting, and the low-dose setting, for which you need to set the flag 'lowdose' to True.
If you do not wish to save intermediate steps of the flow, then set the flag 'save' to False.
If you already have trained the conditional MMD flow and want to visualize the results, then set the flag 'visualize' to True.

The used data for CT is from the LoDoPaB dataset [2], which is available at https://zenodo.org/record/3384092##.Ylglz3VBwgM.
The used material's microstructures data, which you can find in the folder 'material_data', have been acquired in the frame of the EU Horizon 2020 Marie Sklodowska-Curie Actions Innovative Training Network MUMMERING (MUltiscale, Multimodal and Multidimensional imaging for EngineeRING, Grant Number 765604) at the beamline TOMCAT of the SLS by A. Saadaldin, D. Bernard, and F. Marone Welford. 

## References

[1] P. Hagemann, J. Hertrich, F. Altekrüger, R. Beinert, J. Chemseddine, G. Steidl.
Posterior Sampling Based on Gradient Flows of the MMD with Negative Distance Kernel.
ArXiv preprint #2310.03054

[2] J. Leuschner, M. Schmidt, D. O. Baguer and P. Maass.
LoDoPaB-CT, a benchmark dataset for low-dose computed tomography reconstruction.
Scientific Data, 9(109), 2021.

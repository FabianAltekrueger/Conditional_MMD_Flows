# Posterior Sampling Based on Gradient Flows of the MMD with Negative Distance Kernel

This code belongs to the paper 'Posterior Sampling Based on Gradient Flows of the MMD with Negative Distance Kernel' [1] available at https://openreview.net/forum?id=YrXHEb2qMb. Please cite the paper, if you use our code.

It contains implementations for the experiments of Section 4. 
For questions or bug reports, please contact Fabian Altekrüger (fabian.altekrueger@hu-berlin.de) or Johannes Hertrich (j.hertrich@math.tu-berlin.de).

## REQUIREMENTS

The code requires several Python packages. You can create a conda environment using `environment.yaml` by the command
```python
conda env create --file=environment.yaml
```
Additionally, for the CT example you have to install the latest version of ODL via pip:
```python
pip install https://github.com/odlgroup/odl/archive/master.zip --upgrade
```

## USAGE 

You can start the training of the conditional MMD flow by calling the script 'run_{dataset}_{setting}.py' for the datasets MNIST, FashionMNIST, CIFAR10 and CelebA as well as a CT dataset and a material's microstructures dataset. For CT you have the choice between the limited angle setting, which is the default setting, and the low-dose setting, for which you need to set the flag 'lowdose' to True.
If you do not wish to save intermediate steps of the flow, then set the flag 'save' to False.
If you already have trained the conditional MMD flow and want to visualize the results, then set the flag 'visualize' to True.

The used data for CT is from the LoDoPaB dataset [2], which is available at https://zenodo.org/record/3384092##.Ylglz3VBwgM.
The used material's microstructures data, which you can find in the folder 'material_data', have been acquired in the frame of the EU Horizon 2020 Marie Sklodowska-Curie Actions Innovative Training Network MUMMERING (MUltiscale, Multimodal and Multidimensional imaging for EngineeRING, Grant Number 765604) at the beamline TOMCAT of the SLS by A. Saadaldin, D. Bernard, and F. Marone Welford. 

## References

[1] P. Hagemann, J. Hertrich, F. Altekrüger, R. Beinert, J. Chemseddine, G. Steidl.
Posterior Sampling Based on Gradient Flows of the MMD with Negative Distance Kernel.
International Conference on Learning Representations 2024

[2] J. Leuschner, M. Schmidt, D. O. Baguer and P. Maass.
LoDoPaB-CT, a benchmark dataset for low-dose computed tomography reconstruction.
Scientific Data, 9(109), 2021.

## CITATION
```python
@inproceedings{HHABCS2024,
    author    = {Hagemann, Paul and Hertrich, Johannes and Altekrüger, Fabian and Beinert, Robert and Chemseddine, Jannis and Steidl, Gabriele},
    title     = {Posterior Sampling Based on Gradient Flows of the {MMD} with Negative Distance Kernel},
    booktitle={The Twelfth International Conference on Learning Representations},
    year={2024}
}
```


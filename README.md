[![License: CC BY-NC-SA 4.0](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-ff69b4.svg)](http://creativecommons.org/licenses/by-nc-sa/4.0/)

# THIS IS A FORK

## InpaintNet: Learning to Traverse Latent Spaces for Musical Score Inpaintning

### About

This repository contains the source code and dataset for training a deep learning-based model to perform *inpainting* on musical scores, i.e., to connect two musical excerpts in a musically meaningful manner (see figures below for schematics). 

<p align="center">
    <img src=figs/inpainting_block_diagram.png alt="Inpainting Task Schematic" width="500">
</p>

The approach followed relies on training a RNN-based architecture to learn to traverse the latent space of a VAE-based deep generative model.

<p align="center">
    <img src=figs/approach_schematic.png alt="Inpainting Approach Schematic" width="500">
</p>

### Installation and Usage
Install `anaconda` or `miniconda` by following the instruction [here](https://docs.conda.io/projects/conda/en/latest/user-guide/install/).

Create a new conda environment using the `enviroment.yml` file located in the root folder of this repository. The instructions for the same can be found [here](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file).

To install, either download / clone this repository. Open a new terminal, `cd` into the root folder of this repository and run the following command

    pip install -e .

### Contents

The contents of this repository are as follows: 
* `DatasetManger`: Module for handling data.
* `AnticipationRNN`: Module implementing model, trainer and tester classes for the AnticipationRNN model. 
* `MeasureVAE`: Module implementing model, trainer and tester classes for the MeasureVAE model.
* `LatentRNN`: Module implementing model, trainer and tester classes for the LatentRNN model.
* `utils`: Module with model and training utility classes and methods
* other scripts to train / test the models


### Attribution

This research work is published as a conference [paper](http://archives.ismir.net/ismir2019/paper/000040.pdf) at ISMIR, 2019. Arxiv Preprint available [here](https://arxiv.org/abs/1907.01164).

> Ashis Pati, Alexander Lerch, Gaëtan Hadjeres. "Learning to Traverse Latent Spaces for Musical Score Inpaintning", Proc. of the 20th International Society for Music Information Retrieval Conference (ISMIR), Delft, The Netherlands, 2019.

```
@inproceedings{pati2019inpainting,
  title={Learning to Traverse Latent Spaces for Musical Score Inpaintning},
  author={Pati, Ashis and Lerch, Alexander and Hadjeres, Gaëtan},
  booktitle={20th International Society for Music Information Retrieval Conference (ISMIR)},
  year={2019},
  address={Delft, The Netherlands}
}
```

Please cite the above publication if you are using the code/data in this repository in any manner. 


<a name="License"></a>License
--------------------
<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-sa/4.0/88x31.png" /></a><br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/">Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License</a>.

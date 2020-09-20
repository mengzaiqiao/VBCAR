## Our VBCAR model have been integrated to the [Beta-Recsys](https://beta-recsys.readthedocs.io/en/latest/notes/models.html) framework.


This repository contains the Python implementation for VBCAR. Further details about VBCAR can be found in our paper:

## Variational Bayesian Context-aware Representation for Grocery Recommendation
>Meng, Zaiqiao, Richard McCreadie, Craig Macdonald, and Iadh Ounis. "Variational Bayesian Context-aware Representation for Grocery Recommendation." arXiv preprint arXiv:1909.07705 (2019).

## Introduction
Grocery recommendation is an important recommendation use-case, which aims to predict which items a user might choose to buy in the future, based on their shopping history. However, existing methods only represent each user and item by single deterministic points in a low-dimensional continuous space. In addition, most of these methods are trained by maximizing the co-occurrence likelihood with a simple Skip-gram-based formulation, which limits the expressive ability of their embeddings and the resulting recommendation performance. In this paper, we propose the Variational Bayesian Context-Aware Representation (VBCAR) model for grocery recommendation, which is a novel variational Bayesian model that learns the user and item latent vectors by leveraging basket context information from past user-item interactions. We train our VBCAR model based on the Bayesian Skip-gram framework coupled with the amortized variational inference so that it can learn more expressive latent representations that integrate both the non-linearity and Bayesian behaviour. Experiments conducted on a large real-world grocery recommendation dataset show that our proposed VBCAR model can significantly outperform existing state-of-the-art grocery recommendation methods.

## Requirements
=================
* pytorch (1.1.0=py3.6_cuda10.0.130_cudnn7.5.1_0)
* python 3.6
* scikit-learn
* scipy
Detail package dependencies can be found at myenv.yml

## Run the demo
=================
### console demo
```bash
python main.py
```
### jupyter notebook demo
* VBCR_example_random_feature.ipynb

>  If there is any issue with our codes or model, please don't hesitate to let us know.


## Citation

If you want to use our codes and datasets in your research, please cite:

```shell
@article{meng2019variational,
  title={Variational Bayesian Context-aware Representation for Grocery Recommendation},
  author={Meng, Zaiqiao and McCreadie, Richard and Macdonald, Craig and Ounis, Iadh},
  journal={arXiv preprint arXiv:1909.07705},
  year={2019}
}
```

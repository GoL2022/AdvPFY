# Adversarial Purification with the Manifold Hypothesis (AAAI 2024)

> **Repository status:** demo / under reconstruction

This repository contains demo code and notebooks for "Adversarial Purification with the Manifold Hypothesis" (AAAI 2024).

Contents
- Notebooks
  - [training_demo.ipynb](training_demo.ipynb) — training/demo on FashionMNIST. Trains and saves:
    - VAE classifier: ./model/vae_clf.pth
    - ST-AE classifier: ./model/stae_clf.pth
  - [purify_attack_demo.ipynb](purify_attack_demo.ipynb) — runs PGD attacks and purifies adversarial examples using the purify routines; visualizes reconstructions and purified examples.
  - [CIFAR10_ResNet.ipynb](CIFAR10_ResNet.ipynb) — end-to-end CIFAR-10 demo using a ResNet encoder and ResNet-VAE; trains and attack the models with PGD attack.

- Scripts / modules
  - [pgd_purify.py](pgd_purify.py) — attack and purification utilities:
    - [`pgd_purify.pgd_linf`](pgd_purify.py) — PGD L_inf attacker
    - [`pgd_purify.vae_purify`](pgd_purify.py) — purification using the VAE classifier ELBO
    - [`pgd_purify.stae_purify`](pgd_purify.py) — purification using standard autoencoder reconstruction loss
  - [model/nn_model.py](model/nn_model.py) — model definitions:
    - [`model.nn_model.VAEClassifier`](model/nn_model.py)
    - [`model.nn_model.StAEClassifier`](model/nn_model.py)
    - [`model.nn_model.ResNetEnc`](model/nn_model.py)
    - [`model.nn_model.ResNetVAE`](model/nn_model.py)

Quick start
1. Install requirements 
```sh 
  pip install -r requirements.txt
```
2. Open and run:
   - For demo on FashionMNIST: [training_demo.ipynb](training_demo.ipynb) → then [purify_attack_demo.ipynb](purify_attack_demo.ipynb).
   - For end-to-end CIFAR-10 + ResNet demo: [CIFAR10_ResNet.ipynb](CIFAR10_ResNet.ipynb).
3. Use the purify functions from the notebooks or programmatically:
   - import and call [`pgd_purify.vae_purify`](pgd_purify.py) / [`pgd_purify.stae_purify`](pgd_purify.py) to purify adversarial batches.
   - use [`pgd_purify.pgd_linf`](pgd_purify.py) to generate adversarial examples.

Examples (notebook usage)
- training_demo.ipynb demonstrates joint training of the VAE classifier and standard AE with ELBO and cross-entropy.
- purify_attack_demo.ipynb shows:
  - loading saved models (./model/vae_clf.pth, ./model/stae_clf.pth),
  - generating adversarial samples with [`pgd_purify.pgd_linf`](pgd_purify.py),
  - purifying them with [`pgd_purify.vae_purify`](pgd_purify.py) and [`pgd_purify.stae_purify`](pgd_purify.py),
  - visualizing reconstructions and purified images.


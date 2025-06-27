# Gaussian Mixture Variational Auto-Encoder

Gaussian-mixture variational auto-encoder [^1], or GM-VAE for short, can be used for deep unsupervised clustering. For example, given the MNIST dataset, GM-VAE can automatically learn to group the images into clusters, each of which contains (mostly) images with the same digit.

For interested readers, this repository provides a fast (re-)implementation of gaussian mixture variational auto-encoder in [JAX](https://docs.jax.dev/en/latest/index.html). One could easily train the model for 1k epochs with around 20 minutes on an RTX4090 GPU.

With suitable choices of hyperparameters, after training for 1k epochs, the model could achieve 87.76% clustering accuracy (purity), which is on par with the numbers reported in the original blog post.

[^1]: Gaussian Mixture VAE, https://ruishu.io/2016/12/25/gmvae/

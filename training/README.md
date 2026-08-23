# Training Workspace

From-scratch / non-LoRA training pipelines (see the root README and
`ARCHITECTURE.md` for the full workspace map):

- [adult-income-logreg](./adult-income-logreg/) — logistic regression from scratch (raw numpy) on UCI Adult / Census Income
- [cifar10-vqvae](./cifar10-vqvae/) — VQ-VAE from scratch (torch) on CIFAR-10
- [fashion-mnist-dcgan](./fashion-mnist-dcgan/) — DCGAN from scratch (torch, hand-written conv nets + label smoothing) on Fashion-MNIST
- [flow-matching-mnist](./flow-matching-mnist/) — flow matching / rectified flow from scratch (torch, hand-written UNet + ODE sampler) on MNIST
- [imdb-sentiment-cnn](./imdb-sentiment-cnn/) — Text CNN from scratch (torch, random embeddings) on the Large Movie Review Dataset
- [mnist-kmeans](./mnist-kmeans/) — k-means from scratch (raw numpy) on MNIST
- [mnist-vae](./mnist-vae/) — VAE from scratch (torch) on MNIST
- [rvq-audio-codec](./rvq-audio-codec/) — neural audio codec with residual vector quantization from scratch (torch, hand-written SEANet encoder/decoder + RVQ + multi-scale STFT discriminator) on LJSpeech
- [vit-cifar10](./vit-cifar10/) — Vision Transformer from scratch (torch, hand-written patch embed + pre-LN blocks + multi-head attention) on CIFAR-10

# Training Workspace

From-scratch / non-LoRA training pipelines (see the root README and
`ARCHITECTURE.md` for the full workspace map):

- [adult-income-logreg](./adult-income-logreg/) — logistic regression from scratch (raw numpy) on UCI Adult / Census Income
- [cifar10-vqvae](./cifar10-vqvae/) — VQ-VAE from scratch (torch) on CIFAR-10
- [dit-cifar100](./dit-cifar100/) — class-conditional DiT (Sora-style diffusion transformer, hand-written adaLN-Zero blocks + conditional-OT flow matching + classifier-free guidance) from scratch (torch) on CIFAR-100
- [fashion-mnist-dcgan](./fashion-mnist-dcgan/) — DCGAN from scratch (torch, hand-written conv nets + label smoothing) on Fashion-MNIST
- [flow-matching-mnist](./flow-matching-mnist/) — flow matching / rectified flow from scratch (torch, hand-written UNet + ODE sampler) on MNIST
- [imdb-sentiment-cnn](./imdb-sentiment-cnn/) — Text CNN from scratch (torch, random embeddings) on the Large Movie Review Dataset
- [mae-cifar100](./mae-cifar100/) — Masked Autoencoder from scratch (torch, vit-cifar10's hand-written patch-embed/block encoder + mask token + lightweight decoder, judged by a hand-written linear probe) on CIFAR-100
- [mnist-kmeans](./mnist-kmeans/) — k-means from scratch (raw numpy) on MNIST
- [mnist-vae](./mnist-vae/) — VAE from scratch (torch) on MNIST
- [rvq-audio-codec](./rvq-audio-codec/) — neural audio codec with residual vector quantization from scratch (torch, hand-written SEANet encoder/decoder + RVQ + multi-scale STFT discriminator) on LJSpeech
- [vit-cifar10](./vit-cifar10/) — Vision Transformer from scratch (torch, hand-written patch embed + pre-LN blocks + multi-head attention) on CIFAR-10

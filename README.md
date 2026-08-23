# Training Workspace

![Python](https://img.shields.io/badge/python-3.10%2B-blue) ![License: MIT](https://img.shields.io/badge/license-MIT-blue)

Model Training, Pre-Training and Fine-Tuning workspace, scaling from a single RTX 3090 (24GB).

## Repository

- [training](./training/)
    - Train RVQ Audio Codec on [LJSpeech-1.1](https://keithito.com/LJ-Speech-Dataset/) dataset
    - Train UNet (Flow Matching) on [MNIST](www.kaggle.com/datasets/hojjatk/mnist-dataset) dataset
    - Train DCGAN on [Fashion-MNIST](https://www.kaggle.com/datasets/zalando-research/fashionmnist)
    - Train TextCNN on [Large Movie Review](https://ai.stanford.edu/~amaas/data/sentiment/) dataset
    - Train VQ-VAE on [CIFAR-10](https://cave.cs.toronto.edu/kriz/cifar.html) dataset
    - Train VAE on [MNIST](www.kaggle.com/datasets/hojjatk/mnist-dataset) dataset
    - Train K-Means on [MNIST](www.kaggle.com/datasets/hojjatk/mnist-dataset) dataset
    - Train Logistic Regression on [uci.edu/adult](https://archive.ics.uci.edu/dataset/2/adult) dataset
- [fine-tuning](./fine-tuning/)
    - Fine-tune [Qwen2.5-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct) on [CNN/DailyMail](https://huggingface.co/datasets/abisee/cnn_dailymail) dataset
    - Fine-tune [Vicuna-7b-v1.5](https://huggingface.co/lmsys/vicuna-7b-v1.5) on [CNN/DailyMail](https://huggingface.co/datasets/abisee/cnn_dailymail) dataset
- [pre-training](./pre-training/)
    - OCR PNG pages with [Surya OCR](https://github.com/datalab-to/surya)
    - Summarize OCR text with ([unsloth/gemma-3-4b-it](https://huggingface.co/unsloth/gemma-3-4b-it))
    - Describe page layout/structure image-grounded with ([unsloth/gemma-3-4b-it](https://huggingface.co/unsloth/gemma-3-4b-it))
    - Generate synthetic QA pairs from OCR text with ([unsloth/gemma-3-4b-it](https://huggingface.co/unsloth/gemma-3-4b-it))


## Datasets

Text Datasets:

- [abisee/cnn_dailymail](https://huggingface.co/datasets/abisee/cnn_dailymail)
    - text/summary (312k rows)
    - The CNN / DailyMail Dataset is an English-language dataset containing just over 300k unique news articles as written by journalists at CNN and the Daily Mail. The current version supports both extractive and abstractive summarization. 

- [CIFAR-10](https://cave.cs.toronto.edu/kriz/cifar.html)
    - The CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes, with 6000 images per class. There are 50000 training images and 10000 test images. 

- [Large Movie Review Dataset](https://ai.stanford.edu/~amaas/data/sentiment/)
    - A binary sentiment classification benchmark introduced in Maas et al., ACL 2011. Also known as aclImdb or simply IMDB. 
    - Contains 50,000 polarized movie reviews scraped from IMDB - 25,000 for training and 25,000 for testing.

- [uci.edu/adult](https://archive.ics.uci.edu/dataset/2/adult)
    - tabular classification - Predict whether annual income of an individual exceeds $50K/yr based on census data. Also known as "Census Income" dataset. 
    - DOI: 10.24432/C5XW20

- [MNIST](www.kaggle.com/datasets/hojjatk/mnist-dataset)
    - The MNIST database of handwritten digits has a training set of 60,000 examples, and a test set of 10,000 examples.

- [LJSpeech-1.1](https://keithito.com/LJ-Speech-Dataset/)
    - This is a public domain speech dataset consisting of 13,100 short audio clips of a single speaker reading passages from 7 non-fiction books. A transcription is provided for each clip. Clips vary in length from 1 to 10 seconds and have a total length of approximately 24 hours. 

- [Fashion-MNIST](https://www.kaggle.com/datasets/zalando-research/fashionmnist)
    - Fashion-MNIST is a dataset of Zalando's article images—consisting of a training set of 60,000 examples and a test set of 10,000 examples. Each example is a 28x28 grayscale image, associated with a label from 10 classes.


## Commands

```sh
# --- DCGAN
uv run --directory training/fashion-mnist-dcgan python build_fashion_mnist_dataset.py --data-dir "C:\path\to\fashionmnist" --output-dir data
uv run --directory training/fashion-mnist-dcgan python train_dcgan.py --data-path data/fashion_mnist.npz --num-epochs 50 --batch-size 128 --output-dir runs/fashion_mnist_dcgan
uv run --directory training/fashion-mnist-dcgan python evaluate_dcgan.py --data-path data/fashion_mnist.npz --checkpoint-path runs/fashion_mnist_dcgan/dcgan_epoch0050.pt --output-dir runs/fashion_mnist_dcgan

# --- Residual Vector Quantization (RVQ) in audio codec
uv run --directory training/rvq-audio-codec python build_ljspeech_dataset.py --data-dir "C:\path\to\ljspeech-dataset" --output-dir data
uv run --directory training/rvq-audio-codec python -u train_codec.py --data-dir data --num-epochs 60 --batch-size 32 --output-dir runs/ljspeech_codec
uv run --directory training/rvq-audio-codec python -u evaluate_codec.py --data-dir data --checkpoint-path runs/ljspeech_codec/codec_best.pt --output-dir runs/ljspeech_codec

# --- training/flow-matching-mnist
uv run --directory training/flow-matching-mnist python build_mnist_dataset.py --data-dir "C:\path\to\mnist-dataset" --output-dir data
uv run --directory training/flow-matching-mnist python train_flow.py --data-path data/mnist.npz --base-channels 32 --num-epochs 40 --batch-size 128 --output-dir runs/mnist_flow
uv run --directory training/flow-matching-mnist python evaluate_flow.py --data-path data/mnist.npz --checkpoint-path runs/mnist_flow/flow_best.pt --num-steps 50 --output-dir runs/mnist_flow

# --- training/cifar10-vqvae: custom VQ-VAE from scratch (torch autograd, EMA codebook) on CIFAR-10 ---
uv run --directory training/cifar10-vqvae python build_cifar10_dataset.py --data-dir "C:\path\to\cifar-10-python" --output-dir data
uv run --directory training/cifar10-vqvae python train_vqvae.py --data-path data/cifar10.npz --codebook-size 512 --embedding-dim 64 --commitment-beta 0.25 --num-epochs 100 --batch-size 128 --output-dir runs/cifar10_vqvae
uv run --directory training/cifar10-vqvae python evaluate_vqvae.py --data-path data/cifar10.npz --checkpoint-path runs/cifar10_vqvae/vqvae_best.pt --output-dir runs/cifar10_vqvae

# --- training/mnist-vae: custom convolutional VAE from scratch (torch autograd) on MNIST ---
uv run --directory training/mnist-vae python build_mnist_dataset.py --data-dir "C:\path\to\mnist-dataset" --output-dir data
uv run --directory training/mnist-vae python train_vae.py --data-path data/mnist.npz --latent-dim 32 --beta 1.0 --num-epochs 30 --batch-size 128 --output-dir runs/mnist_vae
uv run --directory training/mnist-vae python evaluate_vae.py --data-path data/mnist.npz --checkpoint-path runs/mnist_vae/vae_best.pt --output-dir runs/mnist_vae

# --- training/mnist-kmeans: k-means from scratch (raw numpy) on raw-pixel MNIST ---
uv run --directory training/mnist-kmeans python build_mnist_dataset.py --data-dir "C:\path\to\mnist-dataset" --output-dir data
uv run --directory training/mnist-kmeans python train_kmeans.py --data-path data/mnist.npz --k 10 --num-iters 50 --output-dir runs/mnist_kmeans
uv run --directory training/mnist-kmeans python evaluate_kmeans.py --data-path data/mnist.npz --centroids-path runs/mnist_kmeans/centroids.npz --output-dir runs/mnist_kmeans

# --- training/adult-income-logreg: logistic regression from scratch (raw numpy) ---
# no scikit-learn/pandas - sigmoid, cross-entropy loss, and gradient descent written by hand
uv run --directory training/adult-income-logreg python build_income_dataset.py --data-dir "C:\path\to\adult" --output-dir data
uv run --directory training/adult-income-logreg python train_logreg.py --data-path data/adult_income.npz --num-epochs 300 --output-dir runs/adult_logreg
uv run --directory training/adult-income-logreg python evaluate_logreg.py --data-path data/adult_income.npz --weights-path runs/adult_logreg/logreg_weights.npz

# --- training/imdb-sentiment-cnn: Text CNN from scratch (torch, random embeddings) on the Large Movie Review Dataset ---
# binary sentiment classification (pos/neg);
uv run --directory training/imdb-sentiment-cnn python build_imdb_dataset.py --data-dir "C:\path\to\aclImdb_v1" --output-dir data
uv run --directory training/imdb-sentiment-cnn python train_cnn.py --data-path data/imdb.npz --num-epochs 20 --batch-size 128 --output-dir runs/imdb_cnn
uv run --directory training/imdb-sentiment-cnn python evaluate_cnn.py --data-path data/imdb.npz --checkpoint-path runs/imdb_cnn/cnn_best.pt --vocab-path data/vocab.txt --output-dir runs/imdb_cnn

# --- pre-training: PDF corpus -> OCR/summary/layout/QA CSVs ---
pre-training\exec_1.bat

# --- fine-tuning/vicuna-7b-lora: Vicuna-7B LoRA (transformers + peft) ---
# text summarization LoRA, loads lmsys/vicuna-7b-v1.5 directly
uv run --directory fine-tuning/vicuna-7b-lora python build_vicuna7b_dataset.py --cnn-dailymail-dir "C:\path\to\cnn_dailymail\3.0.0" --max-samples 2000
uv run --directory fine-tuning/vicuna-7b-lora python train_vicuna7b_lora.py --num-epochs 2 --output-dir runs/vicuna7b_lora
uv run --directory fine-tuning/vicuna-7b-lora python generate_vicuna7b_lora.py --adapter-dir runs/vicuna7b_lora/final_adapter --jsonl-eval data/vicuna7b_train.jsonl --num-samples 5

# --- fine-tuning/qwen25-3b-lora: Qwen2.5-3B LoRA (transformers + peft) ---
# same pattern as vicuna-7b-lora, ChatML prompt format
uv run --directory fine-tuning/qwen25-3b-lora python build_qwen3b_dataset.py --cnn-dailymail-dir "C:\path\to\cnn_dailymail\3.0.0" --max-samples 2000
uv run --directory fine-tuning/qwen25-3b-lora python train_qwen3b_lora.py --num-epochs 2 --output-dir runs/qwen3b_lora
uv run --directory fine-tuning/qwen25-3b-lora python generate_qwen3b_lora.py --adapter-dir runs/qwen3b_lora/final_adapter --jsonl-eval data/qwen3b_train.jsonl --num-samples 5
```

## License

Licensed under the [MIT License](./LICENSE).

# Transformer Model in PyTorch for Single-GPU Training

This repository provides a single file implementation of a Transformer model using PyTorch, optimized for single-GPU environments. The model includes input embeddings, positional encoding, layer normalization, and feed-forward blocks.

## Table of Contents
- [Introduction](#introduction)
- [Model Architecture](#model-architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Introduction

This project implements a Transformer architecture in PyTorch designed to be trained efficiently on a single GPU. It handles large datasets by utilizing optimizations like dropout, layer normalization, and positional encodings to make training more feasible in resource-limited environments.

## Model Architecture

### Components in One File:
- **Input Embeddings**: Converts tokens into high-dimensional vectors.
- **Positional Encoding**: Adds position information to embeddings using sine and cosine functions.
- **Layer Normalization**: Normalizes input across layers for stability.
- **Feed Forward Block**: A simple two-layer neural network with ReLU activation and dropout.

The single file contains the full implementation of these components in PyTorch.

## Installation

1. Clone the repository:
   ```bash
   https://github.com/Archit03/Transformer
   cd transformer

---
title: "SUMO"
summary: "Scaling ViTs and ConvNets for Full Self-Driving"
toc: false
readTime: true
autonumber: false
showTags: false
hideBackToTop: true
breadcrumbs: true
date: "2021-10-30"
---

# Introduction

This project is about scaling Vision Transformers (ViTs) and Convolutional Neural Networks (ConvNets) for full self-driving.

The goal was to learn how to do distributed, fault-tolerant, and scalable training of large models on a large GPU cluster comprising of `~32` nodes or roughly `256x A100-80GB` GPUs using Tensorflow & PyTorch, as well as [`SLURM`](https://slurm.schedmd.com/quickstart.html).

## Dataset

I used the [`BDD100K`](http://bdd-data.berkeley.edu/) dataset for training the model. The dataset is a diverse driving video dataset with over 100K videos, totalling around `2TB` of data. For efficiency, I used a `TFRecord` based pipeline for easy serialization and efficient distributed deserialization of the dataset.

The dataset was stored on AWS buckets, and was accessed using the `s3fs`. `ray` was used for parallelizing the data preprocessing step.

## Model

I used the [`ConvNext`](https://arxiv.org/abs/2201.03545) model as the base model for this project. It managed to outperform the ViT baselines through little hyperparmeter tuning, and was more more lightweight in terms of FLOPs as well as memory usage.

For containerization, I used `Docker` and `Singularity` for creating the container images. The images were then pushed to the Singularity cloud build where they could be cached and downloaded whenever a training run commences.

## Result

This is a sample predicted trajectory of the model on a validation sample:
  
<video height=700 width=750 controls>
  <source src="sample.mp4" type="video/mp4">
</video>
A Python Implementation of Recurrent Neural Network (RNN)
=========================================================

Overview
--------

This repo is the Python implementation of my personal blog
[Introduction to Recurrent Neural Networks (RNNs)](https://leadership.qubitpi.org/posts/rnn/).

The implementation start by summarizing what we concluded at the end of the blog. Given an input of size $\tau > 0$, we
train a model that produces an output of the same size, i.e. $\tau > 0$; the forward propagation of this model is
defined by

$$
    \begin{align*}
        & \boldsymbol{h}^{(t)} = \tanh\left( \boldsymbol{W_{hh}}h^{(t - 1)} + \boldsymbol{W_{xh}}x^{(t)} + \boldsymbol{b_h} \right) \\
        & \boldsymbol{o}^{(t)} = \boldsymbol{W_{yh}}\boldsymbol{h}^{(t)} + \boldsymbol{b_o} \\
        & \boldsymbol{\hat{y}^{(t)}} = softmax(\boldsymbol{o}^{(t)})
    \end{align*}
$$

where

- $\boldsymbol{h}^{(t)}$ is the hidden state vector of size $(\tau - 1)$
- $\boldsymbol{o}^{(t)}$ is the output produced by the model at step $t$ where $t \in \{1, 2, \cdots, \tau\}$
- $\boldsymbol{\hat{y}^{(t)}}$ is the normalized probability of $\boldsymbol{o}^{(t)}$ at $\tau = t$
- $\boldsymbol{b_h}$ is the hidden bias vector of size $\tau$
- $\boldsymbol{b_o}$ is the output bias vector of size $\tau$
- the size of $\boldsymbol{W_{xh}}$ is $(\tau - 1) \times \tau$
- the size of $\boldsymbol{W_{hh}}$ is $(\tau - 1) \times (\tau - 1)$
- the size of $\boldsymbol{W_{xh}}$ is $\tau \times (\tau - 1)$

The loose function of this model is

$$
    \mathcal{L}\left( \{ \boldsymbol{x}^{(1)}, ..., \boldsymbol{x}^{(\tau)} \}, \{ \boldsymbol{y}^{(1)}, ..., \boldsymbol{y}^{(\tau)} \} \right) = \sum_t^{\tau} \mathcal{L}^{(t)} = -\sum_t^{\tau}\log\boldsymbol{\hat{y}}^{(t)}
$$

The update rules with a learning rate of $\eta$ are given by

$$
    \begin{align*}
        & \Delta W_{xh} = -\eta\sum_{t = 1}^\tau \left( diag\left[ 1 - (\boldsymbol{h}^{(t)})^2 \right] \right)^\intercal \nabla_{\boldsymbol{h}^{(t)}}\mathcal{L} \left( \boldsymbol{x}^{(t)} \right) \\ \\
        & \Delta W_{hh} = -\eta\sum_{t = 1}^\tau diag\left[ 1 - \left(\boldsymbol{h}^{(t)}\right)^2 \right] \left( \nabla_{\boldsymbol{h}^{(t)}}\mathcal{L} \right) {\boldsymbol{h}^{(t - 1)}}^\intercal \\ \\
        & \Delta W_{yh} = -\eta\sum_{t = 1}^\tau \left( \boldsymbol{\sigma} - \boldsymbol{p} \right) \boldsymbol{h}^{(t)} \\ \\
        & \Delta b_h = -\eta\sum_{t = 1}^\tau \left( diag\left[ 1 - (\boldsymbol{h}^{(t)})^2 \right] \right)^\intercal \nabla_{\boldsymbol{h}^{(t)}}\mathcal{L} \\ \\
        & \Delta b_o = -\eta\sum_{t = 1}^\tau \boldsymbol{\sigma} - \boldsymbol{p}
    \end{align*}
$$

where

$$
    \nabla_{\boldsymbol{h}^{(t)}}\mathcal{L} = \left( \boldsymbol{W_{yh}} \right)^\intercal \nabla_{\boldsymbol{o}^{(t)}}\mathcal{L} + \boldsymbol{W_{hh}}^\intercal \nabla_{\boldsymbol{h}^{(t + 1)}}\mathcal{L} \left( diag\left[ 1 - (\boldsymbol{h}^{(t + 1)})^2 \right] \right)
$$

Everything is intuitive implementation-wise, except for the $\nabla_{\boldsymbol{h}^{(t)}}\mathcal{L}$.
Specifically, we need to discuss the computations of

- $\nabla_{\boldsymbol{o}^{(t)}}\mathcal{L}$, and
- $\nabla_{\boldsymbol{h}^{(t + 1)}}\mathcal{L}$.

$$
    \nabla_{\boldsymbol{o}^{(t)}}\mathcal{L} = -\nabla_{\boldsymbol{o}^{(t)}}\sum_t^{\tau}\log\boldsymbol{\hat{y}}^{(t)} = -\nabla_{\boldsymbol{o}^{(t)}}\sum_t^{\tau}\log\sigma(\boldsymbol{o^{(t)}}) = -\sum_t \nabla_{\boldsymbol{o}^{(t)}}\log\sigma(\boldsymbol{o}^{(t)}) = -\sum_t^{\tau}\frac{1}{\sigma(\boldsymbol{o}^{(t)})}\left( -\sigma(\boldsymbol{o}^{(t)})\sigma(\boldsymbol{o}^{(t)}) \right) = \sum_t^{\tau}\sigma(\boldsymbol{o^{(t)}})
$$

$$
    \begin{align}
        \nabla_{\boldsymbol{h}^{(t + 1)}}\mathcal{L} &= -\nabla_{\boldsymbol{h}^{(t + 1)}}\sum_t^{\tau}\log\boldsymbol{\hat{y}}^{(t)} \\
        & = -\nabla_{\boldsymbol{h}^{(t + 1)}}\sum_t^{\tau}\log\sigma(\boldsymbol{o^{(t)}}) \\
        & = -\sum_t^{\tau}\nabla_{\boldsymbol{h}^{(t + 1)}}\log\sigma(\boldsymbol{o^{(t)}}) \\
        & = -\sum_t^{\tau}\frac{1}{\sigma(\boldsymbol{o^{(t)}})}\frac{\partial \sigma(\boldsymbol{o^{(t)}})}{\partial \boldsymbol{h}^{(t + 1)}} \\
        & = \sum_t^{\tau}\frac{1}{\sigma(\boldsymbol{o^{(t)}})}\sigma(\boldsymbol{o^{(t)}})^2W_{yh} \\
        & = \sum_t^{\tau}\sigma(\boldsymbol{o^{(t)}})W_{yh} \\
    \end{align}
$$

Therefore

$$
    \begin{align}
    \nabla_{\boldsymbol{h}^{(t)}}\mathcal{L} &= \left( \boldsymbol{W_{yh}} \right)^\intercal \sum_t^{\tau}\sigma(\boldsymbol{o^{(t)}}) + \boldsymbol{W_{hh}}^\intercal \sum_t^{\tau}\sigma(\boldsymbol{o^{(t)}})W_{yh} \left( diag\left[ 1 - (\boldsymbol{h}^{(t + 1)})^2 \right] \right) \\
    & = \sum_t^{\tau}\left[ \left( \boldsymbol{W_{yh}} \right)^\intercal \sigma(\boldsymbol{o^{(t)}}) + \boldsymbol{W_{hh}}^\intercal \sigma(\boldsymbol{o^{(t)}})W_{yh} \left( diag\left[ 1 - (\boldsymbol{h}^{(t + 1)})^2 \right] \right) \right]
    \end{align}
$$
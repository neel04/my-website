---
title: "ASURA"
summary: "Asymptotically Universal Recursive Architecture"
toc: true
readTime: true
autonumber: false
showTags: false
hideBackToTop: true
breadcrumbs: true
date: "2025-10-10"
math: true
---

# Abstract

**ASURA** is a simple, recursive variant of Universal Transformers [^3] aimed at language modeling with improved stability and scalability. It applies a shared block across depth (recursion over iterations), augments it with long skip connections and extra normalizations, obtaining strong performance while preserving the same relative FLOPs efficiency.

## Motivation

- Scaling alone is insufficient: depth and iterative computation matter for tasks requiring multi-step reasoning and composition.

- Externalized chains (CoT) are brittle and can be unfaithful; latent-space computation with recursion avoids compounding token-level errors.

- Parameter sharing over depth offers efficiency and generalization potential, but requires stability to scale to realistic corpora.

- Conventional UTs (Universal Transformers) however can be unstable, can diverge at scale and often don't utilize each recursive iteration.

## Related Work

- **Recursive/parameter‑shared architectures**. Universal Transformers add depth‑wise sharing and ACT [^3]; Deep‑Equilibrium Networks adopt a fixed‑point, “infinite‑depth” view [^4]. Neural GPU demonstrates algorithmic generalization with tricks applied to convnets [^5], while DTNet/Deep‑Thinking highlights the utility of the `recall` mechanism for enabling stability and OOD length extrapolation [^6].

- **Adaptive computation**. Chain‑of‑Thought prompting increases output length to approximate extra compute [^7], but raises faithfulness and attribution concerns [^8] [^11] and can aggravate exposure‑bias errors in autoregression [^9]. Internal recursion (parameter sharing + iterative refinement) keeps compute in the latent space and, with the right normalization/skips, can be more stable.

- **Length/OOD generalization**. Vanilla transformers struggle to extrapolate to longer contexts under standard positional encodings [^10]. Recursive inductive biases aids in length generalization, but is often not easily parallelizable or scalable.

# Method

## Notation

We begin by introducing some notation. The model consumes a sequence of $x_i \in \mathcal{D}_n$ where $\mathcal{D}_n$ denotes the dataset/corpus.
A sequence for the model is thus $(x_0, ..., x_n)$, producing intermediate latent state denoted $(z_0, ..., z_n)$ and mapping to an output $(y_0, ..., y_n)$ where $n$ is the sequence length.
We use the standard $W_i \in \mathbb{F}^{m \times n}$ notation for denoting an $m$ by $n$ matrix in the field $F = \mathbb{R}$, at the $i$-th index.

The concatenation of $n$ arrays $A_0, \ldots, A_n$ along their corresponding last dimensions is denoted by the concatenation operator $\bigoplus: (\mathbb{F}^{i \times k} \times \mathbb{F}^{i \times k} \times ...) \rightarrow \mathbb{F}^{i \times nk}$.

Thus, we can write:

$$
A_{\text{concat}} = A_0 \bigoplus A_1 \bigoplus \ldots \bigoplus A_n = \bigoplus_{i=1}^n A_i
$$

## Base Architecture

We're using a [NanoGPT](https://github.com/karpathy/nanoGPT)/[Transformer++](https://arxiv.org/abs/2003.04974)-styled transformer.

Let a sequence be $X = (x_0,\ldots,x_n)$, hidden states $Z = (z_0,\ldots,z_n)$, outputs $Y = (y_0,\ldots,y_n)$, with embeddings in $\mathbb{R}^{n\times d}$. Self-attention (single head) on $X$:

$$
Q = X W_q,\quad K = X W_k,\quad V = X W_v,\quad W_q,W_k,W_v \in \mathbb{R}^{d\times d}
$$

$$
\operatorname{Attention}(Q,K,V) = \operatorname{softmax}\left(\frac{QK^\top}{\sqrt{d}}\right) V
$$

Where a block is defined as (pre-norm, concise form):

$$
\mathcal{B}(x;\theta) = \operatorname{LayerNorm}\big(\operatorname{MLP}(\operatorname{MHA}(\operatorname{LayerNorm}(x)))\big)
$$

and borrow the standard GeLU activation:

$$
\sigma_{\text{GeLU}}(x) = \tfrac{1}{2} \, x \, \Big(1 + \tanh\big(\sqrt{2/\pi}\,(x + 0.044715\,x^3)\big)\Big)
$$


## Input Injection

We introduce a projected long-skip to stabilize deep recursion and improve gradient flow. For tensors $X_{0},\ldots,X_{n}$ with compatible last dims and a projection $W_{\mathrm{proj}} \in \mathbb{R}^{n k\times k}$:

$$
\operatorname{ProjConcat}(X_{0},\ldots,X_{n}) = \Big( \bigoplus_{i=0}^{n} X_{i} \Big) W_{\mathrm{proj}}.
$$

![ProjConcat: project-and-concatenate long skip](asura_concat_proj.drawio.svg#full "ProjConcat: concatenate residuals, then project with W_proj.")

Where $X_i$ is the $i$-th residual of the network. In practice:

$$
X_{\mathrm{in}} = \operatorname{LayerNorm}(\operatorname{Embed}(X_{0}))
$$

and we combine $X_{\mathrm{in}}$ with the previous iteration’s latent $X_{i-1}$.


## ASURA

We're now at a position to express our proposed $\textit{ASURA}$ architecture. We construct our architecture similar to that of DTNet/UT [^3] [^6], wherein we have a set of Attention blocks which are recursively applied $i$-times to the input.

We can define a stack of $L$ blocks/layers $\textit{ASURA} = [\mathcal{B}(\theta_{0}),\ldots,\mathcal{B}(\theta_{L})]$.

Applying it recursively for $i$ iterations on some input $x$, the `fwd` pass looks like:

$$
\begin{align*}
  \textit{ASURA}(x) = (\mathcal{B}(\theta_0) \circ \mathcal{B}(\theta_1) \circ ... \mathcal{B}(\theta_L))(x) \newline
\end{align*}
$$

Where $\{ \theta_i \in \Theta : \forall i, j:  \theta_i \neq \theta_j \}$ are indexed parameter vectors, wherein And we denote parameter-shared, depthwise recurrence via:

$$
\begin{align*}
  \hat{X}_{i + 1} = X_{1} + \textit{ASURA}\( \hat{X}_i \) 
\end{align*}
$$

However, this naive formulation is unstable. Below we present tricks 

## Decoupled `LayerNorm`


![Decoupled/Unshared Layernorms](asura_unshared_LN.drawio.svg#full "Ensure each iteration is normalized uniquely.")

## Notes on Efficiency

- Backprop through \(i\) iterations increases memory roughly with context length, batch size, hidden dim, and layers. We use activation checkpointing/scan-style rematerialization to trade compute for memory (treeverse checkpointing with known horizon \(i\)).

## Status

- Draft, concise blog version derived from LaTeX. Further edits to tighten scope, add ablations, and include figures are planned.

## References

[^1]: Vaswani et al. “Attention Is All You Need.” 2017. https://arxiv.org/abs/1706.03762

[^2]: Radford et al. “Improving Language Understanding by Generative Pre‑Training.” 2018. https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf

[^3]: Dehghani et al. “Universal Transformers.” 2018. https://arxiv.org/abs/1807.03819

[^4]: Bai et al. “Deep Equilibrium Models.” 2019. https://arxiv.org/abs/1909.01377

[^5]: Kaiser & Sutskever. “Neural GPU.” 2015. https://arxiv.org/abs/1511.08228

[^6]: Bansal & Schwarzschild. “Deep Thinking Networks / DTNet.” 2022. https://arxiv.org/abs/2202.05826

[^7]: Wei et al. “Chain‑of‑Thought Prompting.” 2022. https://arxiv.org/abs/2201.11903

[^8]: Lanham et al. “On the Faithfulness of Chain‑of‑Thought.” 2023. https://arxiv.org/abs/2307.13702

[^9]: Bengio et al. “Scheduled Sampling for Sequence Prediction with Recurrent Neural Networks.” 2015. https://arxiv.org/abs/1506.03099

[^10]: Discussion on length extrapolation and positional encoding limitations (e.g., RoPE/PI limits). 2023. https://arxiv.org/abs/2305.19466

[^11]: Reasoning models don't always say what they think. 2025. https://www.anthropic.com/research/reasoning-models-dont-say-think

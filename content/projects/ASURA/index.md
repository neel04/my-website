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

## Abstract

**ASURA** is a simple, recursive variant of Universal Transformers [^3] aimed at language modeling with improved stability and scalability. It applies a shared block across depth (recursion over iterations), augments it with long skip connections and extra normalizations, obtaining strong performance while preserving the same relative FLOPs cost.

## Motivation

While the resurgence of interest in recursive architectures is warranted, current literature still lacks a rigorous treatment of recursion strategies. Industry and academia often presume that repeatedly applying the network $n$ times is sufficient. We see no reason to believe that naive recursion is optimal.

This prompts the question: why should we focus on recursive architectures in the first place?

- Parameter scaling in isolation is insufficient; robust multi-step reasoning and compositional inference demand greater depth and iterative computation - in line with the serial hypothesis [^14].

- Externalized chains (CoT) are brittle and can be unfaithful [^15]; latent-space computation with recursion avoids compounding token-level errors and allows greater bandwidth for information transfer.

- Parameter sharing over depth offers *efficiency* and **generalization** potential, but requires stability to scale to realistic corpora.

- The holy grail however, is the ability to convert any UT to be able to perform arbitrary, unbounded **inference-time compute** natively in the latent space.

Conventional UTs (Universal Transformers) can be unstable, can diverge at scale and often don't utilize each recursive iteration to their full potential. In this work, we develop tricks to alleviate such problems and scale a real LLM to a substantial degree, within the constraints of our academic budget, and empirically demonstrate its performance improvement w.r.t strong baselines.

## Related Work

- **Recursive/parameter‑shared architectures**. Universal Transformers add depth‑wise sharing and ACT [^3]; Deep‑Equilibrium Networks adopt a fixed‑point, “infinite‑depth” view [^4]. Neural GPU demonstrates algorithmic generalization with tricks applied to convnets [^5], while DTNet/Deep‑Thinking highlights the utility of the `recall` mechanism for enabling stability and OOD length extrapolation [^6].

- **Adaptive computation**. Chain‑of‑Thought prompting increases output length to approximate extra compute [^7], but raises faithfulness and attribution concerns [^8] [^11] and can aggravate exposure‑bias errors in autoregression [^9]. Internal recursion (parameter sharing + iterative refinement) keeps compute in the latent space and, with the right normalization/skips, can be more stable.

- **Length/OOD generalization**. Vanilla transformers struggle to extrapolate to longer contexts under standard positional encodings [^10]. Recursive inductive biases aids in length generalization, but is often not easily parallelizable or scalable.

## Notation

We begin by introducing some notation. The model consumes a sequence of $x_i \in \mathcal{D}_n$ where $\mathcal{D}_n$ denotes the dataset/corpus.
A sequence for the model is thus $(x_0, ..., x_n)$, producing intermediate latent state denoted $(z_0, ..., z_n)$ and mapping to an output $(y_0, ..., y_n)$ where $n$ is the sequence length.
We use the standard $W_i \in \mathbb{F}^{m \times n}$ notation for denoting an $m$ by $n$ matrix in the field $F = \mathbb{R}$, at the $i$-th index.

The concatenation of $n$ arrays $A_0, \ldots, A_n$ along their corresponding last dimensions is denoted by the concatenation operator $\bigoplus: (\mathbb{F}^{i \times k} \times \mathbb{F}^{i \times k} \times ...) \rightarrow \mathbb{F}^{i \times nk}$.

Thus, we can write:

$$
A_{\text{concat}} = A_0 \bigoplus A_1 \bigoplus \ldots \bigoplus A_n = \bigoplus_{i=1}^n A_i
$$

## Method

### Base Architecture

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


### ASURA

We're now at a position to express our proposed $\textit{ASURA}$ architecture. We construct our architecture similar to that of DTNet/UT [^3] [^6], wherein we have a set of Attention blocks which are recursively applied $i$-times to the input.

We can define a stack of $L$ blocks/layers $\textit{ASURA} = [\mathcal{B}(\theta_{0}),\ldots,\mathcal{B}(\theta_{L})]$.

Applying it recursively for $i$ iterations on some input $(x_0, \dots, x_{t-1})$, the `fwd` pass looks like:

$$
\begin{aligned}
  \mathbb{P}(x_t | x_{t-1}, \dots, x_0) =  \textit{ASURA}(x_t) = (\mathcal{B}(\theta_0) \circ \mathcal{B}(\theta_1) \circ \dots \circ \mathcal{B}(\theta_L))(x_\{t-1})
\end{aligned}
$$

Where $\\{ \theta_i \in \Theta : \forall i, j:  \theta_i \neq \theta_j \\}$ are indexed parameter vectors. We also denote parameter-shared, depthwise recurrence in the residual ($\hat{X}_k$) space via:

$$
\begin{align}
  \hat{X}\_{i+1} = \text{ASURA} \(\hat{X}\_i\)
\end{align}
$$

However, this naive recursive formulation isn't optimal. Below we present how we derived the $\textit{ASURA}$ architecture starting from equation $(1)$.

### Deep Residual

One of the major problems with UTs is their lack of ineffectiveness w.r.t iterations. Prior work has demonstrated that ideally we should obtain a performance benefit $\propto \log({\text{depth}})$ [^12]. However, in practice that's hard to achieve. 

We introduce a projected deep residual connection to stabilize the network and improve gradient flow. For tensors $X_{0},\ldots,X_{n}$ with compatible last dims and a projection $W_{\mathrm{proj}} \in \mathbb{R}^{n k\times k}$:

$$
\operatorname{ProjConcat}(X_{0},\ldots,X_{n})^k = \Big( \bigoplus_{i=0}^{n} X_{i} \Big) W_{\mathrm{proj}}^k.
$$

![ProjConcat: project-and-concatenate long skip](asura_concat_proj.drawio.svg#full "ProjConcat: concatenate residuals, then project with W_proj.")

Where $X_i$ is the $i$-th residual of the network. In practice, for input $X_0$:

$$
\hat{X}\_{\mathrm{skip}}^t = \Big( \operatorname{Emb}(X_{0}) \oplus \hat{X}\_{\mathrm{t - 1}} \Big) \times W_{\mathrm{proj}}^k
$$

for recursive iteration $t \in \[1, \text{max\\_iters}\]$ and $\operatorname{Emb}$ is the standard `embed`-ing operator.

Astute readers would notice that this is similar to the recall mechanism introduced in [^6] and "Prelude" block in [^13]. However, Deep Residual is a more expressive way to perform input injection: 

1. Instead of simply propogating the `embed`-ed $X_0$, we downsample it's linear combination with the previous latent $X\_{t-1}$, allowing the network to dynamically adjust the information it wants to retain for the current iteration.

2. By relaxing parameter-sharing on $W\_{\mathrm{proj}}^t$, we increase the expressiveness of the model by *coupling* the projection to each iteration.

3. By directly injecting useful information as an explicit input for every iteration, we preserve the residual stream "bandwidth" [^17] by minimizing the amount of information the network has to propogate through its latent space across iterations. 

4. This operation also downsamples the concatenated inputs, thus improving parameter-count and FLOPs by reducing the embedding dimension by half.

Additionally, the input-dependence injects some dynamic behavior which can (hopefully) improve stability and act as a weak "gate" of sorts, to promote interactions between the latent and the original prompt that the current iteration needs.

From this, we can now replace our equation $(1)$ simply as:

$$
\begin{align}
  \hat{X}\_{i+1} = \text{ASURA}(\operatorname{ProjConcat}\_i(\operatorname{Emb}(X_{0}), \\: X_{i} )) =  \text{ASURA} \(X_{\mathrm{skip}}^i )
\end{align}
$$

### Decoupled Per-iteration `LayerNorm`

Prior work [^3] [^5] [^6] often doesn't explicitly design the architecture around `LayerNorm`s. The conventional wisdom is that recursing for $n$ iterations handles the norms and scaling appropriately during training, since `SGD` would explicitly optimize that.

However, we'd like to point out that there's little cost in "un-sharing" the normalization layers in return for stability. Additionally, it doesn't complicate the architecture design substantially and is a relatively simple and cheap way to accomplish our goal.


![Decoupled/Unshared Layernorms](asura_unshared_LN.drawio.svg#full "Ensure each iteration is normalized uniquely.")

Thus, equation $(2)$:
$$
\begin{aligned}
  \hat{X}\_{i+1} = \text{ASURA}(\operatorname{ProjConcat}\_i(\operatorname{Emb}(X_{0}), \\: X_{i} )) =  \text{ASURA} \(X_{\mathrm{skip}}^i )
\end{aligned}
$$

becomes:

$$
\begin{align}
  \implies \hat{X}\_{i+1} = \operatorname{LayerNorm}^i \( \text{ASURA} \(X_{\mathrm{skip}}^i \) \)
\end{align}
$$

### Decoupled Post-`LayerNorm`

While this may be sufficient, we go one step further - we place these un-"parameter shared" norms after **each layer** as well. Prior work would implicitly share these norms. Theoretically however, it makes sense that the statistics of the activations at each iteration would necessarily be slightly different and thus taking it into account would improve stability and performance.

This is effectively performing standard pre-`LN` for every $\mathcal{B}_i$ but applies an post-`LN` that's shared across the whole iteration.

![Block level norm unsharing](asura_block_ln.drawio.svg#full "Pre-`LN` is still standard, i.e unique for each block but implicitly shared across every iteration.")

In our sweeps, we didn't notice any major performance improvements - however for larger runs, it helped tremendously in avoiding exploding gradients and alleviated loss spikes.

![Loss curve for a 350M UT](./loss_curve_350M_x3_transparent.svg#light "Sample loss curve for a ~1B (350M * 3 iters) ASURA")

![Loss curve for a 350M UT](./loss_curve_dark.svg#dark "Sample loss curve for a ~1B (350M * 3 iters) ASURA")

Continuing from equation $(3)$, we now make the iteration dependence explicit by folding the per‑iteration post‑normalizations into the block. Define the depthwise block at iteration $i$ as $\operatorname{ASURA}\_i(\cdot)$ composed of alternating layers and iteration‑specific norms:

$$
\operatorname{ASURA}\_i(x)
  = \Big( 
      \mathcal{B}\_0(\theta_0) \circ \operatorname{LN}\_i \circ \\: \mathcal{B}\_1(\theta_1) \circ \operatorname{LN}\_i \circ \dots
     \Big)(x)
$$

and the recurrence becomes

$$
\begin{align}
\hat{X}\_{i+1} = \operatorname{ASURA}\_i\left(X\_{\mathrm{skip}}^{i}\right)
\end{align}
$$

In this equation $(4)$, the $\operatorname{LN}\_i^{(\ell)}$ denotes the post‑`LayerNorm` associated with layer $\ell$ at iteration $i$ and the operator now explicitly consumes the iteration index.

### Parameter Relaxation

Another aspect of training large, recursive architectures is the distinct lack of flexibility. Unlike conventional transformers, UTs are constrained by construction. However, for various reasons we want an ability to dynamically adjust the type of computation performed per iteration.

The intuition behind this choice is rooted in circuit theory. At every recursive application, we often want to address a subset of the circuits. There's been a substantial body of work [^16] [^17] that suggests that complex circuits are often formed via the "atomic" operations performed by the composition of attention heads.

We take inspiration from a tangential work [^18]. We "relax" the static parameters by injecting a non-parameter shared low-rank adapter. Thus, we have control over the amount of "fresh" parameters injected and provide a way for the network to modulate its own outputs in a data dependent fashion.

Our low-rank adapter of choice is the `ABBA` [^19] family of adapters. `ABBA` exploits the fact that high-rank updates can be achieved by the hadamard operator (denoted $\odot$) via taking inspiration from [HiRA](https://openreview.net/forum?id=TwJrTz9cRS) line of work. Effectively, the formulation is:

$$
\begin{aligned}
  \text{HiRA: }\Delta W = W\_0 \odot (BA)
\end{aligned}
$$

leveraging the property that $W\_1, W\_2$ with ranks $r\_1$ and $r\_2$ respectively "satisfy $\text{rank}(W\_1 \odot W\_2) \leq r\_1 \cdot r\_2$" and thus allowing us to greatly increase the effective rank of the operation via cheap elementwise operations easily parallelizable on modern accelerators.

However this fails to improve expressivity as we're still restricted to the subspace defined by $W\_0$.

`ABBA` works around this by parameterizing the projections to be learnable thus achieving effective rank of $r\_1 \cdot r\_2$ and improving expressivity.

$$
\begin{aligned}
  \text{ABBA: }\Delta W = \gamma (B\_1 A\_1) \odot (B\_2 A\_2)
\end{aligned}
$$

where $\gamma \in \mathbb{R}$ is the scaling factor for stability. 

![ABBA Block](asura_abba_lora.drawio.svg#full "An `ABBA` block. Figure recreated from the paper for aesthetic purposes.")

In practice, to avoid $m \times n$ matrix materialization we leverage the Khatri-Rao factorization as presented in the paper to compute the product in a memory efficient manner:

$$
\begin{aligned}
(B\_2 A\_2)
  = \underbrace{(B\_1 \odot\_r B\_2)}\_{m\times r\_1 r\_2}\
    \underbrace{\bigl(A\_1^{\top} \odot\_r A\_2^{\top}\bigr)^{\top}}\_{r\_1 r\_2 \times n}
\end{aligned}
$$

Additionally, `ABBA` assumes we're operating in the conventional fine-tuning regime wherein the original model weights are frozen. Thus, we discard the original authors' suggestion to carefully initialize the `ABBA` adapter via the `SVD`. Instead, we simply treat the adapter layers as trainable - similar to the other layers.

Thus, integrating the adapter in our blocks, $\mathcal{B}_i$ looks like:

![Parameter relaxation](./asura_adapter.drawio.svg#full "Parameter relaxation via `ABBA` blocks")

## Pseudocode

Putting everything together, we get something similar to this:


$$
\begin{aligned}
&\textbf{Pseudocode for ASURA} \newline
&\textbf{Input: } \text{parameters } \theta,\; \text{input } X,\; \text{iterations } i \newline
&\textbf{for } \texttt{batch\\_idx} = 1,2,\ldots \textbf{ do} \newline
&\quad X\_{\mathrm{in}} \leftarrow \texttt{embed\\_and\\_ln}(X) \newline
&\quad \texttt{latent} \leftarrow X\_{\mathrm{in}} \newline
&\quad \textbf{for } \texttt{iteration} = 0,1,\ldots, i \textbf{ do} \newline
&\qquad \hat{y}\_{\mathrm{proj}} \leftarrow \texttt{proj\\_and\\_concat}(\left[\texttt{latent},\; X\_{\mathrm{in}}\right]) \newline
&\qquad \texttt{latent} \leftarrow \operatorname{ASURA}\_i(\hat{y}\_{\mathrm{proj}}) \newline
&\qquad \texttt{latent} \leftarrow \texttt{LayerNorm}(\texttt{latent}) \newline
&\quad \textbf{end for} \newline
&\textbf{end for}
\end{aligned}
$$


## Notes on Efficiency

- Backprop through \(i\) iterations increases memory roughly with context length, batch size, hidden dim, and layers. We use activation checkpointing/scan-style rematerialization to trade compute for memory (treeverse checkpointing with known horizon \(i\)).


## Limitations

This project is very much a WIP. However, there are still some theoretical problems:

1. Because at each iteration, the $W\_{\mathrm{proj}}$ is unique, not shared, we cannot compute more iterations than the network has seen during training at inference time. In practice this is not an issue since the network can't handle OOD iterations anyways. However, follow-up work aims to address this issue and allow UTs to extrapolate for OOD number of iterations.

# Appendix

- Gory training details

<div class="wandb-embed" aria-label="Interactive W&B panel: Decoupled LayerNorms loss">
  <iframe
    src="https://wandb.ai/neel/ReAct_Jax/runs/3i_FW_100B_6/panel/z2473zhjk?nw=nwuserneel"
    loading="lazy"
    style="width: 100%; height: 520px; border: 0;"
    title="W&B Panel: Decoupled LayerNorms loss">
  </iframe>
  <p><a href="https://wandb.ai/neel/ReAct_Jax/runs/3i_FW_100B_6/panel/z2473zhjk?nw=nwuserneel" target="_blank" rel="noopener">Open interactive panel in W&B</a></p>
</div>


# References

[^1]: Vaswani et al. “Attention Is All You Need.” 2017. [link](https://arxiv.org/abs/1706.03762)

[^2]: Radford et al. “Improving Language Understanding by Generative Pre‑Training.” 2018. [link](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf)

[^3]: Dehghani et al. “Universal Transformers.” 2018. [link](https://arxiv.org/abs/1807.03819)

[^4]: Bai et al. “Deep Equilibrium Models.” 2019. [link](https://arxiv.org/abs/1909.01377)

[^5]: Kaiser & Sutskever. “Neural GPU.” 2015. [link](https://arxiv.org/abs/1511.08228)

[^6]: Bansal & Schwarzschild. “Deep Thinking Networks / DTNet.” 2022. [link](https://arxiv.org/abs/2202.05826)

[^7]: Wei et al. “Chain‑of‑Thought Prompting.” 2022. [link](https://arxiv.org/abs/2201.11903)

[^8]: Lanham et al. “On the Faithfulness of Chain‑of‑Thought.” 2023. [link](https://arxiv.org/abs/2307.13702)

[^9]: Bengio et al. “Scheduled Sampling for Sequence Prediction with Recurrent Neural Networks.” 2015. [link](https://arxiv.org/abs/1506.03099)

[^10]: Discussion on length extrapolation and positional encoding limitations (e.g., RoPE/PI limits). 2023. [link](https://arxiv.org/abs/2305.19466)

[^11]: Reasoning models don't always say what they think. 2025. [link](https://www.anthropic.com/research/reasoning-models-dont-say-think)

[^12]: Reasoning with Latent Thoughts: On the Power of Looped Transformers. 2024. [link](https://arxiv.org/abs/2502.17416)

[^13]: Scaling up Test-Time Compute With Latent Reasoning: A Recurrent Depth Approach. 2025. [link](https://arxiv.org/abs/2502.05171)

[^14]: The Serial Scaling Hypothesis. 2025. [link](https://arxiv.org/abs/2507.12549)

[^15]: On the Biology of a Large Language Model. 2025. [link](https://transformer-circuits.pub/2025/attribution-graphs/biology.html#dives-cot)

[^16]: https://arxiv.org/abs/1803.03635. 2018. [link](https://arxiv.org/abs/1803.03635)

[^17]: A Mathematical Framework for Transformer Circuits. 2021. [link](https://transformer-circuits.pub/2021/framework/index.html)

[^18]: Relaxed Recursive Transformers: Effective Parameter Sharing with Layer-wise LoRA. 2024. [link](https://arxiv.org/abs/2410.20672)

[^19]: ABBA: Highly Expressive Hadamard Product Adaptation for Large Language Models. 2025. [link](https://arxiv.org/abs/2505.14238v1)

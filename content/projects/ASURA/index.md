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

**ASURA** is a simple, recursive variant of Universal Transformers (UTs) [^3] aimed at language modeling, with improved stability and scalability. It applies a shared block across depth (recursion over iterations), augmented with long skip connections and extra normalizations. In our setup, it achieves strong performance, outperforming our baselines while preserving approximately the same relative `FLOP`s *w.r.t* standard UTs.

## Motivation

There has been a resurgence of interest in recursive architectures recently. However, current literature still lacks a rigorous treatment of recursion strategies. Industry and academia alike often presume that repeatedly applying the network $n$ times naively is often sufficient. We see no reason to believe that this is optimal. `ASURA` is our effort to develop tricks & techniques to improve recursive architectures and push them to be competitive with vanilla transformers.

But why should we focus on recursive architectures in the first place? we highlight some of the advantages below:

- Parameter scaling in isolation is insufficient. Robust multi-step reasoning and compositional inference demand greater depth and iterative computation - in line with the serial hypothesis [^14].

- Externalized chains (CoT) are brittle and can be unfaithful [^15]. Latent-space computation with recursion avoids compounding token-level errors and allows greater bandwidth for information transfer.

- Parameter sharing over depth offers **generalization** potential. In line with the compression hypothesis, increasing per-parameter efficiency should significantly improve the capabilities of LMs.

- UTs are a good tradeoff between serial and parallel computation, in line with the current paradigm and thus is the more cost-effective approach if we wish to reuse existing infrastructure.

- UTs have an inductive bias that delays memorization due to their very formulation. This implies that UTs are worse at smaller scales compared to vanilla transformers for knowledge tasks, but as we scale it up, the delta should vanish and we should expect to see substantial returns on reasoning (especially multi-hop).

- The holy grail, however, is the ability to convert any UT to perform arbitrary, unbounded **inference-time compute** natively in the latent space.

Conventional UTs however can be unstable, can diverge at scale and often don't utilize each recursive iteration to their full potential. In this work, we develop techniques to alleviate such problems and scale a real LLM to a substantial degree within the constraints of our academic budget. We empirically demonstrate improvements against strong baselines.

## Related Work

- **Recursive/parameter‑shared architectures**. Universal Transformers add depth‑wise sharing and ACT [^3]; Deep‑Equilibrium Networks adopt a fixed‑point, “infinite‑depth” view [^4]. Neural GPU demonstrates algorithmic generalization with techniques applied to convolutional networks [^5], while DTNet/Deep‑Thinking highlights the utility of the `recall` mechanism for enabling stability and OOD length extrapolation [^6]. Recently, there has been a rise in interest for hierarchial methods [^25] [^26] [^23] as well, which has proven to work well in simpler domains (i.e gridworlds or ARC-AGI) but their effectiveness in language modelling remains to be seen.

- **Adaptive computation**. Chain‑of‑Thought prompting increases output length to approximate extra compute [^7], but raises faithfulness and attribution concerns [^8] [^11] and can aggravate exposure‑bias errors in autoregression [^9]. Internal recursion (parameter sharing + iterative refinement) keeps compute in the latent space and, with the right normalizations and skip connections, can be more stable.

- **Length/OOD generalization**. Vanilla transformers struggle to extrapolate to longer contexts under standard positional encodings [^10]. However, such recursive inductive biases are often not easily parallelizable or scalable.

## Notation

We begin by introducing some notation. The model consumes a sequence of $x_i \in \mathcal{D}_n$ where $\mathcal{D}_n$ denotes the dataset/corpus.
A sequence for the model is thus $(x_0, ..., x_n)$, producing intermediate latent state denoted $(z_0, ..., z_n)$ and mapping to an output $(y_0, ..., y_n)$ where $n$ is the sequence length.
We use the standard $W_i \in \mathbb{R}^{m \times n}$ to denote an $m\times n$ matrix at index $i$.

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

We use a pre‑norm block:

$$
\mathcal{B}(x;\theta) = \operatorname{LayerNorm}\big(\operatorname{MLP}(\operatorname{MHA}(\operatorname{LayerNorm}(x)))\big)
$$

and we use the standard GeLU activation:

$$
\sigma_{\text{GeLU}}(x) = \tfrac{1}{2} \\, x \\, \Big(1 + \tanh\big(\sqrt{2/\pi}\\,(x + 0.044715\\,x^3)\big)\Big)
$$


### ASURA

We are now in a position to present our proposed $\textit{ASURA}$ architecture. We construct our architecture similar to that of DTNet/UT [^3] [^6], wherein we have a set of attention blocks that are recursively applied $i$ times to the input.

We can define a stack of $L$ blocks/layers $\textit{ASURA} = [\mathcal{B}(\theta_{0}),\ldots,\mathcal{B}(\theta_{L})]$.

Applying it recursively for $i$ iterations on some input $(x_0, \dots, x_{t-1})$, the forward pass is:

$$
\begin{aligned}
  \mathbb{P}(x_t | x_{t-1}, \dots, x_0) =  \textit{ASURA}(x_t) = (\mathcal{B}(\theta_0) \circ \mathcal{B}(\theta_1) \circ \dots \circ \mathcal{B}(\theta_L))(x_\{t-1})
\end{aligned}
$$

Let $\\{ \theta_i \in \Theta : \forall i, j:  \theta_i \neq \theta_j \\}$ denote indexed parameter vectors. We also denote parameter-shared, depthwise recurrence in the residual ($\hat{X}_k$) space via:

$$
\begin{align}
  \hat{X}\_{i+1} = \text{ASURA} \(\hat{X}\_i\)
\end{align}
$$

However, this naive recursive formulation isn't optimal. Next, we derive the $\textit{ASURA}$ architecture starting from equation $(1)$.

### Deep Residual

One of the major problems with UTs is their lack of effectiveness across iterations. Prior work has demonstrated that ideally we should obtain a performance benefit $\propto \log({\text{depth}})$ [^12]. However, in practice that's hard to achieve. 

We introduce a projected deep residual connection to stabilize the network and improve gradient flow. For tensors $X_{0},\ldots,X_{n}$ with compatible last dimensions and a projection $W_{\mathrm{proj}} \in \mathbb{R}^{n k\times k}$:

$$
\operatorname{ProjConcat}(X_{0},\ldots,X_{n})^k = \Big( \bigoplus_{i=0}^{n} X_{i} \Big) W_{\mathrm{proj}}^k.
$$

![ProjConcat: project-and-concatenate long skip](asura_concat_proj.drawio.svg#full "ProjConcat: concatenate residuals, then project with W_proj.")

Compared to naive recurrence, this reduces information bottlenecks in the residual stream by providing a direct, projected pathway from the input embedding and the previous latent, which improves gradient flow and utilization across iterations.

Here, $X_i$ denotes the $i$‑th residual of the network. In practice (for input $X_0$):

$$
\hat{X}\_{\mathrm{skip}}^i = \Big( \operatorname{Emb}(X_{0}) \oplus \hat{X}\_{\mathrm{i - 1}} \Big) \times W_{\mathrm{proj}}^k
$$

for recursive iteration $i \in \[1, \text{max\\_iters}\]$ and $\operatorname{Emb}$ is the standard embedding operator.

Note the similarity to the recall mechanism introduced in [^6] and the "Prelude" block in [^13]. However, Deep Residual is a more expressive way to perform input injection: 

1. Instead of simply propagating the `embed`-ed $X_0$, we downsample its linear combination with the previous latent $X\_{t-1}$, allowing the network to dynamically adjust the information it wants to retain for the current iteration.

2. By relaxing parameter-sharing on $W\_{\mathrm{proj}}^i$, we increase the expressiveness of the model by *coupling* the projection to each iteration.

3. By directly injecting useful information as an explicit input for every iteration, we preserve the residual stream "bandwidth" [^17] by minimizing the amount of information the network has to propagate through its latent space across iterations. 

4. This operation also downsamples the concatenated inputs, thus improving parameter-count and FLOPs by reducing the embedding dimension by half.

Additionally, the input‑dependence injects some dynamic behavior which can improve stability and act as a weak gate, promoting interactions between the latent and the original prompt that the current iteration needs.

From this, we can now rewrite equation $(1)$ as:

$$
\begin{align}
  \hat{X}\_{i+1} = \text{ASURA}(\operatorname{ProjConcat}\_i(\operatorname{Emb}(X_{0}), \\: X_{i} )) =  \text{ASURA} \(X_{\mathrm{skip}}^i )
\end{align}
$$

{{< note title="Note" >}}
The `ProjConcat` block is iteration-dependent. Each step $i$ has its own $W_{\mathrm{proj}}^i$, which is how we inject dynamic behavior into the otherwise shared ASURA block.
{{< /note >}}

### Decoupled Per-iteration `LayerNorm`

Prior work [^3] [^5] [^6] typically does not design the architecture around `LayerNorm`s, assuming that recursing for $n$ iterations and optimization will handle normalization and scaling during training.
However, activation statistics can drift across iterations (internal covariate shift), so per‑iteration normalization helps stabilize training [^27].

While this can be also remedied via adopting post-norm, it's considered a suboptimal choice according to current literature. Thus, ASURA ends up using effectively best of both worlds - the standard pre-LN block for well-behaved gradients and a `LayerNorm` after every iteration to provide a post-LN like effect.

Since there's little cost in "un-sharing" the normalization layers in return for stability. Additionally, it doesn't complicate the architecture design substantially and is a relatively simple and cheap way to accomplish our goal.

![Decoupled/Unshared LayerNorms](asura_unshared_LN.drawio.svg#full "Ensure each iteration is normalized uniquely.")

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

We chose to go one step further — we place un-"shared" norms after **each layer/block** as well. Prior work would implicitly share these norms across iterations. Theoretically, however, it makes sense that activations for the same block at a different iteration might introduce some covariate shift and thus factoring it in might be beneficial for us.

This is effectively performing standard pre-`LN` for every $\mathcal{B}_i$, but also applying an iteration-specific post-`LN` shared across all blocks in the iteration.

![Block level norm unsharing](asura_block_ln.drawio.svg#full "Pre-`LN` is still standard, i.e., unique for each block but implicitly shared across every iteration.")

In our sweeps, we didn't notice major performance improvements. However, for larger runs, leveraging both norm strategies reduced exploding gradients and loss spikes. We hope to run ablations to determine how beneficial either architectural decision is in isolation.

![Loss curve for a 350M UT](./loss_curve_350M_x3_transparent.svg#light "Sample loss curve for a ~`1B` (350M * 3 iters) `ASURA`")

![Loss curve for a 350M UT](./loss_curve_dark.svg#dark "Sample loss curve for a ~`1B` (350M * 3 iters) `ASURA`")

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

Another aspect of training large, recursive architectures is their limited flexibility. Unlike conventional transformers, UTs are constrained by construction. We seek to dynamically adjust the type of computation performed per iteration [^13]. Different iterations tend to target different subcircuits; LoRA‑style adapters provide iteration‑specific capacity.

The intuition behind this choice is rooted in circuit theory. At every recursive application, we often want to address a subset of the circuits. There's been a substantial body of work [^16] [^17] that suggests that complex circuits are often formed via the "atomic" operations performed by the composition of attention heads.

We take inspiration from a tangential work [^18]. We "relax" the static parameters by injecting a non‑parameter‑shared low‑rank adapter. Thus, we have control over the amount of "fresh" parameters injected and provide a way for the network to modulate its own outputs in a data‑dependent fashion.

Our low-rank adapter of choice is the `ABBA` [^19] family of adapters. `ABBA` exploits the fact that high-rank updates can be achieved by the Hadamard operator (denoted $\odot$), inspired by the [HiRA](https://openreview.net/forum?id=TwJrTz9cRS) line of work. Effectively, the formulation is:

$$
\begin{aligned}
  \text{HiRA: }\Delta W = W\_0 \odot (BA)
\end{aligned}
$$

leveraging the property that $W\_1, W\_2$ with ranks $r\_1$ and $r\_2$ respectively "satisfy $\text{rank}(W\_1 \odot W\_2) \leq r\_1 \cdot r\_2$" and thus allowing us to greatly increase the effective rank of the operation via cheap elementwise operations easily parallelizable on modern accelerators.

However, this does not improve expressivity because we remain restricted to the column space defined by $W\_0$.

`ABBA` addresses this by parameterizing the projections to be learnable, thus achieving an effective rank of $r\_1 \cdot r\_2$ and improving expressivity.

$$
\begin{aligned}
  \text{ABBA: }\Delta W = \gamma (B\_1 A\_1) \odot (B\_2 A\_2)
\end{aligned}
$$

where $\gamma \in \mathbb{R}$ is the scaling factor for stability. 

![ABBA Block](asura_abba_lora.drawio.svg#full "An `ABBA` block. Figure recreated from the paper for aesthetic purposes.")

In practice, to avoid $m \times n$ matrix materialization, we leverage the Khatri-Rao factorization as presented in the paper to compute the product in a memory-efficient manner:

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

At a high level, the forward pass is:

<div class="algorithm">
  <div class="algorithm-title"><i>Algorithm 1:</i> <code>ASURA</code></div>
  <div class="algorithm-body">

$$
\begin{aligned}
&\texttt{Pseudocode for ASURA} \newline
&\texttt{Input: } \text{parameters } \theta, \\; \text{input } X,\; \text{iterations } i \newline
&\newline
&\texttt{for } \text{batch\\_idx} = 1,2,\ldots \texttt{ do} \newline
&\quad X\_{\mathrm{in}} \leftarrow \texttt{embed\\_and\\_ln}(X) \newline
&\quad \texttt{latent} \leftarrow X\_{\mathrm{in}} \newline
&\quad \texttt{for } \text{iteration} = 0,1,\ldots, i \texttt{ do} \newline
&\qquad \hat{y}\_{\mathrm{proj}} \leftarrow \texttt{proj\\_and\\_concat}(\left[\texttt{latent}, \\; X\_{\mathrm{in}}\right]) \newline
&\qquad \texttt{latent} \leftarrow \operatorname{ASURA}\_i(\hat{y}\_{\mathrm{proj}}) \newline
&\qquad \texttt{latent} \leftarrow \texttt{LayerNorm}(\texttt{latent}) \newline
&\quad \texttt{end for} \newline
&\texttt{end for} \\newline
&\newline
&\texttt{logits} \\leftarrow \\texttt{head}(\\texttt{latent}) \\newline
&\texttt{loss} \\leftarrow \\texttt{CrossEntropy}(\\texttt{logits}, \\texttt{target})
\end{aligned}
$$

  </div>
</div>


## Notes on Efficiency

Since we're relying on simple Backpropagation-through-time (BPTT) for training an $i$‑iteration‑deep recursive architecture, this incurs memory costs on the order of $\approx \mathcal{O}(isbhL)$ [^20], where $s$ is the context length, $b$ is the batch size, $h$ is the hidden dimension size, and $L$ denotes the number of layers in the model.

To alleviate this, we use activation checkpointing/scan-style rematerialization to trade compute for memory (treeverse checkpointing with known horizon $i$).

Thus, we opt for a checkpointed version of a $\texttt{scan}$ to trade off compute for memory by rematerializing intermediate activations instead of storing them in memory.

This is accomplished by leveraging the offline variant of recursive "classical treeverse" checkpointing algorithm as implemented in the excellent [equinox](https://github.com/patrick-kidger/equinox) library as part of its internal toolkit [here](https://github.com/patrick-kidger/equinox/blob/cf1cf3310c870f4255e4d2cede770c8af3ae00bf/equinox/internal/_loop/loop.py#L131).


# Results

At the scales we operate at, we don't obtain enough signals from traditional benchmarks. Thus, we rely more on `LAMBADA` to judge the improvements in performance between models - a proxy for a "smooth" metric to judge capabilities.

We hope to secure additional compute and hopefully scale up these results in the short-term.

### 150M model

~80B tokens:

| **Model** | **LAMBADA (ppl)** | **MMLU (acc)** | **PIQA (acc)** | **WinoGrande (acc)** |
| --- | --- | --- | --- | --- |
| Baseline | 46.54 | 22.95% | 65.07% | 51.93% |
| ASURA (x3) | 32.73 | 23.02% | 67.1% | 52.4% |


Here, the red line is the ASURA run for 3 iterations (3i).

![Parameter relaxation](./val_ppl_150M_light_bg_legend_fixed.svg#light "Val/ppl for 150M model")

![Parameter relaxation](./val_ppl_150M_dark_bg_legend_fixed.svg#dark "Val/ppl for 150M model")


### 350M model

~80B tokens. Again, we see a non-significant bump in `LAMBADA` performance.

| **Benchmark** | **Baseline (350M)** | **ASURA (x3)** |
| --- | --- | --- |
| LAMBADA (ppl) | 22.09 | 19.184 |
| MMLU (acc) | 22.88% | 23.86% |
| PIQA (acc) | 68.77% | 69.74% |
| WinoGrande (acc) | 52.95% | 54.22% |
| BoolQ (acc) | 58.47% | 54.89 |
| ARC-Easy (acc) | 40.4% | 40.47% |
| HellaSwag (acc) | 40.42% | 42.74% |
| OpenBookQA (acc) | 28.2% | 28.8% |
| CommonsenseQA (acc) | 19.49% | 19.65% |

Evidently, it seems we could keep training for a lot more tokens and gain a better delta over baseline. So in the next run, we scale tokens instead of parameters.

![Parameter relaxation](./val_ppl_350M_light_bg_legend_clip_nobreaks.svg#light "Val/ppl for 350M model")

![Parameter relaxation](./val_ppl_350M_dark_bg_legend_clip_nobreaks.svg#dark "Val/ppl for 350M model")


## [WIP]

More results WIP

---


## Limitations

This project is very much a WIP. However, there are  some theoretical problems:

- We test at a fairly limited scale due to time and resource constraints. We hope to scale up this work and develop the `ASURA` architecture further.

- At each iteration, the $W\_{\mathrm{proj}}$ and the `ABBA` adapters are unique, i.e., not shared across iterations.

Thus, at inference we cannot compute more iterations than the network has seen during training; in practice, however, this is acceptable, as the network cannot handle OOD iterations anyway. Follow‑up work aims to address this limitation and allow UTs to extrapolate for OOD iteration counts.

Additionally, because of the way we've built our accelerators, we benefit from batching - thus computing a static, pre-computed amount of iterations is more performant than resorting to `bsz=1`. HRM/TRM [^23] [^26] style models for instance adopt this for tackling the [ARC-AGI](https://arcprize.org/arc-agi) challenge. 

Still, we recognize that this is a limitation especially if we wish to achieve results in a similar vein to Geiping et al. [^13]. Follow‑up work aims to address this limitation and allow UTs to extrapolate to OOD numbers of iterations.

## Future Ideas?

**Hierarchy:** One way to address this limitation is to incorporate a hierarchical component, inspired by HRM [^23], CTM [^24], HTM [^25].

The gist is a complex, hierarchical transformation of the latent $\hat{X}_i$ and the input $X_0$ that updates the adapter weights each iteration.

While in this work we use the traditional, `LoRA`‑like adapter formulation, more expressive approaches are possible. The adapter is effectively a *key* or a *"gate"* that matches against what subset of circuits we wish to activate.

This can be done through other, more elegant formulations.

For example, one could naively imagine an attention-like mechanism except applied depthwise — treating each layer's weights as a token in the sequence and matching against our "key" — the adapter's latent.

This would treat it as a more classical match & retrieval problem wherein we consume the adapter information and previous latent, producing a weighted combination of each neuron's outputs. ([Hopfield](https://arxiv.org/abs/2008.02217) networks?)

One can easily imagine approximations to this mechanism performing well by better matching adapter selection to subcircuit activation. Given more work in MechInter and especially studying how circuits form in DNNs, perhaps this would be a viable direction in the future. 

**Score Function:** UTs have a weak correspondence with traditional diffusion models. Is there a lens via which we can view each recursive iteration? If we take the conventional score‑function perspective, are there any specific inductive biases/constraints/priors we can impose on each iteration to make it more amenable to scaling and improve performance?

Is there an equivalence that allows us to unify the fixed‑point DEQ formulation and diffusion models to reap both their benefits without resorting to solvers or expensive denoising‑based sampling procedures?

**Holy Grail:** Training UTs is often a sore point for many since in practice it's expensive. Could we perhaps train vanilla LLMs with some specific recipe to embed certain inductive biases that make it easier to convert them into a recursive architecture later on?

We know Knowledge acquisition isn't hard - so if we're completely focusing on reasoning it doesn't make sense to start afresh. If we can transfer most of the knowledge from vanilla LMs into recursive architectures, it would provide an easy boost for pre-training and make them cheap enough to scale.

**Adaptive Time-Computation:** Extending these models to unbounded iterations is a hard problem. Only a handful of works have attempted [^13] to do so and have (somewhat) succeeded.

However, it's clear that latent-adaptive computation unlocks a lot of benefits for reasoning. There are powerful ideas, such as parallel-time adaptive computation and MCTS-styled approaches, that may emerge as this paradigm is adopted.

Geiping et al. [^13] use truncated BPTT (i.e., backprop only the last iteration) which is efficient to train, however comes with severe performance penalties. While the `n + k` style training recipe works for extrapolating to OOD iterations, it's clearly not ideal.

It might thus be easier to fine-tune or convert an `ASURA`-styled architecture towards a Geiping et al. styled "post-training" phase to extrapolate for unbounded iterations.

We see that as a very promising direction and would hope that labs might be interested in scaling up an architecture of this flavour.

**More Recurrence**: Recurrence is a great prior. It forces representations and circuits to be modular, and increases parameter efficiency. However, RNNs are recurrent on the timestep axis - and UTs are recurrent depth-wise. Some works have already unified this idea (somewhat) but we can extend it to more axes and perhaps unlock some other capabilities/efficiency gains.

# Appendix

Here we briefly describe the methodology we used to perform the experiments:

Most of our time/compute was spent on hyperparameter tuning to compare various methods and evaluate their performance gain *w.r.t*. the baseline.

We've tried our best to ensure that the runs are bug-free, and the results are as reproducible as possible. This has been much easier thanks to `JAX`'s explicit control over randomness and TPUs' static architecture.

- We used a TPE (Tree-Structured Parzen Estimator [^21]) with a fixed seed to ensure reproducibility. This was deployed in conjunction with an aggressive `PercentilePruner` to keep only the top $25$-%ile of the runs and prune the rest.

- For both the `FineWeb` and `OpenWebText` dataset, we only hypertune with $1\\%$ of the data, i.e., $\approx 1B$ tokens.

- We use a standard, competitive pre-`LN` transformer baseline with `RoPE` [^22] and the aforementioned `GeLU` activation inline.

- The standard `AdamW` optimizer recipe is used, with a `cosine`-decay scheduler with warmup.

- We employ the stable version of cross‑entropy with PaLM-style z‑loss for mixed precision numerical stability.


# Credits

- [`TRC`](https://sites.research.google/trc/about/) (`TPU` Research Cloud Program) for sponsoring all the compute used for this project for its entire duration.

- Google Developers Expert Program (for $1000 in `GCP` Cloud Credits)

- Patrick Kidger especially, for developing Equinox, as well as the maintainers of the wider `JAX` ecosystem who work tirelessly to push forward scientific computing.

# Citation

If you found this technical report useful, you can cite us by:

```bibtex
@misc{asura2025,
  title        = {ASURA: Asymptotically Universal Recursive Architecture},
  author       = {Neel Gupta},
  year         = {2025},
  howpublished = {Project Page},
  url          = {https://neel04.github.io/my-website/projects/ASURA/}
}
```

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

[^10]: Kazemnejad et al. “Discussion on length extrapolation and positional encoding limitations (e.g., RoPE/PI limits).” 2023. [link](https://arxiv.org/abs/2305.19466)

[^11]: Reasoning models don't always say what they think. 2025. [link](https://www.anthropic.com/research/reasoning-models-dont-say-think)

[^12]: Saunshi et al. “Reasoning with Latent Thoughts: On the Power of Looped Transformers.” 2024. [link](https://arxiv.org/abs/2502.17416)

[^13]: Geiping et al. “Scaling up Test-Time Compute With Latent Reasoning: A Recurrent Depth Approach.” 2025. [link](https://arxiv.org/abs/2502.05171)

[^14]: Liu et al. “The Serial Scaling Hypothesis.” 2025. [link](https://arxiv.org/abs/2507.12549)

[^15]: On the Biology of a Large Language Model. 2025. [link](https://transformer-circuits.pub/2025/attribution-graphs/biology.html#dives-cot)

[^16]: Frankle & Carbin. 2018. [link](https://arxiv.org/abs/1803.03635)

[^17]: A Mathematical Framework for Transformer Circuits. 2021. [link](https://transformer-circuits.pub/2021/framework/index.html)

[^18]: Bae et al. “Relaxed Recursive Transformers: Effective Parameter Sharing with Layer-wise LoRA.” 2024. [link](https://arxiv.org/abs/2410.20672)

[^19]: Singhal et al. “ABBA: Highly Expressive Hadamard Product Adaptation for Large Language Models.” 2025. [link](https://arxiv.org/abs/2505.14238v1)

[^20]: Transformer Math 101. 2023. [link](https://blog.eleuther.ai/transformer-math/)

[^21]: Watanabe et al. “Tree-Structured Parzen Estimator: Understanding Its Algorithm Components and Their Roles for Better Empirical Performance.” 2023. [link](https://arxiv.org/abs/2304.11127)

[^22]: Su et al. “RoFormer: Enhanced Transformer with Rotary Position Embedding.” 2021. [link](https://arxiv.org/abs/2104.09864)

[^23]: Wang et al. “Hierarchical Reasoning Model.” 2025. [link](https://arxiv.org/abs/2506.21734)

[^24]: Continuous Thought Machines. 2025, [link](https://sakana.ai/ctm/)

[^25]: Hierarchical Temporal Memory. 20xx. [link](https://en.wikipedia.org/wiki/Hierarchical_temporal_memory)

[^26]: Jolicoeur‑Martineau. “Less is More: Recursive Reasoning with Tiny Networks.” 2025. [link](https://arxiv.org/abs/2510.04871)

[^27]: Ioffe & Szegedy. “Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift.” 2015. [link](https://proceedings.mlr.press/v37/ioffe15.html)

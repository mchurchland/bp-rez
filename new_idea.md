# Biological Processing Unit or Generic Reservoir?

## Controlled replication and causal ablation of a connectome-derived recurrent network

## Core idea

Replicate the main experiments in Yu et al., *Biological Processing Units: Leveraging an Insect Connectome to Pioneer Biofidelic Neural Architectures*, while replacing the larval *Drosophila* connectome with carefully matched random and perturbed recurrent networks.

The central question is:

> Does the biological connectome provide a task-relevant computational advantage beyond that of a generic sparse recurrent reservoir with comparable size, weights, signs, and dynamics?

The study should be framed as a neutral controlled replication rather than a predetermined rebuttal. A positive result for the connectome would strengthen the original claim; equivalent or superior performance from the null networks would show that the reported performance is not evidence of a connectome-specific advantage.

## Motivation

Yu et al. embed the larval *Drosophila* connectome as a fixed recurrent core. A GNN or CNN encodes chess positions, a learned input projection drives the connectome, and a learned output projection decodes its activity. The connectome-derived recurrent weights are not trained. This is therefore a reservoir-computing architecture with a biologically derived reservoir and, for chess, a substantial learned encoder.

The paper compares the connectome with a partially frozen feedforward MLP for image classification, but it does not report a matched random recurrent-reservoir baseline. Consequently, its experiments show that the connectome *can* serve as a reservoir, but do not establish that its biological topology or synaptic organization makes it a better reservoir.

This concern is closely related to Brunton et al., *The Digital Sphinx: Can a Worm Brain Control a Fly Body?* Their deliberately implausible worm-brain/fly-body model produces realistic fly walking after training an artificial motor interface. They argue that the worm connectome is functioning as a recurrent feature generator and could plausibly be replaced by a random RNN because the learned interface carries much of the task-specific burden. The Digital Sphinx is a conceptual caution rather than an experimental random-network comparison, making the proposed study a direct empirical test of that argument in the BPU setting.

## Main hypotheses

### Null hypothesis

Once size, sparsity, weight statistics, signs, spectral radius, input/output dimensions, and training budget are controlled, the biological connectome does not consistently outperform matched random reservoirs.

### Alternative hypothesis

The biological connectome produces a reproducible advantage in accuracy, sample efficiency, robustness, or dynamical capacity that cannot be explained by coarse graph statistics or weight distributions.

## Experimental conditions

The original connectome should be compared with an ensemble of null and ablation models, not with a single random draw.

1. **Original connectome**
   - Preserve the published directed topology, contact-count weights, inferred signs, and sensory/internal/output partition.

2. **Magnitude shuffle**
   - Preserve topology and signs.
   - Randomly permute weight magnitudes over existing edges.
   - Tests whether the placement of synaptic strengths matters.

3. **Sign shuffle**
   - Preserve topology and magnitudes.
   - Randomly permute signs while preserving the global excitatory/inhibitory ratio.
   - Tests whether biological sign placement matters.

4. **Independent sign-and-magnitude shuffle**
   - Preserve topology only.
   - Randomize sign and magnitude placement independently.

5. **Directed degree-preserving rewiring**
   - Preserve each node's in-degree and out-degree as closely as possible.
   - Reassign or resample weights and signs under explicitly stated constraints.
   - Tests whether higher-order topology matters beyond the degree sequence.

6. **Block-preserving random graph**
   - Sample from a fitted directed, signed stochastic block model.
   - Tests whether mesoscale population structure is sufficient.

7. **Matched Erdős--Rényi reservoir**
   - Match neuron count, edge count or density, weight distribution, and sign ratio.
   - Provides a completely random topology baseline.

8. **Tuned standard echo-state network**
   - Use a conventional sparse random reservoir with the same state dimension and a comparable parameter budget.
   - Tune it under the same validation and compute budget as the connectome model.

9. **Encoder/readout-only control**
   - Remove or bypass the recurrent reservoir while retaining a parameter-matched prediction head.
   - Especially important for chess, where the learned GNN or CNN encoder may already perform much of the computation.

## Essential matching and fairness controls

Random networks should be matched to the connectome on:

- Number of recurrent units.
- Number of directed edges or connection density.
- Input and output population sizes.
- Empirical weight-magnitude distribution.
- Excitatory/inhibitory sign ratio.
- Presence or absence of self-loops.
- Activation function.
- Number of recurrent update steps.
- Input scaling and bias treatment.
- Trainable parameter count.
- Optimizer, learning-rate schedule, batch size, and training budget.
- Hyperparameter-search budget.

### Spectral and activation matching

Reservoir performance can change drastically with spectral radius, input scaling, recurrent depth, and activation saturation. A biological/random comparison could therefore be confounded by basic dynamical scale.

Two complementary regimes should be reported:

1. **Fixed published configuration:** apply exactly the same preprocessing and hyperparameters to every reservoir. This tests plug-in robustness.
2. **Dynamics-matched configuration:** rescale reservoirs to common spectral-radius targets and check activation distributions. This tests topology after controlling for elementary dynamical differences.

Each reservoir family should also receive the same validation budget for tuning. Otherwise, a well-tuned connectome would be compared unfairly with arbitrarily parameterized random networks.

## Replication sequence

### Phase 1: Reproduce the original result

- Reproduce the reported MNIST and CIFAR-10 accuracies with the biological connectome.
- Record all implementation choices that are missing or ambiguous in the paper.
- Compare parameter counts with the reported values.
- Verify the neuron partitions and the exact recurrent matrix used.

If the original result cannot be reproduced because code or methodological details are unavailable, describe the work as a **controlled reimplementation**, not an exact replication.

### Phase 2: Run matched null models

- Freeze the encoder, input/output architecture, optimizer, data split, and training protocol.
- Substitute each reservoir condition.
- Use multiple graph realizations and multiple training seeds.
- A reasonable starting point is 20--50 reservoir draws per stochastic condition and at least three training seeds per draw, subject to computational cost.

### Phase 3: Mechanistic analysis

Measure properties that may explain performance differences:

- Kernel rank.
- Generalization rank.
- Linear memory capacity.
- Information-processing capacity.
- Effective dimensionality or participation ratio.
- Activation saturation and dead-unit fraction.
- Stability across recurrent steps.
- Sensitivity to spectral radius, input scaling, leak rate, and bias.
- Robustness to weight, sign, and topology perturbations.

This phase moves the study beyond the observation that random reservoirs work and asks *why* particular reservoirs succeed.

### Phase 4: Chess, if reproducible

The chess experiment should be attempted only after obtaining enough implementation detail to define:

- The exact prediction target: state value, action value, or move logits.
- The readout equation and output activation.
- The loss function.
- How legal moves are generated and ranked.
- How a predicted value is converted into a move without search.
- How the value model is used at minimax leaves.
- Whether the GNN/CNN encoder is trained end-to-end.
- The recurrent depth and state initialization.

The published description is not currently sufficient to reconstruct these choices confidently. MNIST and CIFAR-10 are therefore the cleaner starting point.

## Statistical analysis

The biological connectome is one observed graph, whereas the random controls define distributions of graphs. The biological result should be located within each empirical null distribution.

Report:

- Mean, standard deviation, and confidence intervals across graph and training seeds.
- The percentile of the biological connectome within each null distribution.
- Effect sizes, not only significance tests.
- Paired comparisons where training seeds and data splits can be aligned.
- Performance as a function of the hyperparameters rather than only the best point.
- Sample efficiency and compute cost in addition to final accuracy.

The main inference should not depend on one favorable random seed or one selected hyperparameter configuration.

## Interpretation

Possible outcomes include:

1. **Matched random reservoirs perform equivalently.**
   - The BPU results demonstrate generic reservoir computation, not a biological-connectome advantage.
   - Claims about biofidelic computational benefit should be weakened.

2. **The degree-preserving null performs equivalently, but Erdős--Rényi does not.**
   - Coarse degree structure may explain the advantage; detailed biological wiring is unnecessary.

3. **Block-preserving models perform equivalently.**
   - Mesoscale organization may be sufficient.

4. **Only the original connectome consistently wins.**
   - This supports a specific biological structural advantage.
   - The perturbation results can identify whether topology, sign placement, or magnitude placement is responsible.

5. **The encoder/readout-only model performs similarly.**
   - The learned interfaces, rather than the reservoir, may account for most of the task performance.

6. **Performance depends primarily on spectral or activation matching.**
   - The apparent architectural advantage may be a dynamical-scaling effect rather than a topological one.

## Contribution and positioning

The strongest framing is:

> Performance alone establishes that a connectome is a usable computational substrate. Establishing that biological wiring contributes something special requires matched recurrent null models and causal perturbations.

This project would contribute:

- An open replication or reimplementation of the BPU architecture.
- Proper random-recurrent and topology-preserving baselines.
- A causal decomposition of topology, signs, and weight placement.
- A dynamical explanation of any observed performance differences.
- A bridge between connectome-derived AI claims and reservoir-computing theory.

The work is valuable regardless of whether it confirms or challenges the original paper.

## Reproducibility concerns to resolve

Before implementation, request code or clarification from the BPU authors if possible. Important ambiguities include:

- The precise connectome dataset and preprocessing.
- Why the reported sensory, internal, and output populations do not sum exactly to the rounded network size.
- Recurrent-matrix normalization or scaling.
- State initialization and number of recurrent steps.
- Training details and loss functions.
- Whether chess encoders are trained end-to-end.
- The chess readout and move-selection procedure.
- Exact data splits and evaluation scripts.

Document every assumption. Where an original choice cannot be recovered, test plausible alternatives and report sensitivity rather than silently selecting one.

## Minimum viable version

A tractable first paper or thesis extension could focus only on MNIST and CIFAR-10:

1. Reimplement the biological BPU.
2. Compare it with magnitude shuffle, sign shuffle, degree-preserving rewiring, matched Erdős--Rényi, and a tuned ESN.
3. Use at least 20 graph seeds and multiple training seeds.
4. Match spectral radius and activation scale.
5. Report accuracy, sample efficiency, kernel/generalization rank, memory capacity, and robustness across hyperparameters.
6. Release the full pipeline and generated matrices.

Chess can be added later if the missing methodological details become available.

## Possible titles

- **Biological Processing Unit or Generic Reservoir? A Controlled Replication of Connectome-Derived Recurrent Networks**
- **Does Biological Wiring Matter? Matched Random Controls for Connectome-Based Reservoir Computing**
- **From Connectome to Reservoir: Testing the Computational Specificity of a Biological Processing Unit**
- **A Fly Brain or Just a Good Reservoir? Causal Controls for Connectome-Derived AI**

## Primary references

- Yu, S., Qin, Z., Liu, T., Xu, B., Vogelstein, R. J., Brown, J., & Vogelstein, J. T. (2025). *Biological Processing Units: Leveraging an Insect Connectome to Pioneer Biofidelic Neural Architectures*. [arXiv:2507.10951](https://arxiv.org/abs/2507.10951)
- Brunton, B. W., Abe, E. T. T., Hu, L. J., & Tuthill, J. C. (2026). *The Digital Sphinx: Can a Worm Brain Control a Fly Body?* [Paper](https://faculty.washington.edu/tuthill/docs/TheSphinx_2026.pdf) and [code](https://github.com/Brunton-Lab/DigitalSphinx2026)
- Jin, Z., Zhu, Y., Zhang, C., & Sui, Y. (2026). *Whole-Brain Connectomic Graph Model Enables Whole-Body Locomotion Control in Fruit Fly*. [arXiv:2602.17997](https://arxiv.org/abs/2602.17997). This related work already uses degree-preserving rewiring, a random graph, and MLP controls, and should be considered when positioning novelty.
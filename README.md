# Constrained Deep Learning using Conditional Gradient and Applications in Computer Vision

## Abstract
A number of results have recently demonstrated the benefits of incorporating various constraints
when training deep architectures in vision and machine learning. The advantages range from guarantees
for statistical generalization to better accuracy to compression. But support for general
constraints within widely used libraries remains scarce and their broader deployment within many
applications that can benefit from them remains under-explored.

In this project, we revisit a classical first order
scheme from numerical optimization, Conditional Gradients (CG), that has, thus far had limited
applicability in training deep models.

More details about the project can be found in [arxiv](https://arxiv.org/abs/1803.06453)


## Code details
See *resnet-in-tensorflow*  for the resnet experiments

See *dcgan-completion.tensorflow* for GAN experiments

See *path-sgd* for path-norm experiments

See *norms.py* for various CG updates for simple constrained problems mentioned in the main paper.

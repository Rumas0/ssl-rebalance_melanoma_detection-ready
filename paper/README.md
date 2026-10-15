# Paper

**Shifting vs. Lifting Melanoma Recall under Severe Class Imbalance: An Empirical Study of Loss-Level, Representation-Level, and Generative Interventions**

I. S. Tunib (Daffodil International University), Md. Sazzadur Ahamed (Daffodil International University), M. F. Bin (Jashore University of Science and Technology)

**Status:** manuscript has been submitted to "Computer Methods and Programs in Biomedicine Update" publication of Elsevier and is under review now. This folder is reserved for publication materials and will be updated with a link once a preprint or published version exists.

## Summary

A three-class (BKL, MEL, NV) melanoma pipeline on ISIC 2019 is studied under severe class imbalance and a tiny labeled set (691 images, 35 melanoma): SimCLR pre-training on 24,000 unlabeled images, then fine-tuning with melanoma oversampling, melanoma-specific augmentation, and focal loss. The paper separates three intervention levels and finds that a loss-level change (adaptive class weighting) shifts melanoma recall along a fixed precision–recall curve rather than lifting it; a representation-level change (supervised contrastive loss with squeeze-and-excitation attention) does lift melanoma Average Precision; and a generative change (DCGAN augmentation) hurts, with feature-space diagnostics showing mode collapse. External validation on HAM10000 exposes a large domain gap that a small real supplement partly closes.

Full results: [`../results/README.md`](../results/README.md).

## Manuscript figures in this repository

File numbers follow the manuscript's figure numbers.

| Manuscript figure | File |
|---|---|
| Fig. 1: ISIC 2019 data usage | [`fig01_data_split.png`](../figures/fig01_data_split.png) |
| Fig. 2: Representative images | [`fig02_representative_images.png`](../figures/fig02_representative_images.png) |
| Fig. 3: Two-stage pipeline | [`fig03_two_stage_pipeline.png`](../figures/fig03_two_stage_pipeline.png) |
| Fig. 4: Encoder architecture | [`fig04_encoder_architecture.png`](../figures/fig04_encoder_architecture.png) |
| Fig. 5: PR curve with adaptive operating points | [`fig05_pr_curve_adaptive.png`](../figures/fig05_pr_curve_adaptive.png) |
| Fig. 6: Per-seed ROC and PR curves | [`fig06_roc_pr_per_seed.png`](../figures/fig06_roc_pr_per_seed.png) |
| Fig. 7: Per-class Average Precision | [`fig07_per_class_ap.png`](../figures/fig07_per_class_ap.png) |
| Fig. 8: Pooled ROC and PR curves | [`fig08_roc_pr_pooled.png`](../figures/fig08_roc_pr_pooled.png) |
| Fig. 9: Confusion matrices | [`fig09_confusion_matrices.png`](../figures/fig09_confusion_matrices.png) |
| Fig. 10: Bootstrap confidence intervals | [`fig10_bootstrap_ci.png`](../figures/fig10_bootstrap_ci.png) |
| Fig. 11: Synthetic melanoma samples | [`fig11_gan_samples.png`](../figures/fig11_gan_samples.png) |
| Fig. 12: Effect of a real HAM10000 supplement | [`fig12_ham_supplement.png`](../figures/fig12_ham_supplement.png) |
| Fig. 13: Fine-tuning convergence | [`fig13_finetune_convergence.png`](../figures/fig13_finetune_convergence.png) |

## Planned contents

Add here only when appropriate for public release: a preprint link or PDF, the LaTeX source, and the bibliography file. Until the manuscript is submitted and the publisher's policy on sharing is clear, keep the full source out of the public repository.

## Citing

See [`../CITATION.cff`](../CITATION.cff). Please cite the manuscript once its bibliographic record is finalized.

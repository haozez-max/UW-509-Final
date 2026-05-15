# EEP 595 Privacy-Preserving ML - Project Part 2 Progress Report

Project: Evaluating Privacy-Utility Tradeoffs in Machine Learning  
Team: Joe Zhang, Mingwei Xu, Houser Zhang  
Due: May 16, 2026

## Progress and Execution

Our proposal committed to benchmarking differential privacy (DP), federated learning (FL), and homomorphic encryption (HE) on a shared classification task. For Part 2, we built a runnable Google Colab pipeline that validates the experimental structure before scaling to larger models and datasets. The current implementation uses scikit-learn's digits dataset (1,797 grayscale 8x8 handwritten digit images), normalizes pixels to `[0, 1]`, and uses an 80/20 stratified train-test split with seed 509.

Completed components include a non-private centralized logistic-regression baseline, a NumPy softmax classifier with per-example gradient clipping and Gaussian noise as a DP-SGD-style prototype, a five-client IID FedAvg simulation with 30 communication rounds and two local epochs per round, and an HE feasibility track focused on encrypted inference rather than encrypted training.

## Preliminary Results

The table below is intended for direct use in the final report. Accuracy and macro-F1 measure utility, train-test gap and confidence-based membership-inference AUC provide preliminary privacy-leakage proxies, and runtime records practical cost.

| Method | Test acc. | Macro-F1 | Gap | MIA AUC | Runtime |
|---|---:|---:|---:|---:|---:|
| Centralized baseline | 98.33% | 0.9833 | 0.0027 | 0.5191 | 0.051s |
| DP-style, noise 0.5 | 97.22% | 0.9718 | 0.0034 | 0.5191 | 0.231s |
| DP-style, noise 1.0 | 96.67% | 0.9667 | 0.0020 | 0.5188 | 0.233s |
| DP-style, noise 2.0 | 95.00% | 0.9500 | 0.0020 | 0.5119 | 0.260s |
| FedAvg, 5 IID clients | 96.94% | 0.9693 | 0.0020 | 0.5134 | 0.088s |

Interpretation: the centralized baseline reaches 98.33% test accuracy. The DP-style runs show the expected utility cost as noise increases, falling from 97.22% at noise 0.5 to 95.00% at noise 2.0. FedAvg reaches 96.94% under the IID client split, confirming that the FL pipeline works before we introduce non-IID data partitions. The confidence-based MIA AUC proxy stays close to 0.5, suggesting weak confidence-based membership distinguishability on this small task; this should be treated as preliminary rather than a complete privacy audit.

Figure 1: DP noise multiplier versus test accuracy and confidence-based membership-inference proxy. See `part2_dp_noise_curve.png`.

## Obstacles and Challenges

- The current dataset is intentionally small, so it validates the pipeline but not final-scale performance.
- The DP prototype implements clipping and Gaussian noise, but formal epsilon accounting remains future work.
- The FL experiment currently uses IID clients; non-IID splits are likely to create harder convergence behavior.
- HE is computationally expensive for training, so the current scope is encrypted inference feasibility.

## Engagement with the Literature

Abadi et al. introduced DP-SGD as the central mechanism behind our DP prototype: per-example gradient clipping limits any individual record's contribution, while Gaussian noise provides the privacy mechanism. Our current implementation captures this operational idea, but the privacy accountant from the paper is not yet implemented.

McMahan et al. motivates the FL component through FedAvg, where clients train locally and the server aggregates model updates. Our IID five-client simulation is a first sanity check of this training loop; the next step is to reproduce the more realistic non-IID setting emphasized by federated learning literature.

Lee et al. helps frame the HE track. The key lesson is that encrypted neural-network inference can protect client inputs, but it adds substantial computational overhead and requires careful model choices. This is why our Part 2 work treats HE as inference-only feasibility rather than full encrypted training.

Basu et al. reinforces the importance of benchmarking privacy mechanisms under shared tasks and metrics. We used that idea to keep baseline, DP, and FL results comparable instead of evaluating each technique in isolation.

## Remaining Plan

- Add formal DP privacy accounting, preferably using an RDP accountant or Opacus-style epsilon reporting.
- Run non-IID FL experiments and compare convergence against the current IID FedAvg curve.
- Attempt the optional TenSEAL encrypted linear inference demo and report plaintext versus encrypted latency.
- If time allows, scale from digits to MNIST or a small CNN so the final report better matches the original scope.
- Prepare final figures, a short demo screenshot, and presentation slides centered on the observed tradeoffs.

## Planned Class Presentation

We plan to present the project as a staged comparison: first the non-private baseline, then how DP changes utility and privacy proxies, then how FL changes the trust and deployment model, and finally why HE is promising but costly for encrypted inference. The demo will use the Colab notebook and the generated result table/curves.

## References

- Abadi et al., Deep Learning with Differential Privacy, CCS 2016.
- McMahan et al., Communication-Efficient Learning of Deep Networks from Decentralized Data, AISTATS 2017.
- Lee et al., Privacy-Preserving Machine Learning with Fully Homomorphic Encryption for Deep Neural Network, IEEE Access 2022.
- Basu et al., Benchmarking Differential Privacy and Federated Learning for BERT Models, arXiv 2021.

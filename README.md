# Self-Training LLM & Speculative Decoding

## 1. Description
This project explores **Speculative Decoding** to accelerate the inference of a Class-Conditional GPT model. We trained a Teacher model on the **Mini-ImageNet** dataset (specifically dog classes) using VQ-VAE tokens. To enable faster inference, we distilled the Teacher's knowledge into smaller **Student** models (10L, 8L, 6L, 4L) using a **Hybrid Distillation** strategy (Hard Cross-Entropy + Soft KL Divergence). The goal was to use the fast Student to "draft" tokens that the Teacher validates in parallel, theoretically speeding up generation without losing quality.

**Key Technologies:**
*   **Architecture**: Decoder-only Transformer (GPT) with KV Caching.
*   **Tokenizer**: VQ-VAE (Vector Quantized Variational Autoencoder).
*   **Algorithm**: Speculative Sampling (Leviathan et al.).

## 2. Code Repository Outline
The codebase is organized into modular components:

```
├── models/                  # Core Architecture
│   ├── model.py             # GPT Transformer (w/ KV Cache, Class-Conditioning)
│   └── vqvae.py             # VQ-VAE for image tokenization
├── data/                    # Data Pipeline
│   ├── data_utils.py        # Dataset loaders & Palette handling
│   ├── prepare_dog_dataset.py # Splitting/Filtering Mini-ImageNet
│   └── precompute_tokens.py # Offline tokenization
├── training/                # Training Scripts
│   ├── teacher/             # Teacher training (Autoregressive)
│   ├── student/             # Knowledge Distillation (Hybrid)
│   └── vqvae/               # VQ-VAE training
├── inference/               # Inference & Benchmarking
│   ├── inference_speculative.py # Core Speculative Decoding Loop
│   ├── benchmark_hybrid.py      # Comprehensive Benchmarking Suite
│   └── benchmark_teacher.py     # Teacher-only Benchmarks
└── verify/                  # Unit Tests & Sanity Checks
    ├── verify_teacher.py    # Overfitting checks
    └── verify_vqvae.py      # Reconstruction checks
```

## 3. Example Commands for Execution

### Training the Teacher
```bash
python training/teacher/train_teacher_unconditional.py
```

### Distilling a Student (Hybrid)
```bash
# Student ID: 1=10L, 2=8L, 3=6L, 4=4L
python training/student/train_distill_hybrid.py --student_id 4
```

### Running Benchmarks (Speed & Quality)
```bash
python inference/benchmark_hybrid.py
```

### Speculative Decoding (Demo)
```bash
python inference/inference_speculative_unconditional.py
```

## 4. Results & Observations

We benchmarked the systems on **Unconditional Dog Generation**.

### Quality (FID/IS)
The Students achieved comparable image quality to the Teacher, with the 8L student slightly outperforming the Teacher in Inception Score (likely due to smoothing).

| Model | FID (Lower is Better) | IS (Higher is Better) |
| :--- | :--- | :--- |
| **Teacher** | 171.03 | 5.40 |
| **Student 1 (10L)** | 161.63 | 5.39 |
| **Student 2 (8L)** | 160.66 | **5.49** |
| **Student 4 (4L)** | 207.45 | 4.25 |

### Speculative Performance
Despite the Students being **5x faster** in standalone mode, Speculative Decoding resulted in a net **slowdown**.

| Model | Standalone Speed | Speculative Speed | Acceptance Rate | Speedup |
| :--- | :--- | :--- | :--- | :--- |
| **Teacher** | 74.42 tok/s | N/A | N/A | 1.0x |
| **Student 4L** | **373.18 tok/s** | 30.65 tok/s | **4.00%** | **0.41x** |

**Observations**:
*   **The "Frankenstein" Effect**: The Teacher is highly overfit to its specific training trajectories. While the Student learns a valid "general" distribution of dogs, it often picks a token (e.g., "brown ear") that the Teacher (expecting "black ear") assigns near-zero probability to.
*   **Rejection Cost**: With an acceptance rate of ~4%, the system spends most of its time rejecting drafts and reverting the KV cache, which is computationally expensive.
*   **Conclusion**: Efficient Speculative Decoding requires **Manifold Alignment**. The Student must not just be "good"; it must be "identical" to the Teacher. Future work should focus on **On-Policy Distillation (DAGGER)** or **Regularizing the Teacher** to be less strict.

## 5. Weights & Biases
[Link to WandB Project Board](https://wandb.ai/ltc2125-columbia-university/speculative-decoding-distillation)

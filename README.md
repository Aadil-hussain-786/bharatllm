#  Bharat-LLM: The Sovereign Model Suite 🇮🇳

**Empowering the Future of Indian AI with frontier-scale reasoning and hardware-efficient architectures.**

---

## 1. Bharat-LLM: 100B Parameter Sovereign Model

**Bharat-LLM** is a frontier-scale, **100 Billion Parameter** Large Language Model specifically engineered for the linguistic diversity, cultural nuances, and computational requirements of the Indian subcontinent.

<p align="center">
    <img src="docs/assets/Bharat_Labs_professional_202604191607.jpeg" width="600" alt="Bharat Labs Infrastructure">
</p>

### Key Features (100B)
*   **100B MoE Architecture**: Features 64 specialized experts with **Top-2 Gating**.
*   **TPU-Native Training**: Optimized for **Google Colab Free Tier TPU v2-8 / v3-8** using `torch_xla`.
*   **Indic Vocabulary Expansion**: Custom tokenizer with 32,000+ newly injected tokens.
*   **Elite Knowledge Distillation**: Trained using high-intelligence "Teachers" via Groq LPU infrastructure.

---

## 2. Bharat-3B Smart-Core

**A revolutionary 3B parameter language model with Deep Equilibrium Layers, Recurrent Memory, and Mixture of Softmaxes — optimized for efficiency.**

### Architecture Highlights (3B)
- **Deep Equilibrium (DEQ)**: Single weight-tied block with 100+ effective layers via fixed-point iteration.
- **Recurrent Memory Transformer (RMT)**: Theoretically infinite context window (128k+ tokens).
- **Mixture of Softmaxes (MoS)**: 10 sub-expert output heads for superior expressiveness.
- **Recursive Distillation**: Learning from Gemini, Llama 405B, and GPT-4 consensus.

### Model Specifications (3B)

| Parameter | Value |
|-----------|-------|
| Total Parameters | 3.0B |
| Hidden Dimension | 2560 |
| Attention Heads | 32 |
| DEQ Iterations | 20 (effective 100+ layers) |
| Vocab Size | 50,257 |
| Context Length | 128,000 tokens |
| MoS Experts | 10 |
| Memory Tokens (RMT) | 128 |

---

##  Project Structure

```text
bharat_llm/
├── configs/            # Model Hyperparameters & Configs
├── docs/assets/        # Project Visualizations & Media
├── logs/               # Training & Evaluation Logs
├── notebooks/          # Interactive Training (Colab)
├── scripts/            # Entry point scripts
├── src/                # Modular Source Code (JAX/PyTorch)
└── requirements.txt    # Unified Dependencies
```

##  License
Bharat-LLM is released under the **Apache 2.0 License** and includes **Proprietary Patent Pending (DPIIT Startup India)** components for the Smart-Core architecture.

---
<p align="center">
  <b>Built with ❤️ by Bharat Labs</b><br>
  <i>"Shaping the future of Indian Intelligence, one parameter at a time."</i>
</p>

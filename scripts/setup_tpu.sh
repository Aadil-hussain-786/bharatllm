#!/bin/bash
# ============================================
# Bharat-3B Smart-Core: TPU Cluster Setup
# ============================================
# Phase 1, Step 1.1: Configure JAX/Flax on TPU v3-8
# Run this on Google Cloud TPU VM or Colab

set -e

echo "🚀 Setting up Bharat-3B TPU Environment..."
echo "============================================"

# 1. Detect TPU type
echo "[1/6] Detecting TPU hardware..."
if python3 -c "import jax; print(jax.devices())" 2>/dev/null | grep -q "TPU"; then
    echo "✅ TPU detected!"
    python3 -c "import jax; print(f'Devices: {jax.devices()}')"
    python3 -c "import jax; print(f'Device count: {jax.device_count()}')"
else
    echo "⚠️ No TPU found. Checking for GPU..."
    if python3 -c "import jax; print(jax.devices())" 2>/dev/null | grep -q "gpu"; then
        echo "GPU detected. Training will use GPU fallback."
    else
        echo "CPU-only mode. Training will be slow."
    fi
fi

# 2. Install core dependencies
echo ""
echo "[2/6] Installing JAX with TPU support..."
pip install --quiet --upgrade pip

# JAX for TPU
pip install --quiet "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html

# 3. Install ML libraries
echo ""
echo "[3/6] Installing ML libraries..."
pip install --quiet \
    flax>=0.8.0 \
    optax>=0.2.0 \
    orbax-checkpoint>=0.5.0 \
    equinox>=0.11.0 \
    chex>=0.1.8 \
    ml-collections>=0.1.1 \
    einops>=0.8.0

# 4. Install data & tokenizer tools
echo ""
echo "[4/6] Installing data pipeline tools..."
pip install --quiet \
    sentencepiece>=0.2.0 \
    tokenizers>=0.19.0 \
    datasets>=2.20.0 \
    grain>=0.2.0

# 5. Install teacher model APIs
echo ""
echo "[5/6] Installing teacher model APIs..."
pip install --quiet \
    google-generativeai>=0.7.0 \
    groq>=0.9.0 \
    openai>=1.35.0

# 6. Install monitoring & eval
echo ""
echo "[6/6] Installing monitoring tools..."
pip install --quiet \
    wandb>=0.17.0 \
    tensorboard>=2.17.0 \
    rich>=13.7.0 \
    tqdm>=4.66.0

# Verify installation
echo ""
echo "============================================"
echo "🔍 Verifying installation..."
python3 << 'VERIFY'
import sys
print(f"Python: {sys.version}")

import jax
print(f"JAX: {jax.__version__}")
print(f"Devices: {jax.devices()}")
print(f"Default backend: {jax.default_backend()}")

import flax
print(f"Flax: {flax.__version__}")

import optax
print(f"Optax: {optax.__version__}")

import jax.numpy as jnp
# Quick compute test
x = jnp.ones((1024, 1024))
y = jnp.matmul(x, x)
print(f"Matrix multiply test: shape={y.shape}, sum={float(jnp.sum(y))}")

# Memory info (TPU)
for device in jax.devices():
    print(f"Device: {device.platform} | {device}")

print("\n✅ All systems operational! Ready for Bharat-3B training.")
VERIFY

echo ""
echo "🎉 TPU setup complete!"
echo "============================================"

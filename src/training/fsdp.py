"""
Bharat-3B Smart-Core: FSDP (Fully Sharded Data Parallel)
=========================================================
Distributes model weights across all 8 TPU cores for memory efficiency.

FSDP Strategy:
    - Shard weights across all TPU cores
    - Each core holds only 1/8th of the total parameters
    - All-gather before forward pass, reduce-scatter after backward
    - With 128GB total HBM (8 × 16GB), we can fit 3B+ parameters

Mesh Layout:
    TPU v3-8 has 8 cores arranged in a 2D mesh.
    We use (data_parallel=1, model_parallel=8) for maximum sharding.
"""

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec, NamedSharding
from jax.experimental import mesh_utils
import flax.linen as nn
from typing import Dict, Any, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


def create_device_mesh(
    mesh_shape: Tuple[int, ...] = (1, 8),
    axis_names: Tuple[str, ...] = ("dp", "mp"),
) -> Mesh:
    """
    Create a JAX device mesh for distributed training.
    
    Args:
        mesh_shape: Shape of the device mesh.
                    (1, 8) = pure model parallelism
                    (2, 4) = 2-way data parallel, 4-way model parallel
                    (8, 1) = pure data parallelism
        axis_names: Names for mesh axes.
    
    Returns:
        JAX Mesh object.
    """
    devices = jax.devices()
    num_devices = len(devices)

    logger.info(f"Found {num_devices} devices: {[d.platform for d in devices]}")

    total_mesh_size = 1
    for s in mesh_shape:
        total_mesh_size *= s

    if total_mesh_size != num_devices:
        logger.warning(
            f"Mesh shape {mesh_shape} requires {total_mesh_size} devices "
            f"but found {num_devices}. Adjusting..."
        )
        # Fallback: pure model parallelism across available devices
        mesh_shape = (1, num_devices)

    device_mesh = mesh_utils.create_device_mesh(mesh_shape)
    mesh = Mesh(device_mesh, axis_names=axis_names)

    logger.info(f"Created mesh: shape={mesh_shape}, axes={axis_names}")
    return mesh


def get_sharding_rules() -> Dict[str, PartitionSpec]:
    """
    Define how model parameters are sharded across the mesh.
    
    Sharding Strategy:
        - Embedding: shard vocab dimension across model-parallel axis
        - Attention Q/K/V: shard head dimension
        - FFN: shard intermediate dimension
        - LayerNorm/bias: replicate (no sharding)
    
    Returns:
        Dict mapping parameter name patterns to PartitionSpec.
    """
    return {
        # Embeddings
        "token_embedding/embedding": PartitionSpec("mp", None),

        # Attention projections
        "q_proj/kernel": PartitionSpec(None, "mp"),
        "k_proj/kernel": PartitionSpec(None, "mp"),
        "v_proj/kernel": PartitionSpec(None, "mp"),
        "o_proj/kernel": PartitionSpec("mp", None),

        # FFN (SwiGLU)
        "gate_proj/kernel": PartitionSpec(None, "mp"),
        "up_proj/kernel": PartitionSpec(None, "mp"),
        "down_proj/kernel": PartitionSpec("mp", None),

        # MoS expert projections
        "expert_projections/kernel": PartitionSpec(None, "mp"),
        "expert_transform_*/kernel": PartitionSpec(None, "mp"),

        # Memory projections (RMT)
        "mem_read_*/kernel": PartitionSpec(None, "mp"),
        "mem_write_*/kernel": PartitionSpec(None, "mp"),

        # Norms (replicated)
        "*/weight": PartitionSpec(None),
        "*/bias": PartitionSpec(None),

        # Default: replicate
        "default": PartitionSpec(None),
    }


def shard_params(
    params: Dict,
    mesh: Mesh,
    rules: Optional[Dict[str, PartitionSpec]] = None,
) -> Dict:
    """
    Apply sharding rules to model parameters.
    
    Args:
        params: Model parameter PyTree.
        mesh: JAX device mesh.
        rules: Sharding rules. If None, uses default rules.
    
    Returns:
        Sharded parameter PyTree.
    """
    if rules is None:
        rules = get_sharding_rules()

    def _get_partition_spec(path: str) -> PartitionSpec:
        """Find matching partition spec for parameter path."""
        for pattern, spec in rules.items():
            if pattern == "default":
                continue
            # Simple pattern matching
            if "*" in pattern:
                prefix = pattern.split("*")[0]
                suffix = pattern.split("*")[-1]
                if path.startswith(prefix) and path.endswith(suffix):
                    return spec
            elif pattern in path:
                return spec
        return rules.get("default", PartitionSpec(None))

    def _shard_leaf(path_parts, leaf):
        """Shard a single parameter leaf."""
        path = "/".join(str(p) for p in path_parts)
        spec = _get_partition_spec(path)

        # Verify spec dimensions match parameter dimensions
        if len(spec) > len(leaf.shape):
            logger.warning(
                f"PartitionSpec {spec} has more dims than param {path} "
                f"(shape={leaf.shape}). Using replicated."
            )
            spec = PartitionSpec(*([None] * len(leaf.shape)))

        sharding = NamedSharding(mesh, spec)
        return jax.device_put(leaf, sharding)

    return jax.tree_util.tree_map_with_path(
        lambda path, leaf: _shard_leaf(
            [str(p) for p in path], leaf
        ),
        params,
    )


def shard_batch(
    batch: Dict[str, jnp.ndarray],
    mesh: Mesh,
) -> Dict[str, jnp.ndarray]:
    """
    Shard input batch across data-parallel axis.
    
    Args:
        batch: Dict with "input_ids", "attention_mask", "labels", etc.
        mesh: JAX device mesh.
    
    Returns:
        Sharded batch.
    """
    data_spec = PartitionSpec("dp", None)  # Shard batch dim

    sharded_batch = {}
    for key, value in batch.items():
        sharding = NamedSharding(mesh, data_spec)
        sharded_batch[key] = jax.device_put(value, sharding)

    return sharded_batch


class FSDPTrainState:
    """
    Training state with FSDP sharding.
    
    Wraps model parameters, optimizer state, and step counter
    with proper sharding for distributed training.
    """

    def __init__(
        self,
        params: Dict,
        optimizer_state: Any,
        step: int,
        mesh: Mesh,
    ):
        self.params = params
        self.optimizer_state = optimizer_state
        self.step = step
        self.mesh = mesh

    @classmethod
    def create(
        cls,
        model: nn.Module,
        optimizer: Any,
        rng_key: jnp.ndarray,
        mesh: Mesh,
        dummy_input: jnp.ndarray,
    ) -> "FSDPTrainState":
        """
        Initialize training state with sharded parameters.
        
        Args:
            model: Flax model.
            optimizer: Optax optimizer.
            rng_key: Random key for initialization.
            mesh: Device mesh.
            dummy_input: Dummy input for shape inference.
        
        Returns:
            FSDPTrainState with sharded params.
        """
        # Initialize parameters
        variables = model.init(rng_key, dummy_input)
        params = variables.get("params", variables)

        logger.info(f"Initialized model with {_count_params(params):,} parameters")

        # Shard parameters
        with mesh:
            sharded_params = shard_params(params, mesh)

        # Initialize optimizer state
        opt_state = optimizer.init(sharded_params)

        return cls(
            params=sharded_params,
            optimizer_state=opt_state,
            step=0,
            mesh=mesh,
        )


def _count_params(params: Dict) -> int:
    """Count total parameters in a PyTree."""
    return sum(p.size for p in jax.tree_util.tree_leaves(params))


def compute_memory_usage(params: Dict) -> Dict[str, float]:
    """
    Compute memory usage per component (in MB).
    
    Args:
        params: Model parameters.
    
    Returns:
        Dict with per-component memory in MB.
    """
    memory = {}

    def _accumulate(path, leaf):
        component = str(path[0]) if path else "other"
        size_mb = leaf.size * leaf.dtype.itemsize / (1024 ** 2)
        memory[component] = memory.get(component, 0) + size_mb

    jax.tree_util.tree_map_with_path(
        lambda path, x: _accumulate([str(p) for p in path], x),
        params,
    )

    memory["total_mb"] = sum(memory.values())
    memory["total_gb"] = memory["total_mb"] / 1024
    return memory

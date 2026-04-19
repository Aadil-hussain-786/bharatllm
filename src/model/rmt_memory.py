"""
Bharat-3B Smart-Core: Recurrent Memory Transformer (RMT)
=========================================================
Enables theoretically infinite context window by segmenting long
sequences and passing memory tokens between segments.

Key Idea:
    Instead of processing 128k tokens at once (impossible on TPU v3),
    we split into segments of ~16k tokens. Between segments, special
    "memory tokens" carry information forward, acting as a compressed
    summary of everything seen so far.

Memory Flow:
    Segment 1: [MEM_READ] + [tokens_1...16k] + [MEM_WRITE] -> Memory_1
    Segment 2: [Memory_1] + [tokens_16k...32k] + [MEM_WRITE] -> Memory_2
    ...
    Segment 8: [Memory_7] + [tokens_112k...128k]             -> Final Output

Advantages:
    1. Linear memory scaling with context length (not quadratic!)
    2. Each segment is processed independently with full attention
    3. Memory tokens act as a "neural compressed context"
    4. Compatible with DEQ (memory just adds to the input)

References:
    - Bulatov et al., "Recurrent Memory Transformer" (NeurIPS 2022)
    - Bulatov et al., "Scaling Transformer to 1M tokens" (2023)
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Optional, Tuple


class MemoryReadModule(nn.Module):
    """
    Reads from memory tokens using cross-attention.
    
    Takes the current segment's hidden states and attends to
    memory tokens from previous segments. This injects long-range
    context information into the current segment's processing.
    
    Attributes:
        hidden_size: Model dimension.
        num_memory_tokens: Number of memory slots.
        num_heads: Number of attention heads for memory reading.
        use_cross_attention: Whether to use cross-attention (True) or concatenation.
        dtype: Computation dtype.
    """
    hidden_size: int = 2560
    num_memory_tokens: int = 128
    num_heads: int = 8
    use_cross_attention: bool = True
    dtype: jnp.dtype = jnp.bfloat16

    @nn.compact
    def __call__(
        self,
        hidden_states: jnp.ndarray,
        memory: Optional[jnp.ndarray] = None,
        deterministic: bool = True,
    ) -> jnp.ndarray:
        """
        Read from memory and inject into hidden states.
        
        Args:
            hidden_states: (batch, seq_len, hidden_size) - current segment
            memory: (batch, num_memory_tokens, hidden_size) - from previous segment
            deterministic: Whether in eval mode.
        
        Returns:
            Enhanced hidden states with memory-injected context.
        """
        if memory is None:
            # First segment: initialize memory tokens as learnable parameters
            memory_init = self.param(
                "initial_memory",
                nn.initializers.normal(stddev=0.02),
                (1, self.num_memory_tokens, self.hidden_size),
            )
            batch_size = hidden_states.shape[0]
            memory = jnp.broadcast_to(
                memory_init, (batch_size, self.num_memory_tokens, self.hidden_size)
            ).astype(self.dtype)

        if self.use_cross_attention:
            # Cross-attention: hidden_states attend to memory
            head_dim = self.hidden_size // self.num_heads

            # Q from hidden states, K/V from memory
            q = nn.Dense(
                self.hidden_size, use_bias=False, dtype=self.dtype, name="mem_read_q"
            )(hidden_states)
            k = nn.Dense(
                self.hidden_size, use_bias=False, dtype=self.dtype, name="mem_read_k"
            )(memory)
            v = nn.Dense(
                self.hidden_size, use_bias=False, dtype=self.dtype, name="mem_read_v"
            )(memory)

            batch_size, seq_len, _ = hidden_states.shape
            mem_len = memory.shape[1]

            # Reshape for multi-head attention
            q = q.reshape(batch_size, seq_len, self.num_heads, head_dim)
            k = k.reshape(batch_size, mem_len, self.num_heads, head_dim)
            v = v.reshape(batch_size, mem_len, self.num_heads, head_dim)

            # Transpose: (batch, heads, seq/mem_len, head_dim)
            q = jnp.transpose(q, (0, 2, 1, 3))
            k = jnp.transpose(k, (0, 2, 1, 3))
            v = jnp.transpose(v, (0, 2, 1, 3))

            # Attention
            scale = jnp.sqrt(head_dim).astype(self.dtype)
            attn_weights = jnp.matmul(q, jnp.transpose(k, (0, 1, 3, 2))) / scale
            attn_weights = jax.nn.softmax(attn_weights.astype(jnp.float32), axis=-1)
            attn_weights = attn_weights.astype(self.dtype)

            mem_context = jnp.matmul(attn_weights, v)
            mem_context = jnp.transpose(mem_context, (0, 2, 1, 3))
            mem_context = mem_context.reshape(batch_size, seq_len, self.hidden_size)

            # Project and add to hidden states
            mem_proj = nn.Dense(
                self.hidden_size, use_bias=False, dtype=self.dtype, name="mem_read_out"
            )(mem_context)

            # Gated addition
            gate = nn.Dense(
                self.hidden_size, dtype=self.dtype, name="mem_read_gate"
            )(hidden_states)
            gate = jax.nn.sigmoid(gate)

            return hidden_states + gate * mem_proj
        else:
            # Simple concatenation: prepend memory tokens to sequence
            return jnp.concatenate([memory, hidden_states], axis=1)


class MemoryWriteModule(nn.Module):
    """
    Writes to memory tokens after processing a segment.
    
    Compresses the current segment's information into memory tokens
    that will be passed to the next segment. This is the "write" side
    of the memory system.
    
    Update Strategies:
        - "gated": Learnable gate between old and new memory (default)
        - "additive": Add new information to existing memory
        - "replace": Fully replace memory (no carry-over)
    
    Attributes:
        hidden_size: Model dimension.
        num_memory_tokens: Number of memory slots.
        update_type: Memory update strategy.
        dtype: Computation dtype.
    """
    hidden_size: int = 2560
    num_memory_tokens: int = 128
    update_type: str = "gated"
    dtype: jnp.dtype = jnp.bfloat16

    @nn.compact
    def __call__(
        self,
        hidden_states: jnp.ndarray,
        old_memory: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        """
        Write segment information to memory.
        
        Args:
            hidden_states: (batch, seq_len, hidden_size) - processed segment
            old_memory: (batch, num_memory_tokens, hidden_size) - previous memory
        
        Returns:
            Updated memory of shape (batch, num_memory_tokens, hidden_size).
        """
        batch_size = hidden_states.shape[0]

        # Compress segment into memory-sized representation
        # Using learned pooling via cross-attention
        memory_queries = self.param(
            "memory_write_queries",
            nn.initializers.normal(stddev=0.02),
            (1, self.num_memory_tokens, self.hidden_size),
        )
        memory_queries = jnp.broadcast_to(
            memory_queries, (batch_size, self.num_memory_tokens, self.hidden_size)
        ).astype(self.dtype)

        # Cross-attention: memory queries attend to hidden states
        num_heads = 8
        head_dim = self.hidden_size // num_heads
        seq_len = hidden_states.shape[1]

        q = nn.Dense(
            self.hidden_size, use_bias=False, dtype=self.dtype, name="mem_write_q"
        )(memory_queries)
        k = nn.Dense(
            self.hidden_size, use_bias=False, dtype=self.dtype, name="mem_write_k"
        )(hidden_states)
        v = nn.Dense(
            self.hidden_size, use_bias=False, dtype=self.dtype, name="mem_write_v"
        )(hidden_states)

        q = q.reshape(batch_size, self.num_memory_tokens, num_heads, head_dim)
        k = k.reshape(batch_size, seq_len, num_heads, head_dim)
        v = v.reshape(batch_size, seq_len, num_heads, head_dim)

        q = jnp.transpose(q, (0, 2, 1, 3))
        k = jnp.transpose(k, (0, 2, 1, 3))
        v = jnp.transpose(v, (0, 2, 1, 3))

        scale = jnp.sqrt(head_dim).astype(self.dtype)
        attn_weights = jnp.matmul(q, jnp.transpose(k, (0, 1, 3, 2))) / scale
        attn_weights = jax.nn.softmax(attn_weights.astype(jnp.float32), axis=-1)
        attn_weights = attn_weights.astype(self.dtype)

        new_memory = jnp.matmul(attn_weights, v)
        new_memory = jnp.transpose(new_memory, (0, 2, 1, 3))
        new_memory = new_memory.reshape(
            batch_size, self.num_memory_tokens, self.hidden_size
        )

        # Project
        new_memory = nn.Dense(
            self.hidden_size, use_bias=False, dtype=self.dtype, name="mem_write_out"
        )(new_memory)

        # Apply memory update strategy
        if old_memory is None or self.update_type == "replace":
            return new_memory

        if self.update_type == "gated":
            # Learned gate between old and new memory
            gate_input = jnp.concatenate([old_memory, new_memory], axis=-1)
            gate = nn.Dense(
                self.hidden_size, dtype=self.dtype, name="mem_update_gate"
            )(gate_input)
            gate = jax.nn.sigmoid(gate)
            return gate * new_memory + (1.0 - gate) * old_memory

        elif self.update_type == "additive":
            # Simple additive update with normalization
            combined = old_memory + new_memory
            return nn.RMSNorm(epsilon=1e-5, dtype=self.dtype, name="mem_norm")(combined)

        else:
            raise ValueError(f"Unknown update type: {self.update_type}")


class RMTMemory(nn.Module):
    """
    Recurrent Memory Transformer — Full Memory System.
    
    Wraps the read/write modules and handles the segment-level
    recurrence that enables 128k+ token context.
    
    Usage:
        1. Split input into segments
        2. For each segment:
           a. Memory Read: inject previous memory into current segment
           b. DEQ Processing: find fixed point of transformer block
           c. Memory Write: compress current segment into memory
        3. Concatenate outputs from all segments
    
    Attributes:
        hidden_size: Model dimension.
        num_memory_tokens: Number of memory slots per segment.
        num_segments: Number of segments to split input into.
        segment_length: Tokens per segment.
        memory_update: Update strategy for memory.
        use_cross_attention: Whether to use cross-attention for memory read.
        dtype: Computation dtype.
    """
    hidden_size: int = 2560
    num_memory_tokens: int = 128
    num_segments: int = 8
    segment_length: int = 16_000
    memory_update: str = "gated"
    use_cross_attention: bool = True
    dtype: jnp.dtype = jnp.bfloat16

    def setup(self):
        self.memory_read = MemoryReadModule(
            hidden_size=self.hidden_size,
            num_memory_tokens=self.num_memory_tokens,
            use_cross_attention=self.use_cross_attention,
            dtype=self.dtype,
            name="memory_read",
        )
        self.memory_write = MemoryWriteModule(
            hidden_size=self.hidden_size,
            num_memory_tokens=self.num_memory_tokens,
            update_type=self.memory_update,
            dtype=self.dtype,
            name="memory_write",
        )

    def segment_input(
        self,
        hidden_states: jnp.ndarray,
    ) -> list[jnp.ndarray]:
        """
        Split input into segments for sequential processing.
        
        Args:
            hidden_states: (batch, seq_len, hidden_size)
        
        Returns:
            List of segments, each (batch, segment_length, hidden_size).
        """
        batch_size, seq_len, hidden_size = hidden_states.shape
        num_segments = min(
            self.num_segments,
            (seq_len + self.segment_length - 1) // self.segment_length,
        )

        segments = []
        for i in range(num_segments):
            start = i * self.segment_length
            end = min(start + self.segment_length, seq_len)
            segment = hidden_states[:, start:end, :]

            # Pad if necessary
            if segment.shape[1] < self.segment_length:
                pad_len = self.segment_length - segment.shape[1]
                padding = jnp.zeros(
                    (batch_size, pad_len, hidden_size),
                    dtype=segment.dtype,
                )
                segment = jnp.concatenate([segment, padding], axis=1)

            segments.append(segment)

        return segments

    def process_segment(
        self,
        segment: jnp.ndarray,
        memory: Optional[jnp.ndarray],
        process_fn,
        deterministic: bool = True,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Process a single segment with memory read/write.
        
        Args:
            segment: (batch, segment_length, hidden_size)
            memory: Previous memory or None for first segment
            process_fn: The DEQ processing function
            deterministic: Whether in eval mode
        
        Returns:
            Tuple of (processed_segment, updated_memory).
        """
        # Step 1: Memory Read — inject context from previous segments
        enhanced = self.memory_read(
            segment,
            memory=memory,
            deterministic=deterministic,
        )

        # Step 2: Core processing (DEQ layer handles this)
        processed = process_fn(enhanced)

        # Step 3: Memory Write — compress this segment into memory
        new_memory = self.memory_write(
            processed,
            old_memory=memory,
        )

        return processed, new_memory

    def __call__(
        self,
        hidden_states: jnp.ndarray,
        process_fn,
        deterministic: bool = True,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Full RMT forward pass with segment-level recurrence.
        
        Args:
            hidden_states: (batch, seq_len, hidden_size) - full input
            process_fn: Function to process each segment (DEQ layer)
            deterministic: Whether in eval mode
        
        Returns:
            Tuple of (processed_output, final_memory).
            processed_output: (batch, seq_len, hidden_size)
            final_memory: (batch, num_memory_tokens, hidden_size)
        """
        batch_size, seq_len, hidden_size = hidden_states.shape

        # Check if segmentation is needed
        if seq_len <= self.segment_length:
            # Short sequence: no segmentation needed
            enhanced = self.memory_read(
                hidden_states, memory=None, deterministic=deterministic
            )
            processed = process_fn(enhanced)
            memory = self.memory_write(processed, old_memory=None)
            return processed, memory

        # Split into segments
        segments = self.segment_input(hidden_states)

        # Process segments sequentially with memory propagation
        memory = None
        processed_segments = []

        for i, segment in enumerate(segments):
            processed, memory = self.process_segment(
                segment=segment,
                memory=memory,
                process_fn=process_fn,
                deterministic=deterministic,
            )
            processed_segments.append(processed)

        # Concatenate all processed segments
        output = jnp.concatenate(processed_segments, axis=1)

        # Trim to original sequence length
        output = output[:, :seq_len, :]

        return output, memory

import torch
from torch import nn
import torch.nn.functional as F
from transformers import Gemma3TextConfig

from nanovllm.layers.activation import SiluAndMul
from nanovllm.layers.attention import Attention
from nanovllm.layers.layernorm import RMSNorm
from nanovllm.layers.linear import QKVParallelLinear, MergedColumnParallelLinear, RowParallelLinear
from nanovllm.layers.rotary_embedding import get_rope
from nanovllm.layers.embed_head import VocabParallelEmbedding, ParallelLMHead


class GeluAndMul(nn.Module):
    """Gelu activation with gating, using PyTorch's tanh approximation."""

    def __init__(self):
        super().__init__()

    @torch.compile
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, y = x.chunk(2, -1)
        return F.gelu(x, approximate="tanh") * y


class Gemma3Attention(nn.Module):

    def __init__(
        self,
        config: Gemma3TextConfig,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        rope_theta: float = 1000000,
        rope_scaling: tuple | None = None,
        max_position_embeddings: int = 131072,
        bias: bool = False,
        bias_o_proj: bool = False,
        query_pre_attn_scalar: int | None = None,
        layer_idx: int = 0,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.layer_idx = layer_idx
        # Gemma3 uses layer_types to determine sliding window vs global attention
        self.layer_type = config.layer_types[layer_idx] if hasattr(config, "layer_types") and config.layer_types else None
        self.sliding_window = getattr(config, "sliding_window", 4096) if self.layer_type == "sliding_attention" else None
        tp_size = 1
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        assert self.total_num_kv_heads % tp_size == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        # Gemma3 uses explicit head_dim from config
        head_dim = getattr(config, "head_dim", None)
        if head_dim is None:
            head_dim = self.hidden_size // self.total_num_heads
        self.head_dim = head_dim
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        # Gemma3 uses query_pre_attn_scalar for scaling instead of head_dim^-0.5
        if query_pre_attn_scalar is not None:
            self.scaling = query_pre_attn_scalar**-0.5
        else:
            self.scaling = self.head_dim**-0.5
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings

        self.qkv_proj = QKVParallelLinear(
            hidden_size=hidden_size,
            head_size=self.head_dim,
            total_num_heads=self.total_num_heads,
            total_num_kv_heads=self.total_num_kv_heads,
            bias=bias,
        )

        self.o_proj = RowParallelLinear(
            input_size=self.total_num_heads * self.head_dim,
            output_size=hidden_size,
            bias=bias_o_proj,
        )

        # Gemma3 uses QK normalization
        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)

        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=self.max_position_embeddings,
            base=self.rope_theta,
            rope_scaling=rope_scaling,
        )

        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            self.num_kv_heads,
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        qkv = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q = q.view(-1, self.num_heads, self.head_dim)
        k = k.view(-1, self.num_kv_heads, self.head_dim)
        v = v.view(-1, self.num_kv_heads, self.head_dim)
        # Apply QK normalization before rotary embedding
        q = self.q_norm(q)
        k = self.k_norm(k)
        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v)
        output = self.o_proj(attn_output.flatten(1, -1))
        return output


class Gemma3MLP(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_activation: str,
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            input_size=hidden_size,
            output_sizes=[intermediate_size] * 2,
            bias=bias,
        )
        self.down_proj = RowParallelLinear(
            input_size=intermediate_size,
            output_size=hidden_size,
            bias=bias,
        )
        # Gemma3 uses gelu_pytorch_tanh activation
        if hidden_activation == "gelu_pytorch_tanh":
            self.act_fn = GeluAndMul()
        elif hidden_activation == "silu":
            self.act_fn = SiluAndMul()
        else:
            raise ValueError(f"Unsupported activation: {hidden_activation}")

    def forward(self, x):
        x = self.gate_up_proj(x)
        x = self.act_fn(x)
        x = self.down_proj(x)
        return x


class Gemma3DecoderLayer(nn.Module):

    def __init__(
        self,
        config: Gemma3TextConfig,
        layer_idx: int = 0,
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx
        # Gemma3 uses different rope_theta for global vs local attention
        layer_type = config.layer_types[layer_idx] if hasattr(config, "layer_types") and config.layer_types else None
        if layer_type == "sliding_attention":
            rope_theta = getattr(config, "rope_local_base_freq", 10000)
        else:
            rope_theta = getattr(config, "rope_theta", 1000000)
        rope_scaling = getattr(config, "rope_scaling", None)
        max_position_embeddings = getattr(config, "max_position_embeddings", 131072)
        attention_bias = getattr(config, "attention_bias", False)
        query_pre_attn_scalar = getattr(config, "query_pre_attn_scalar", 256)

        self.self_attn = Gemma3Attention(
            config=config,
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=getattr(config, "num_key_value_heads",
                                 config.num_attention_heads),
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            max_position_embeddings=max_position_embeddings,
            bias=attention_bias,
            bias_o_proj=attention_bias,
            query_pre_attn_scalar=query_pre_attn_scalar,
            layer_idx=layer_idx,
        )
        
        hidden_activation = getattr(config, "hidden_activation", "gelu_pytorch_tanh")
        self.mlp = Gemma3MLP(
            hidden_size=self.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_activation=hidden_activation,
            bias=False,
        )
        
        # Gemma3 has 4 layer norms: pre/post attention and pre/post MLP
        self.input_layernorm = RMSNorm(config.hidden_size,
                                       eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size,
                                                eps=config.rms_norm_eps)
        self.pre_feedforward_layernorm = RMSNorm(config.hidden_size,
                                                  eps=config.rms_norm_eps)
        self.post_feedforward_layernorm = RMSNorm(config.hidden_size,
                                                   eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Self Attention with pre and post normalization
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(
                hidden_states, residual)
        
        hidden_states = self.self_attn(positions=positions,
                                       hidden_states=hidden_states)
        # Post attention normalization (Gemma3 specific)
        hidden_states = self.post_attention_layernorm(hidden_states)

        # MLP with pre and post normalization
        hidden_states, residual = self.pre_feedforward_layernorm(
            hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        # Post feedforward normalization (Gemma3 specific)
        hidden_states = self.post_feedforward_layernorm(hidden_states)
        
        return hidden_states, residual


class Gemma3Model(nn.Module):

    def __init__(
        self,
        config: Gemma3TextConfig,
    ) -> None:
        super().__init__()
        self.config = config
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([Gemma3DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        # Gemma3 normalizes embeddings by sqrt(hidden_size)
        self.normalizer = config.hidden_size**0.5

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        # Gemma3 multiplies embeddings by sqrt(hidden_size)
        hidden_states = hidden_states * self.normalizer
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(positions, hidden_states, residual)
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


class Gemma3ForCausalLM(nn.Module):
    packed_modules_mapping = {
        "q_proj": ("qkv_proj", "q"),
        "k_proj": ("qkv_proj", "k"),
        "v_proj": ("qkv_proj", "v"),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    def __init__(
        self,
        config: Gemma3TextConfig
    ) -> None:
        super().__init__()
        self.config = config
        self.model = Gemma3Model(config)
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)
        # Gemma3 has final_logit_softcapping disabled by default (None)
        self.final_logit_softcapping = getattr(config, "final_logit_softcapping", None)
        if config.tie_word_embeddings:
            self.lm_head.weight.data = self.model.embed_tokens.weight.data

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        model_output = self.model(input_ids, positions)
        return model_output

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        logits = self.lm_head(hidden_states)
        # Apply final logit softcapping if configured
        if self.final_logit_softcapping is not None and logits is not None:
            logits = logits / self.final_logit_softcapping
            logits = torch.tanh(logits)
            logits = logits * self.final_logit_softcapping
        return logits

# -*- coding: utf-8 -*-
"""ARCHIVED: 为历史 two-Blackwell 环境注册 local transformer-impl 桥。

背景（Day-01 4.4）：slime 训练栈默认 transformer-impl=transformer_engine，
本机未装 TE，改用 --transformer-impl local。local spec 的 mcore 权重命名里
LayerNorm 是独立的 input_layernorm / post_attention_layernorm，
而 mbridge Qwen2Bridge 的映射是 TE 风格（fused 在 linear_qkv/linear_fc1 内），
因此需要为 local spec 补充这两条映射。Qwen2/Qwen3 权重布局一致，其余沿用 Qwen2Bridge。
"""
from mbridge.core import register_model
from mbridge.models.qwen2 import Qwen2Bridge


@register_model("qwen3")
class LocalQwen3Bridge(Qwen2Bridge):
    _OTHER_MAPPING = Qwen2Bridge._OTHER_MAPPING | {
        "input_layernorm.weight": [
            "model.layers.{layer_number}.input_layernorm.weight",
        ],
    }
    # megatron-core 0.16 local spec 用 pre_mlp_layernorm（对应 HF post_attention_layernorm）
    _MLP_MAPPING = Qwen2Bridge._MLP_MAPPING | {
        "pre_mlp_layernorm.weight": [
            "model.layers.{layer_number}.post_attention_layernorm.weight",
        ],
    }

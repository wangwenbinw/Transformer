import torch
import torch.nn as nn


class LlamaRotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        """
        Args:
            dim: 每个注意力头(Head)的维度 (head_dim)
            max_position_embeddings: 预计算的最大序列长度
            base: 频率基数，默认 10000 (Llama 3 可能会更大)
        """
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base

        # 1. 计算频率 Theta (只计算一半维度，因为是成对旋转)
        # 公式: theta_i = 1 / (base ^ (2i / dim))
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))

        # 将 inv_freq 注册为 buffer (不更新参数，但随模型保存)
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # 2. 预先缓存 cos 和 sin 表
        # 为了避免每次 forward 都重新计算，我们在初始化时就生成好
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings,
            device=device,
            dtype=torch.get_default_dtype()
        )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        # 生成位置索引 t: [0, 1, 2, ..., seq_len-1]
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)

        # 外积计算 m * theta
        # freqs shape: [seq_len, dim // 2]
        freqs = torch.outer(t, self.inv_freq)

        # 关键一步：拼接频率以匹配 rotate_half 的格式
        # LLaMA 的实现是将向量分为前半段和后半段，所以这里也直接在 dim 维度拼接
        # 结果: [theta_0, theta_1, ..., theta_0, theta_1, ...]
        emb = torch.cat((freqs, freqs), dim=-1)

        # 计算 cos 和 sin
        # shape: [seq_len, dim]
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        # x: [batch, seq_len, n_heads, head_dim]
        # 如果推理时的序列长度超过了缓存，需要重新计算缓存 (虽然一般 max_pos 设得很大)
        if seq_len > self.max_position_embeddings:
            self.max_position_embeddings = seq_len
            self._set_cos_sin_cache(seq_len, x.device, x.dtype)

        # 返回切片后的 cos 和 sin
        return (
            self.cos_cached[:seq_len, ...].to(dtype=x.dtype),
            self.sin_cached[:seq_len, ...].to(dtype=x.dtype),
        )


def rotate_half(x):
    """
    RoPE 核心技巧：
    把 x 切成两半 x1, x2
    返回 [-x2, x1]
    对应数学公式中的: (-y, x)
    """
    x1 = x[..., : x.shape[-1] // 2]  # 前半部分
    x2 = x[..., x.shape[-1] // 2:]  # 后半部分
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None):
    """
    应用 RoPE 到 Query 和 Key
    """
    # 1. 调整 cos/sin 形状以便广播 (Broadcasting)
    # q, k shape: [batch, n_heads, seq_len, head_dim] (或者是 [batch, seq_len, n_heads, ...])
    # 假设输入 q 是 [batch, seq_len, n_heads, head_dim]
    # cos, sin 原本是 [seq_len, head_dim]
    # 我们需要变为 [1, seq_len, 1, head_dim] 以便广播

    # 注意：这里假设 q 的第二维是 seq_len。如果 q 是 [batch, heads, seq, dim]，则需要 unsqueeze(0).unsqueeze(0)
    # 这是一个常见的 shape 陷阱，这里按 LLaMA 标准写法 [bs, seq, heads, dim] 处理：
    cos = cos.unsqueeze(0).unsqueeze(2)
    sin = sin.unsqueeze(0).unsqueeze(2)

    # 2. 应用公式
    # q_new = (q * cos) + (rotate_half(q) * sin)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)

    return q_embed, k_embed


# ==========================================
# 测试代码
# ==========================================
if __name__ == "__main__":
    # 配置
    batch_size = 2
    seq_len = 5
    n_heads = 4
    head_dim = 64  # 每个头的维度

    # 1. 初始化 RoPE 模块
    rope = LlamaRotaryEmbedding(dim=head_dim, max_position_embeddings=2048)

    # 2. 构造模拟输入 Query 和 Key
    # Shape: [batch, seq_len, n_heads, head_dim]
    q = torch.randn(batch_size, seq_len, n_heads, head_dim)
    k = torch.randn(batch_size, seq_len, n_heads, head_dim)

    print(f"输入 Q shape: {q.shape}")

    # 3. 获取当前长度的 cos 和 sin
    cos, sin = rope(q, seq_len=seq_len)
    print(f"Cos cache shape: {cos.shape}")  # 应该是 [seq_len, head_dim]

    # 4. 应用旋转
    q_rot, k_rot = apply_rotary_pos_emb(q, k, cos, sin)

    print(f"旋转后 Q shape: {q_rot.shape}")

    # 验证一下数值是否发生了变化
    diff = (q_rot - q).abs().sum()
    print(f"旋转前后差异值 (应 > 0): {diff.item():.4f}")

    # 验证 rotate_half 逻辑
    dummy = torch.tensor([1., 2., 3., 4.])  # dim=4
    # split -> [1,2], [3,4]
    # cat -> [-3,-4, 1,2]
    print(f"Rotate_half 验证 [1,2,3,4] -> {rotate_half(dummy)}")

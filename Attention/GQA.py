import torch
from torch import nn


class GroupedQueryAttention(nn.Module):
    def __init__(self, n_head, n_kv_head, d_model, d_head, d_out):
        super(GroupedQueryAttention, self).__init__()
        self.n_head = n_head
        self.n_kv_head = n_kv_head  # 新增：KV头的数量
        self.d_model = d_model
        self.d_head = d_head
        self.d_out = d_out

        # GQA 检查：Query头数必须能被KV头数整除
        if n_head % n_kv_head != 0:
            raise ValueError(f"n_head ({n_head}) must be divisible by n_kv_head ({n_kv_head})")

        # 计算每个 KV 头对应多少个 Q 头 (Group Size)
        self.n_rep = n_head // n_kv_head

        # Q 的内部维度：n_head * d_head
        self.q_inner_dim = n_head * d_head

        # KV 的内部维度：n_kv_head * d_head (MQA是d_head, MHA是n_head*d_head)
        self.kv_inner_dim = n_kv_head * d_head

        # ===============================
        # 线性映射
        # ===============================
        self.q_linear = nn.Linear(d_model, self.q_inner_dim)
        self.k_linear = nn.Linear(d_model, self.kv_inner_dim)
        self.v_linear = nn.Linear(d_model, self.kv_inner_dim)

        self.out_linear = nn.Linear(self.q_inner_dim, self.d_out)

    def forward(self, hidden_state, attention_mask=None):
        batch_size = hidden_state.size(0)
        seq_len = hidden_state.size(1)

        # ===============================
        # Step 1: 线性映射
        # ===============================
        query = self.q_linear(hidden_state)  # (batch, seq, n_head * d_head)
        key = self.k_linear(hidden_state)  # (batch, seq, n_kv_head * d_head)
        value = self.v_linear(hidden_state)  # (batch, seq, n_kv_head * d_head)

        # ===============================
        # Step 2: 拆分多头 & 调整维度
        # ===============================
        # Query: (batch, n_head, seq, d_head)
        query = query.view(batch_size, seq_len, self.n_head, self.d_head).transpose(1, 2)

        # Key/Value: (batch, n_kv_head, seq, d_head)
        key = key.view(batch_size, seq_len, self.n_kv_head, self.d_head).transpose(1, 2)
        value = value.view(batch_size, seq_len, self.n_kv_head, self.d_head).transpose(1, 2)

        # ===============================
        # Step 3: 重复 KV 头 (Repeat KV) - GQA 核心步骤
        # 为了能和 Query 进行矩阵乘法，需要把 KV 复制 n_rep 次，使其头数变成 n_head
        # (batch, n_kv_head, seq, d_head) -> (batch, n_head, seq, d_head)
        # ===============================
        key = self.repeat_kv(key, self.n_rep)
        value = self.repeat_kv(value, self.n_rep)

        # ===============================
        # Step 4: Scaled Dot-Product Attention
        # 现在 Q 和 K 的形状完全一致了: (batch, n_head, seq, d_head)
        # ===============================
        attention_scores = torch.matmul(
            query,
            key.transpose(-2, -1)
        ) / torch.sqrt(torch.tensor(float(self.d_head), device=query.device))

        # ===============================
        # Step 5: Masking
        # ===============================
        if attention_mask is not None:
            attention_scores = attention_scores.masked_fill(attention_mask == 0, float('-inf'))

        # ===============================
        # Step 6: Softmax & Context
        # ===============================
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        output = torch.matmul(attention_probs, value)  # (batch, n_head, seq, d_head)

        # ===============================
        # Step 7: 拼接多头
        # ===============================
        output = (output
                  .transpose(1, 2)
                  .contiguous()
                  .view(batch_size, -1, self.q_inner_dim))

        # ===============================
        # Step 8: 输出线性映射
        # ===============================
        output = self.out_linear(output)

        return output

    def repeat_kv(self, x, n_rep):
        """
        GQA 关键函数：将 KV 头的数量复制 n_rep 倍
        输入 x: (batch, n_kv_head, seq, d_head)
        输出  : (batch, n_head, seq, d_head)
        """
        batch, n_kv_head, seq_len, d_head = x.shape
        if n_rep == 1:
            return x

        # 1. 增加一个维度: (batch, n_kv_head, 1, seq, d_head)
        # 2. 复制 n_rep 次: (batch, n_kv_head, n_rep, seq, d_head)
        # 3. 展平前两个维度: (batch, n_kv_head * n_rep, seq, d_head) -> 即 (batch, n_head, ...)
        return (
            x.unsqueeze(2)
            .expand(batch, n_kv_head, n_rep, seq_len, d_head)
            .reshape(batch, n_kv_head * n_rep, seq_len, d_head)
        )


# %%
if __name__ == '__main__':
    batch_size = 2
    seq_len = 10
    d_model = 768
    d_out = 768

    # === GQA 配置 ===
    n_head = 12  # Query 头数
    n_kv_head = 4  # KV 头数 (分组数), 12/4 = 3, 即每 3 个 Query 共享 1 个 KV
    d_head = 64  # = 768 / 12

    gqa = GroupedQueryAttention(
        n_head=n_head,
        n_kv_head=n_kv_head,
        d_model=d_model,
        d_head=d_head,
        d_out=d_out
    )

    hidden_state = torch.randn(batch_size, seq_len, d_model)

    # causal mask
    causal_mask = torch.tril(torch.ones(seq_len, seq_len))
    causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)

    output = gqa(hidden_state, causal_mask)

    print(f"\n[GQA] 最终 output 形状: {output.shape}")

    # === 参数量对比 ===
    mha_params = (d_model * n_head * d_head) * 3
    # GQA: K, V 的参数量只有 MHA 的 (n_kv_head / n_head)
    gqa_params = (d_model * n_head * d_head) + (d_model * n_kv_head * d_head) * 2

    print(f"[MHA] 投影层参数量估计: {mha_params}")
    print(f"[GQA] 投影层参数量估计: {gqa_params} (介于 MHA 和 MQA 之间)")
import torch
from torch import nn

class MultiQueryAttention(nn.Module):
    def __init__(self,n_head,d_model,d_head,d_out):
        super(MultiQueryAttention, self).__init__()
        self.n_head = n_head
        self.d_model = d_model
        self.d_head = d_head
        self.d_out = d_out

        # Q的内部维度：依然是多头
        self.q_inner_dim = n_head * d_head
        #KV内部的维度，只有1个头，所以等于d_head
        self.kv_inner_dim = d_head

        # ===============================
        # Q / K / V 线性映射
        # Q 输入输出 : (batch, seq, d_model) -> (batch, seq, n_head * d_head)
        # K,V 输入输出: (batch, seq, d_model) -> (batch, seq, d_head) <-- 变小
        self.q_linear = nn.Linear(d_model,self.q_inner_dim)
        self.k_linear = nn.Linear(d_model,self.kv_inner_dim)
        self.v_linear = nn.Linear(d_model,self.kv_inner_dim)

        #输出投影层
        self.out_linear = nn.Linear(self.q_inner_dim,self.d_out)


    def forward(self,hidden_state, attention_mask=None):
        """
           hidden_state:
               (batch_size, seq_len, d_model)
               例子: [2, 10, 768]

           attention_mask:
               (batch, 1, seq_len, seq_len) 或可 broadcast
               causal attention 中用于屏蔽未来 token
        """
        batch_size = hidden_state.size(0)
        seq_len = hidden_state.size(1)

        # ===============================
        # Step 1: 线性映射
        # ===============================
        query = self.q_linear(hidden_state)  # (batch, seq, n_head * d_head)
        key = self.k_linear(hidden_state)  # (batch, seq, d_head)
        value = self.v_linear(hidden_state)  # (batch, seq, d_head)

        # ===============================
        # Step 2: 拆分多头,自动调整了维度
        # Query: (batch, n_head, seq, d_head)
        # Key/Value: (batch, 1, seq, d_head)
        # ===============================
        query = self.split_heads(query)
        # 处理 Key/Value (MQA 特有：增加一个维度用于广播)
        # 变成 (batch, 1, seq, d_head)，这里的 '1' 让它能被 n_head 个 Q 共享
        key = key.view(batch_size, seq_len, 1, self.d_head).transpose(1, 2)
        value = value.view(batch_size, seq_len, 1, self.d_head).transpose(1, 2)

        # ===============================
        # Step 3: Scaled Dot-Product Attention
        # Q: (batch, n_head, seq, d_head)
        # K.T: (batch, 1, d_head, seq)
        # 结果 -> (batch, n_head, seq, seq)
        # ===============================
        attention_scores = torch.matmul(
            query,
            key.transpose(-2, -1)
        ) / torch.sqrt(torch.tensor(float(self.d_head), device=query.device))

        # ===============================
        # Step 4: Masking
        # Mask 形状通常是 (batch, 1, seq, seq)，自动广播到 (batch, n_head, ...)
        # ===============================
        if attention_mask is not None:
            attention_scores = attention_scores.masked_fill(attention_mask == 0, float('-inf'))

        # ===============================
        # Step 5: Softmax
        # ===============================
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # ===============================
        # Step 6: 加权求和 Value
        #
        # Probs: (batch, n_head, seq, seq)
        # V:     (batch, 1, seq, d_head)
        # 结果 -> (batch, n_head, seq, d_head)
        # ===============================
        output = torch.matmul(attention_probs, value)

        # ===============================
        # Step 7: 拼接多头 (Concat Heads)
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


    def split_heads(self, x):
        """
            将 inner_dim 拆分为 n_head × d_head
        """
        batch_size = x.size(0)
        return (
            x.view(batch_size, -1, self.n_head, self.d_head)
            .transpose(1, 2)
        )


# %%
if __name__ == '__main__':
    batch_size = 2
    seq_len = 10
    n_head = 12
    d_model = 768
    d_head = 64  # = 768 / 12
    d_out = 768

    # 使用 MQA
    mqa = MultiQueryAttention(
        n_head=n_head,
        d_model=d_model,
        d_head=d_head,
        d_out=d_out
    )

    hidden_state = torch.randn(batch_size, seq_len, d_model)

    # causal mask
    causal_mask = torch.tril(torch.ones(seq_len, seq_len))
    causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)

    output = mqa(hidden_state, causal_mask)

    print(f"\n[MQA] 最终 output 形状: {output.shape}")

    # 验证参数量差异
    mha_params = (d_model * n_head * d_head) * 3  # Q, K, V
    mqa_params = (d_model * n_head * d_head) + (d_model * d_head) * 2  # Q + K(1) + V(1)
    print(f"[MHA] 投影层参数量估计: {mha_params}")
    print(f"[MQA] 投影层参数量估计: {mqa_params} (显著减少)")
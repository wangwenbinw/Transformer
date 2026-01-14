import torch
from torch import nn

class MultiHeadAttention(nn.Module):
    def __init__(self, n_head, d_model, d_head, d_out):
        super(MultiHeadAttention, self).__init__()
        # ===============================
        # 参数说明
        # n_head : 注意力头数
        # d_model: 输入 token 表示维度
        # d_head : 每个 head 的维度

        # d_out  : attention 输出维度
        # ===============================
        self.n_head = n_head
        self.d_model = d_model
        self.d_head = d_head
        self.d_out = d_out



        #多头拼接后维度
        #在经典MHA中，inner_dim == d_model
        self.inner_dim = n_head * d_head

        # ===============================
        # Q / K / V 线性映射
        #
        # 输入 : (batch, seq_len, d_model)
        # 输出 : (batch, seq_len, inner_dim)
        # ===============================
        self.q_linear = nn.Linear(d_model,self.inner_dim)
        self.k_linear = nn.Linear(d_model,self.inner_dim)
        self.v_linear = nn.Linear(d_model,self.inner_dim)

        #输出投影层
        # (batch, seq_len, inner_dim) -> (batch, seq_len, d_out)
        self.out_linear = nn.Linear(self.inner_dim,self.d_out)


    def forward(self,hidden_state,attention_mask=None):
        """
           hidden_state:
               (batch_size, seq_len, d_model)
               例子: [2, 10, 768]

           attention_mask:
               (batch, 1, seq_len, seq_len) 或可 broadcast
               causal attention 中用于屏蔽未来 token
        """
        batch_size = hidden_state.size(0)

        # ===============================
        # Step 1: 线性映射得到 Q, K, V
        #
        # (batch, seq_len, d_model)
        # -> (batch, seq_len, inner_dim)
        # ===============================
        query = self.q_linear(hidden_state)
        key = self.k_linear(hidden_state)
        value = self.v_linear(hidden_state)

        # ===============================
        # Step 2: 拆分多头,自动调整了维度
        #
        # (batch, seq_len, inner_dim)
        # -> (batch, n_head, seq_len, d_head)
        # ===============================
        query = self.split_heads(query)
        key = self.split_heads(key)
        value = self.split_heads(value)

        # ===============================
        # Step 3: Scaled Dot-Product Attention
        # scores = Q K^T / sqrt(d_head)  size:(batch, n_head, seq_len, seq_len)
        # ===============================
        attention_scores = torch.matmul(
            query,
            key.transpose(-2, -1)
        ) / torch.sqrt(torch.tensor(float(self.d_head), device=query.device))

        # ===============================
        # Step 4: Causal / Padding Mask
        #若不施加 causal mask → 行为等价于 Encoder self-attention
        #若施加下三角（causal）mask → 行为等价于 Decoder self-attentio
        # attention_mask可以广播和attention_scores一样的形状，然后
        # attention_mask为0的位置对应的attention_scores值为-inf,从而
        #求softmax后这个位置的值为0
        # ===============================
        if attention_mask is not None:
            attention_scores = attention_scores.masked_fill(attention_mask == 0, float('-inf'))

        # ===============================
        # Step 5: softmax 归一化
        # ===============================
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # ===============================
        # Step 6: 加权求和 Value
        #    (batch, n_head, seq_len, seq_len)
        # -> (batch, n_head, seq_len, d_head)
        # ===============================
        output = torch.matmul(attention_probs, value)

        # ===============================
        # Step 7: 拼接多头
        #
        # (batch, n_head, seq_len, d_head)
        # -> (batch, seq_len, inner_dim)
        # ===============================
        output = (output
                  .transpose(1,2)
                  .contiguous()
                  .view(batch_size,-1,self.inner_dim))

        # ===============================
        # Step 8: 输出线性映射
        #
        # -> (batch, seq_len, d_out)
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


#%%
if __name__ == '__main__':
    batch_size = 2
    seq_len = 10

    n_head = 12
    d_model = 768
    d_head = 64        # = 768 / 12


    d_out = 768        # 输出维度

    attn = MultiHeadAttention(
        n_head=n_head,
        d_model=d_model,
        d_head=d_head,
        d_out=d_out
    )

    hidden_state = torch.randn(batch_size, seq_len, d_model)

    # causal mask
    causal_mask = torch.tril(torch.ones(seq_len, seq_len))
    causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)

    output = attn(hidden_state, causal_mask)

    print(f"\n 最终 output 形状: {output.shape}")




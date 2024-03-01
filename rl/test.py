import torch
import torch.nn.functional as F
import torch.nn as nn

class MultiheadSelfAttention(nn.Module):
    def __init__(self, feature_dim=128, num_heads=8, dropout=0.1, attention_dropout=0.1):
        super(MultiheadSelfAttention, self).__init__()
        self.feature_dim = feature_dim
        self.num_heads = num_heads
        self.attn_head_size = feature_dim // num_heads
        self.scales = self.attn_head_size ** -0.5

        # 为每个注意力头定义线性映射
        self.linear_query = nn.Linear(feature_dim, feature_dim)
        nn.init.xavier_normal_(self.linear_query.weight)
        nn.init.constant_(self.linear_query.bias, 0.0)
        self.linear_key = nn.Linear(feature_dim, feature_dim)
        nn.init.xavier_normal_(self.linear_key.weight)
        nn.init.constant_(self.linear_key.bias, 0.0)
        self.linear_value = nn.Linear(feature_dim, feature_dim)
        nn.init.xavier_normal_(self.linear_value.weight)
        nn.init.constant_(self.linear_value.bias, 0.0)

        self.linear_projection = nn.Linear(feature_dim * num_heads, feature_dim)
        nn.init.xavier_normal_(self.linear_projection.weight)
        nn.init.constant_(self.linear_projection.bias, 0.0)

        self.attn_dropout = nn.Dropout(attention_dropout)
        self.proj_dropout = nn.Dropout(dropout)

    def forward(self, x):
        batch_size, seq_len, _ = x.size()

        # 线性映射得到查询、键和值
        q = self.linear_query(x).view(batch_size, seq_len, self.num_heads, -1)
        k = self.linear_key(x).view(batch_size, seq_len, self.num_heads, -1)
        v = self.linear_value(x).view(batch_size, seq_len, self.num_heads, -1)

        # 计算注意力分数
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scales
        attention_weights = F.softmax(scores, dim=-1)

        attention_weights = self.attn_dropout(attention_weights)

        # 加权求和得到每个头的输出
        weighted_sum = torch.matmul(attention_weights, v)

        # 将多头的输出进行连接并重塑形状
        concatenated_representation = weighted_sum.view(batch_size, seq_len, -1)

        # 对连接后的表示进行投影
        projected_representation = self.linear_projection(concatenated_representation.view(-1, self.feature_dim * self.num_heads))
        # new_shape = list(concatenated_representation.shape[:-2]) + [self.feature_dim]    # [nenv, seq_len, all_head_size]
        # projected_representation = projected_representation.reshape(new_shape)
        aggregated_representation = self.proj_dropout(projected_representation)

        return aggregated_representation

# 测试模型
batch_size, seq_len, feature_dim = 32, 10, 64
x = torch.rand((batch_size, seq_len, feature_dim))

# 创建多头注意力模型
multihead_attention = MultiheadSelfAttention(feature_dim=feature_dim, num_heads=8)

# 获取输出
output = multihead_attention(x)

print("Input shape:", x.shape)
print("Output shape:", output.shape)

import torch
from torch import nn

from einops import rearrange
from einops.layers.torch import Rearrange

# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

def posemb_sincos_2d(h, w, dim, temperature: int = 10000, dtype = torch.float32):
    y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    assert (dim % 4) == 0, "feature dimension must be multiple of 4 for sincos emb"
    omega = torch.arange(dim // 4) / (dim // 4 - 1)
    omega = 1.0 / (temperature ** omega)

    y = y.flatten()[:, None] * omega[None, :]
    x = x.flatten()[:, None] * omega[None, :]
    pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim=1)
    return pe.type(dtype)

# classes

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )
    def forward(self, x):
        return self.net(x)

class SharedMLP(nn.Module):
    def __init__(self, dim, mlp_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.ReLU(),
            nn.Linear(mlp_dim, dim),
            nn.ReLU()
        )

    def forward(self, x):
        return self.net(x)

class SelfAttention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64):
        super().__init__()
        inner_dim = dim_head *  heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim = -1)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Linear(inner_dim, dim, bias = False)

    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class CrossAttention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.norm_x = nn.LayerNorm(dim)
        self.norm_y = nn.LayerNorm(1)

        self.attend = nn.Softmax(dim=-1)

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_k = nn.Linear(1, inner_dim, bias=False)
        self.to_v = nn.Linear(1, inner_dim, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(self, x, y):
        x = self.norm_x(x)
        #print("After norm_x:",x.shape)
        y = self.norm_y(y)
        #print("After norm_y:",y.shape)

        q = rearrange(self.to_q(x), "b n (h d) -> b h n d", h=self.heads)
        k = rearrange(self.to_k(y), "b n (h d) -> b h n d", h=self.heads)
        v = rearrange(self.to_v(y), "b n (h d) -> b h n d", h=self.heads)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        out = torch.matmul(attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                SelfAttention(dim, heads = heads, dim_head = dim_head),
                FeedForward(dim, mlp_dim)
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)

class OcclusionQueries(nn.Module):
    def __init__(self, *, image_size, patch_size, dim, depth, heads, mlp_dim, channels = 3, dim_head = 64):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        patch_dim = channels * patch_height * patch_width
        #print("patch_dim:",patch_dim)
        self.to_patch_embedding = nn.Sequential(
            Rearrange("b c (h p1) (w p2) -> b (h w) (p1 p2 c)", p1 = patch_height, p2 = patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.pos_embedding = posemb_sincos_2d(
            h = image_height // patch_height,
            w = image_width // patch_width,
            dim = dim,
        ) 

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim)

        self.cross_attention = CrossAttention(dim, heads=heads, dim_head=dim_head)
        self.self_attention = SelfAttention(dim, heads=heads, dim_head=dim_head)
        self.shared_mlp = SharedMLP(dim, mlp_dim)
        # self.pool = "mean"
        # self.to_latent = nn.Identity()

        # self.linear_head = nn.Linear(dim, num_classes)

    def forward(self, img,y):
        # device = img.device
        # #print("**********************")
        # #print("image:", img.shape)
        # x_patch = self.to_patch_embedding(img)
        # #print("after_patch:",x_patch.shape)
        # x_pos = x_patch + self.pos_embedding.to(device, dtype=x_patch.dtype)
        # #print("after_position:",x_pos.shape)
        # x = self.transformer(x_pos)
        # #print("after_transformer:",x.shape)
        # # 生成维度为[12, 1, 256]的随机向量
        # # y = torch.randn(12, 1, 256)
        y = y.permute(0, 2, 1)  # 使用transpose操作进行维度调整
        x = img  # 使用transpose操作进行维度调整
        #print("x and y:",x.shape,y.shape)
        x= self.cross_attention(x, y)
        #print("after_CA:",x.shape)
        x= self.self_attention(x)
        #print("after_SA:",x.shape)
        x = x.view(x.size(0), -1, x.size(-1))
        #print("after_reshape:",x.shape)
        shared_mlp_out = self.shared_mlp(x)
        #print("after_MLP:",shared_mlp_out.shape)
        # reshaped_x = shared_mlp_out.reshape(12, 1, 100, 100)
        # print(reshaped_x.shape)
        # predicted_M = torch.cat([img, reshaped_x], dim=1)
        # print("OGM:",predicted_M.shape)
        # # x = x.mean(dim = 1)
        # # print("after_mean:",x.shape)
        # # x = self.to_latent(x)
        # # print("after_latent:",x.shape)
        # print("**********************")
        # # return self.linear_head(x)
        # # return x
        # return predicted_M

        return shared_mlp_out

# def main():
#     # 创建一个测试输入
#     batch_size = 12
#     channels = 1
#     image_height = 100
#     image_width = 100
#     test_input = torch.randn(batch_size, channels, image_height, image_width)

#     y = torch.randn(12, 1, 128)

#     # 创建一个SimpleViT模型
#     model = OcclusionQueries(
#         image_size=(image_height, image_width),
#         patch_size=(10, 10),
#         dim=200,
#         depth=12,
#         heads=8,
#         mlp_dim=2048,
#         channels=channels,
#         dim_head=64
#     )

#     # 将输入传递给模型
#     output = model(test_input,y)
#     # 打印输出的形状
#     print("test_input:", test_input.shape)
#     print("输出形状: ", output.shape)

#     # 检查模型的连通性
#     num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
#     print("可训练参数的数量: ", num_params)

# if __name__ == '__main__':
#     main()


def main():
    # 创建一个测试输入
    batch_size = 12
    channels = 1
    image_height = 100
    image_width = 100
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_input = torch.randn(batch_size, channels, image_height, image_width).to(device)  # Move input to device

    y = torch.randn(12, 1, 128).to(device)  # Move y to device

    # 创建一个SimpleViT模型
    model = OcclusionQueries(
        image_size=(image_height, image_width),
        patch_size=(10, 10),
        dim=128,
        depth=12,
        heads=8,
        mlp_dim=2048,
        channels=channels,
        dim_head=64
    ).to(device)  # Move model to device

    # 将输入传递给模型
    # print("$$$$$$$$$$$$",test_input.shape,y.shape)
    output = model(test_input, y)
    # 打印输出的形状
    # print("test_input:", test_input.shape)
    # print("输出形状: ", output.shape)

    # 检查模型的连通性
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("可训练参数的数量: ", num_params)

if __name__ == '__main__':
    main()


from functools import partial
import torchvision.models as models
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import DropPath, trunc_normal_
from timm.models.vision_transformer import Mlp
from torchinfo import summary
from efficientnet_pytorch import EfficientNet
import timm
from functools import partial
import torchvision.models as models
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import DropPath, trunc_normal_
from models.Swin.GCN import TransformerDecoderLayer,TransformerDecoder
from models.Swin.MOE import SparseMoE



from timm.models.vision_transformer import Mlp
from torchinfo import summary

class UPP_Layer(torch.nn.Module):
    def __init__(self, pool_size=7):
        super().__init__()
        self.pool_size = pool_size

    def forward(self, x):
        B, C, H, W = x.shape
        h_bins = torch.linspace(0, H, self.pool_size + 1, dtype=torch.int)
        w_bins = torch.linspace(0, W, self.pool_size + 1, dtype=torch.int)
        out = []
        for i in range(self.pool_size):
            for j in range(self.pool_size):
                h_start, h_end = h_bins[i], h_bins[i+1]
                w_start, w_end = w_bins[j], w_bins[j+1]
                patch = x[:, :, h_start:h_end, w_start:w_end]    # [B,C,h',w']
                patch_pool = patch.mean(dim=(-1,-2))             # [B,C]
                out.append(patch_pool)
        out = torch.stack(out, dim=1)   # [B, pool_size*pool_size, C]
        return out

class MLP(nn.Module):
    def __init__(self, n_embd=384,dropout = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = self.net(x)
        aux_loss = 0
        z_loss = 0
        return x,aux_loss,z_loss

class Life_models(nn.Module):

    def __init__(
        self,
        embed_dim=384,
        juery_nums=6,
        num_experts = 8,
        top_k = 2,
        depth=4,


    ):
        super().__init__()
        self.pretrain_model = timm.create_model('swin_base_patch4_window7_224.ms_in22k_ft_in1k', pretrained=True)

        self.backbone_stages = nn.ModuleList([
            nn.Sequential(
                self.pretrain_model.patch_embed,
                self.pretrain_model.layers[0]
            ),
            self.pretrain_model.layers[1],
            self.pretrain_model.layers[2],
            self.pretrain_model.layers[3]
        ])

        del self.pretrain_model.norm
        del self.pretrain_model.head

        self.UPP_layer3 = UPP_Layer(pool_size=7)

        self.proj_layer3 = nn.Linear(512, embed_dim)
        self.proj_layer4 = nn.Conv2d(1024, embed_dim, kernel_size=1)


        self.juery = juery_nums  

        bunch_layer = TransformerDecoderLayer(  
            d_model=embed_dim,
            dropout=0.0,
            nhead=6,
            activation=F.gelu,
            batch_first=True,
            dim_feedforward=(embed_dim * 4),
            norm_first=True,
        )
        self.bunch_decoder = TransformerDecoder(bunch_layer, num_layers=depth) 
        self.bunch_embedding = nn.Parameter(
            torch.randn(1, self.juery, embed_dim))  

        trunc_normal_(self.bunch_embedding, std=0.02)

        if num_experts == 1:
            self.MOE = MLP(embed_dim)
        else:
            self.MOE = SparseMoE(n_embed=embed_dim,num_experts=num_experts,top_k=top_k)
        self.norm3 = nn.LayerNorm(embed_dim, eps=1e-5, bias=True)
        self.dropout3 = nn.Dropout(0.1)
        self.gamma = nn.Parameter(torch.tensor(0.1))

        self.fc = nn.Linear(embed_dim, 1)



    def forward(self, x):
        features = []
        B = x.shape[0]
        bunch_embedding = self.bunch_embedding.expand(B, -1, -1) 
        for backbone_stage in self.backbone_stages:
            x = backbone_stage(x)
            feature = x.permute(0, 3, 1, 2).contiguous()
            features.append(feature)

        layer4 = self.proj_layer4(features[3])
        layer4 = layer4.flatten(2).permute(0, 2, 1)
        global_vec = layer4.mean(dim=1, keepdim=True)
        output_embedding = bunch_embedding + global_vec

        layer3 = self.UPP_layer3(features[2])
        layer3 = self.proj_layer3(layer3)




        output = self.bunch_decoder(output_embedding, layer3)
        output,aux_loss,z_loss = self._ff_block(self.norm3(output))
        final_score = self.fc(output)
        final_score = final_score.view(B, -1).mean(dim=1, keepdim=True)
        return final_score,aux_loss,z_loss

    def _ff_block(self, x):
        moe_out,aux_loss,z_loss = self.MOE(x)
        x = x + self.gamma * moe_out
        return self.dropout3(x),aux_loss,z_loss


def build_life(
    embed_dim=384,
    juery_nums = 6,
    num_experts = 8,
    top_k = 2,
    depth = 4,

):
    model = Life_models(
        embed_dim=embed_dim,
        juery_nums= juery_nums,
        num_experts= num_experts,
        top_k= top_k,
        depth=depth
    )

    return model


if __name__ == "__main__":
    model = build_life(
        embed_dim=384,
        juery_nums=6,
        num_experts=4,
        top_k=2,
    ).cuda()
    print("===== Model Parameters with Index =====")
    for idx, (name, param) in enumerate(model.named_parameters()):
        print(f"{idx}: {name}, requires_grad={param.requires_grad}, shape={param.shape}")

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {total_params / 1e6:.2f} M")
    own_params = sum(
        p.numel() for name,p in model.named_parameters()
        if p.requires_grad and not name.startswith("pretrain_model")
        )
    print(f"My custom module params: {own_params / 1e6:.2f} M")



    input1 = torch.randn(1, 3, 224, 224).cuda()
    output,_,_ = model(input1)
    print(output.shape)

    # input1 = torch.randn(1, 3, 224, 224)
    summary(model, input_data=[input1], device=torch.device("cuda"))


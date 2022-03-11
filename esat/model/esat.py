import torch
from torch import nn
from einops import  repeat
from einops.layers.torch import Rearrange
from torchvision.models import resnet34


class cv(nn.Module):
    def __init__(self, *, image_size,  patch_size,patch_size_big, num_classes, dim=512, transformer,batch_size):
        super().__init__()

        patch_size2=int(image_size/patch_size)

        self.model = resnet34(pretrained=True)
        modules = list(self.model.children())[:-1]
        self.model = torch.nn.Sequential(*modules)
        self.model.eval()

        self.to_patch_embedding1 = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> (b h w) c p1 p2', p1 = patch_size, p2 = patch_size),
        )
        self.to_patch_embedding2 = nn.Sequential(
            Rearrange('(b h w) d -> b d h w',b=batch_size, h=patch_size2),
        )
        self.conv1 = nn.Conv2d(dim, 128, kernel_size=6,stride=5,padding=1)
        self.conv2 = nn.Conv2d(128, 64, kernel_size=5,stride=5,padding=2)
        self.transformer = transformer
        self.to_patch_embedding3 = nn.Sequential(
            Rearrange('b d h w-> b (h w) d'),
        )

        self.to_patch_embedding4 = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> (b h w) c p1 p2', p1 = patch_size_big, p2 = patch_size_big),
        )
        self.to_patch_embedding5 = nn.Sequential(
            Rearrange('(b c) d -> b c d',b=batch_size),
        )
        self.conv3 =  nn.Linear(dim, 64)


        self.pos_embedding = nn.Parameter(torch.randn(1, patch_size*patch_size + 1, 128))
        self.cls_token = nn.Parameter(torch.randn(1, 1, 128))
        self.transformer = transformer
        self.to_latent = nn.Identity()
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(128),
            nn.Linear(128, num_classes)
        )

    def forward(self, img):
        #small
        x = self.to_patch_embedding1(img)
        x = self.model(x)
        x = torch.squeeze(x)
        x = self.to_patch_embedding2(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.to_patch_embedding3(x)
        # big
        x_b = self.to_patch_embedding4(img)
        x_b = self.model(x_b)
        x_b = torch.squeeze(x_b)
        x_b = self.to_patch_embedding5(x_b)
        x_b = self.conv3(x_b)
        x=torch.cat((x, x_b), dim=2)
        b, n, _ = x.shape
        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.transformer(x)
        x = x[:, 0]
        feature = x.squeeze(0)
        x = self.to_latent(x)
        x = self.mlp_head(x)
        return x,feature






# from nystrom_attention import Nystromformer
# efficient_transformer = Nystromformer(
#     dim = 64,
#     depth = 12,
#     heads = 8,
#     num_landmarks = 256
# )
# model = cv(
#     image_size = 6144,
#     patch_size = 16,
#     patch_size_big = 384,
#     num_classes = 1,
#     transformer = efficient_transformer,
#     batch_size=1
# )
# a=torch.rand((1,3,6144,6144))
# model(a)
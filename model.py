import torch

class ConvBlock(torch.nn.Module):

    def __init__(self, in_channels : int, out_channels : int, stride: int = 1 ) -> None:
        super().__init__()
        self.Conv_0 = torch.nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=stride, padding=1, bias=True)
        self.Norm_0 = torch.nn.InstanceNorm3d(num_features=out_channels, affine=True)
        self.Activation_0 = torch.nn.LeakyReLU(negative_slope=0.01, inplace=True)
        self.Conv_1 = torch.nn.Conv3d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1, bias=True)
        self.Norm_1 = torch.nn.InstanceNorm3d(num_features=out_channels, affine=True)
        self.Activation_1 = torch.nn.LeakyReLU(negative_slope=0.01, inplace=True)
        
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        output = self.Conv_0(input)
        output = self.Norm_0(output)
        output = self.Activation_0(output)
        output = self.Conv_1(output)
        output = self.Norm_1(output)
        output = self.Activation_1(output)
        return output

class UNetHead(torch.nn.Module):
    def __init__(self, in_channels: int, nb_class: int) -> None:
        super().__init__()
        self.Conv = torch.nn.Conv3d(in_channels = in_channels, out_channels = nb_class, kernel_size = 1, stride = 1, padding = 0)
        self.Softmax = torch.nn.Softmax(dim=1)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        output = self.Conv(input)
        output = self.Softmax(output)
        return output
    
class UNetBlock(torch.nn.Module):
    def __init__(self, channels, nb_class: int, i : int = 0) -> None:
        super().__init__()
        self.DownConvBlock = ConvBlock(in_channels=channels[0], out_channels=channels[1], stride=2 if i > 0 else 1)
        if len(channels) > 2:
            self.UNetBlock = UNetBlock(channels[1:], nb_class, i+1)
            self.UpConvBlock = ConvBlock(in_channels=channels[1]*2, out_channels=channels[1])
            self.Head = UNetHead(channels[1], nb_class)
        if i > 0:
            self.CONV_TRANSPOSE = torch.nn.ConvTranspose3d(in_channels = channels[1], out_channels = channels[0], kernel_size = 2, stride = 2, padding = 0)
        
    def forward(self, input: torch.Tensor, i: int = 0) -> tuple[torch.Tensor, list[torch.Tensor]]:
        layers = []
        output = input
        output = self.DownConvBlock(output)

        if hasattr(self, "UNetBlock"):
            output, ls = self.UNetBlock(output, i+1)
            for l in ls:
                layers.append(l)
            output = self.UpConvBlock(output)
            layers.append(self.Head(output))

        if i > 0:
            output = self.CONV_TRANSPOSE(output)
            output = torch.cat((output, input), dim=1)
        return output, layers

def perturb_instance_norm(module: torch.nn.Module, mean: float = 0, std: float = 1):
    """ Apply perturbation to the affine parameters of InstanceNorm3d layers """
    for _, sub_module in module.named_modules():
        if isinstance(sub_module, torch.nn.InstanceNorm3d) and sub_module.affine:
            with torch.no_grad():
                epsilon = torch.randn_like(sub_module.weight) * std + mean
                sub_module.weight.data = (1 + epsilon) * sub_module.weight.data

class Unet_TotalSeg(torch.nn.Module):

    def __init__(self,  channels=[1, 64, 128, 256, 512, 1024], nb_class: int = 2, pertubation: bool = False, perturb_mean: float = 0, perturb_std: float = 1) -> None:
        super().__init__()
        self.UNetBlock = UNetBlock(channels, nb_class=nb_class)
        if pertubation:
            perturb_instance_norm(self, mean=perturb_mean, std=perturb_std)

    def forward(self, input: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        return self.UNetBlock(input)
    
class BayesianUnetTotalSeg(Unet_TotalSeg):
    def __init__(self, channels: list[int], nb_class: int, means, stds):
        super().__init__(channels, nb_class)
        self.means = means
        self.stds = stds
    
    def resample_bayesian_weights(self):
        i = 0
        for _, module in self.named_modules():
            if isinstance(module, torch.nn.InstanceNorm3d) and module.affine:
                weight_mean, bias_mean = self.means[i]
                weight_std, bias_std = self.stds[i]
                module.weight.data = weight_mean.to(0) + weight_std.to(0) * torch.randn_like(weight_mean).to(0)
                module.bias.data = bias_mean.to(0) + bias_std.to(0) * torch.randn_like(bias_mean).to(0)
                i += 1
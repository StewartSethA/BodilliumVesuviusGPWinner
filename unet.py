from torch import nn
import torch.nn.functional as F
import torch

class UNet(nn.Module):
    def __init__(self, complexity=64):
        super(UNet, self).__init__()
        c = complexity

        # Z Convolution (Vertical Convolution along the depth dimension)
        self.conv_z = nn.Conv3d(in_channels=1, out_channels=c, kernel_size=(3, 1, 1), padding=(1, 0, 0))

        # Contracting path
        self.enc_conv1 = self.double_conv(c, c)
        self.enc_conv2 = self.double_conv(c, c*2)
        self.enc_conv3 = self.double_conv(c*2, c*4)
        self.enc_conv4 = self.double_conv(c*4, c*8)
        self.enc_conv5 = self.double_conv(c*8, c*16)

        # Expansive path
        self.up_trans1 = self.up_conv(c*16, c*8)
        self.dec_conv1 = self.double_conv(c*16, c*8)
        self.up_trans2 = self.up_conv(c*8, c*4)
        self.dec_conv2 = self.double_conv(c*8, c*4)
        self.up_trans3 = self.up_conv(c*4, c*2)
        self.dec_conv3 = self.double_conv(c*4, c*2)
        self.up_trans4 = self.up_conv(c*2, c)
        self.dec_conv4 = self.double_conv(c*2, c)

        # Final output
        self.out_conv = nn.Conv2d(c, 1, kernel_size=1)

    def forward(self, x):
        # Apply the Z convolution
        #print("unet input shape", x.shape)

        x = self.conv_z(x)  # [batch_size, 64, depth, height, width]
        #print("conv_z shape", x.shape)

        # Aggregate the depth dimension using max pooling or average pooling
        x, _ = torch.max(x, dim=2)  # Choose max or torch.mean(x, dim=2) for average pooling
        #print("x z agg shape", x.shape)

        # Encoder
        '''
        unet input shape torch.Size([256, 1, 30, 16, 16])                                                                  conv_z shape torch.Size([256, 64, 30, 16, 16])                                                                     
        x z agg shape torch.Size([256, 64, 16, 16])                                                                        
        x1 torch.Size([256, 64, 16, 16])                                                                                   
        x2 torch.Size([256, 128, 8, 8])                                                                                    x3 torch.Size([256, 256, 4, 4])                                                                                    
        x4 torch.Size([256, 512, 2, 2])                                                                                    
        x5 torch.Size([256, 1024, 1, 1])
        '''

        x1,x2,x3,x4,x5 = (None,)*5
        x1 = self.enc_conv1(x)
        x = x1
        #print("x1", x1.shape)
        if x1.shape[-1] >= 2:
          x2 = self.enc_conv2(F.max_pool2d(x1, kernel_size=2)) # 1/2
          x = x2
          #print("x2", x2.shape)
          if x2.shape[-1] >= 2:
            x3 = self.enc_conv3(F.max_pool2d(x2, kernel_size=2)) # 1/4
            x = x3
            #print("x3", x3.shape)
            if x3.shape[-1] >= 2:
              x4 = self.enc_conv4(F.max_pool2d(x3, kernel_size=2)) # 1/8
              x = x4
              #print("x4", x4.shape)
              if x4.shape[-1] >= 2:
                x5 = self.enc_conv5(F.max_pool2d(x4, kernel_size=2)) # 1/16
                x = x5
                #print("x5", x5.shape)

        # Decoder
        if x5 is not None:
          x = self.up_trans1(x5)
        if x4 is not None:
          x = torch.cat([x, x4], dim=1)
          x = self.dec_conv1(x)
        if x3 is not None:
          x = self.up_trans2(x)
          x = torch.cat([x, x3], dim=1)
          x = self.dec_conv2(x)

        if x2 is not None:
          x = self.up_trans3(x)
          x = torch.cat([x, x2], dim=1)
          x = self.dec_conv3(x)

        x = self.up_trans4(x)
        x = torch.cat([x, x1], dim=1)
        x = self.dec_conv4(x)

        x = self.out_conv(x)

        return x

    def double_conv(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def up_conv(self, in_channels, out_channels):
        return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)


# This function initializes the weights in an intelligent way
def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv3d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.ConvTranspose2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm3d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

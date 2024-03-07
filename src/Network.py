import torch

class ConvBlock(torch.nn.Module):

    def __init__(self, in_dim, out_dim, filter_shape):
        """
        :param in_dim: Block input dimension.
        :param out_dim: Block output dimension.
        :param filter_shape: Shape of the filter.
        """
        super(ConvBlock, self).__init__()
        self.block = torch.nn.Sequential(
            torch.nn.Conv2d(in_dim, out_dim, filter_shape, padding='same'),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(out_dim))

    def forward(self, x):
        return self.block(x)


class ResNetBlock(torch.nn.Module):

    def __init__(self, in_dim, out_dim, filter_shape, dropout_rate=0.1,
                 use_attention=False):
        """
        :param in_dim: Input dimension.
        :param out_dim: Output dimension.
        :param filter_shape: Filter shape.
        """
        super(ResNetBlock, self).__init__()
        self.conv_1 = ConvBlock(in_dim, out_dim, filter_shape)
        self.conv_2 = ConvBlock(out_dim, out_dim, filter_shape)
        self.dropout = torch.nn.Dropout2d(dropout_rate)

        if use_attention:
            self.attention = AttentionBlock(out_dim)
        else:
            self.attention = torch.nn.Identity()

        if in_dim != out_dim:
            self.conv_res = torch.nn.Conv2d(in_dim, out_dim, 1,
                                            padding='same')
        else:
            self.conv_res = torch.nn.Identity()

    def forward(self, x):
        y = self.conv_1(x)
        y = self.dropout(y)
        y = self.conv_2(y)
        y = self.attention(y)
        return y + self.conv_res(x)


class AttentionBlock(torch.nn.Module):

    def __init__(self, embed_dim, heads=8, dropout=0.1):
        """
        :param embed_dim: Embedding dimension.
        :param heads: Number of attention heads.
        :param dropout: Dropout probability.
        """
        super(AttentionBlock, self).__init__()
        self.embed_dim = embed_dim
        self.mha = torch.nn.MultiheadAttention(
            embed_dim, heads, batch_first=True, dropout=dropout)
        self.batch_norm = torch.nn.BatchNorm2d(embed_dim)

    def forward(self, x):
        """
        Forward pass of the attention block.
        :param x: Input tensor.
        """
        x_shape = x.shape
        y = self.batch_norm(x)
        y = y.reshape(-1, self.embed_dim,  x_shape[2] * x_shape[3])
        y = torch.transpose(y, 1, 2)
        y, _ = self.mha(y, y, y)
        y = torch.transpose(y, 1, 2)
        y = y.reshape(-1, self.embed_dim, x_shape[2], x_shape[3])
        return x + y


class Unet(torch.nn.Module):

    def __init__(self, channels=(32, 64, 128, 256, 512),
                 strides=(2, 2, 4, 4), filer_shapes=(3, 3, 7, 7),
                 dropout_rate=0.1):
        """
        :param strides: Strides for the pooling layers.
        :param filer_shapes: Filter shapes for the pooling layers.
        """
        super(Unet, self).__init__()

        # Encoder blocks
        self.block_in_1 = ResNetBlock(1, channels[0], 3,
                                      dropout_rate)
        self.block_in_2 = ResNetBlock(channels[0], channels[1], 3,
                                      dropout_rate)
        self.block_in_3 = ResNetBlock(channels[1], channels[2], 3,
                                      dropout_rate)
        self.block_in_4 = ResNetBlock(channels[2], channels[3], 3,
                                      dropout_rate)
        self.block_in_5 = ResNetBlock(channels[3], channels[4], 3,
                                      dropout_rate)

        # Encoder pooling
        self.pool_1 = torch.nn.Conv2d(channels[0], channels[0], filer_shapes[0],
                                      strides[0], padding=strides[0]//2)
        self.pool_2 = torch.nn.Conv2d(channels[1], channels[1], filer_shapes[1],
                                      strides[1], padding=strides[1]//2)
        self.pool_3 = torch.nn.Conv2d(channels[2], channels[2], filer_shapes[2],
                                      strides[2], padding=strides[2]//2)
        self.pool_4 = torch.nn.Conv2d(channels[3], channels[3], filer_shapes[3],
                                      strides[3], padding=strides[3]//2)

        # bottleneck block with attention
        self.bottleneck = ResNetBlock(channels[4], channels[4], 3,
                                      use_attention=True)

        # Decoder blocks
        self.block_out_1 = ResNetBlock(channels[4], channels[3], 3,
                                       dropout_rate)
        self.block_out_2 = ResNetBlock(channels[3], channels[2], 3,
                                       dropout_rate)
        self.block_out_3 = ResNetBlock(channels[2], channels[1], 3,
                                       dropout_rate)
        self.block_out_4 = ResNetBlock(channels[1], channels[0], 3,
                                       dropout_rate)

        # Decoder unpooling
        self.unpool_1 = torch.nn.ConvTranspose2d(
            channels[4], channels[3], filer_shapes[3], strides[3],
            padding=strides[3]//2, output_padding=1)
        self.unpool_2 = torch.nn.ConvTranspose2d(
            channels[3], channels[2], filer_shapes[2], strides[2],
            padding=strides[2]//2, output_padding=1)
        self.unpool_3 = torch.nn.ConvTranspose2d(
            channels[2], channels[1], filer_shapes[1], strides[1],
            padding=strides[1]//2, output_padding=1)
        self.unpool_4 = torch.nn.ConvTranspose2d(
            channels[1], channels[0], filer_shapes[0], strides[0],
            padding=strides[0]//2, output_padding=1)

        # Output layer
        self.ouput = torch.nn.Conv2d(channels[0], 1, 1, padding='same')

    def forward(self, x):
        """
        Forward pass of the network.
        :param x: Input tensor.
        """

        # encoder
        bi1 = self.block_in_1(x)
        bi2 = self.block_in_2(self.pool_1(bi1))
        bi3 = self.block_in_3(self.pool_2(bi2))
        bi4 = self.block_in_4(self.pool_3(bi3))
        bi5 = self.block_in_5(self.pool_4(bi4))

        # bottleneck
        bn = self.bottleneck(bi5)

        # decoder
        bo1 = torch.cat([self.unpool_1(bn), bi4], 1)
        bo2 = torch.cat([self.unpool_2(self.block_out_1(bo1)), bi3], 1)
        bo3 = torch.cat([self.unpool_3(self.block_out_2(bo2)), bi2], 1)
        bo4 = torch.cat([self.unpool_4(self.block_out_3(bo3)), bi1], 1)
        bo5 = self.block_out_4(bo4)
        output = self.ouput(bo5)
        return output


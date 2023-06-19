import mindspore
from mindspore import nn, ops
import numpy as np
import vgg16


class DetectionModel(nn.Cell):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1_up = nn.SequentialCell(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, pad_mode="same",
                      has_bias=True, weight_init="XavierUniform"),
            nn.ReLU()
        )
        self.conv1_down = ContextModule(in_channels, out_channels)
        self.cls = nn.Conv2d(in_channels=2 * out_channels, out_channels=2, kernel_size=1, pad_mode="same",
                             has_bias=True, weight_init="XavierUniform")
        # self.reg = nn.Conv2d(in_channels=2 * out_channels, out_channels=4, kernel_size=1, pad_mode="same")
        self.reg = nn.Conv2d(in_channels=2 * out_channels, out_channels=4 + 4, kernel_size=1, pad_mode="same",
                             has_bias=True, weight_init="XavierUniform")

    def construct(self, x):
        x_up = self.conv1_up(x)
        x_down = self.conv1_down(x)
        # ret = np.concatenate((x_up, x_down), axis=1)
        x = ops.concat((x_up, x_down), axis=1)
        cls = self.cls(x)
        reg = self.reg(x)
        return cls, reg


class ContextModule(nn.Cell):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = nn.SequentialCell(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels // 2, kernel_size=3, pad_mode="same",
                      has_bias=True, weight_init="XavierUniform"),
            nn.ReLU()
        )
        self.conv2_up = nn.SequentialCell(
            nn.Conv2d(in_channels=out_channels // 2, out_channels=out_channels // 2, kernel_size=3, pad_mode="same",
                      has_bias=True, weight_init="XavierUniform"),
            nn.ReLU()
        )
        self.conv2_down = nn.SequentialCell(
            nn.Conv2d(in_channels=out_channels // 2, out_channels=out_channels // 2, kernel_size=3, pad_mode="same",
                      has_bias=True, weight_init="XavierUniform"),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channels // 2, out_channels=out_channels // 2, kernel_size=3, pad_mode="same",
                      has_bias=True, weight_init="XavierUniform"),
            nn.ReLU()
        )

    def construct(self, x):
        x = self.conv1(x)
        x_up = self.conv2_up(x)
        x_down = self.conv2_down(x)
        ret = ops.concat((x_up, x_down), axis=1)
        return ret


class SSH(nn.Cell):
    def __init__(self):
        super().__init__()
        self.vgg = vgg16.vgg16()
        self.conv5_3_maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.m1_detect = DetectionModel(in_channels=128, out_channels=128)
        self.m2_detect = DetectionModel(in_channels=512, out_channels=256)
        self.m3_detect = DetectionModel(in_channels=512, out_channels=256)
        self.m1_dim_red_up = nn.SequentialCell(
            nn.Conv2d(in_channels=512, out_channels=128, kernel_size=1, pad_mode="same", has_bias=True,
                      weight_init="XavierUniform"),
            nn.ReLU()
        )
        self.m1_dim_red_down = nn.SequentialCell(
            nn.Conv2d(in_channels=512, out_channels=128, kernel_size=1, pad_mode="same", has_bias=True,
                      weight_init="XavierUniform"),
            nn.ReLU()
        )
        self.m1_conv1 = nn.SequentialCell(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, pad_mode="same", has_bias=True,
                      weight_init="XavierUniform"),
            nn.ReLU()
        )
        self.bilinear = nn.ResizeBilinear()

    def construct(self, x):
        conv4_3, conv5_3 = self.vgg(x)
        conv5_3_pooling = self.conv5_3_maxpool(conv5_3)
        m1_up = self.m1_dim_red_up(conv5_3)
        # resize_bilinear = nn.ResizeBilinear()
        m1_up = self.bilinear(m1_up, scale_factor=2)
        # m1_up = ops.ResizeBilinearV2(m1_up)
        # m1_up = ops.interpolate(m1_up, mode="bilinear", size=(2 * m1_up.shape[2], 2 * m1_up.shape[3]))
        m1_down = self.m1_dim_red_down(conv4_3)
        m1_out = m1_up + m1_down
        m1_out = self.m1_conv1(m1_out)
        m1_out = self.m1_detect(m1_out)
        m2_out = self.m2_detect(conv5_3)
        m3_out = self.m3_detect(conv5_3_pooling)
        return [m1_out, m2_out, m3_out]


if __name__ == '__main__':
    img = mindspore.Tensor(np.zeros([1, 3, 896, 1024]), mindspore.float32)
    ssh = SSH()
    print(ssh)
    out = ssh.construct(img)
    print(out[0][0].shape, out[0][1].shape)
    print(out[1][0].shape, out[1][1].shape)
    print(out[2][0].shape, out[2][1].shape)

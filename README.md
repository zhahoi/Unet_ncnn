# Unet_ncnn
使用Unet实现指针仪表表盘指针和刻度分割（借助ncnn框架）

## 起

最近做指针仪表检测项目的时候发现，网上写ncnn版本的语义分割代码真的很好，翻遍GitHub都找不出几个，因此只能硬着头皮去写。之前尝试使用DeepLabV3+语义分割模型做指针仪表表盘分割，遇到了很多问题。搞了大半个月还是没解决，去ncnn官网库下搜issues发现有很多人在部署DeepLabV3+都遇到相同的问题，于是毅然决然准备换模型解决。经过重新跑Pytorch模型，转onnx，再转ncnn，经过测试发现没有任何问题，因此有了这个仓库。也算造福大家。

## 承

因为网上语义分割代码参差不齐，还是选择[bubbliiiing](https://github.com/bubbliiiing)这位大神写的代码，选的Pytorch模型是[unet-pytorch](https://github.com/bubbliiiing/unet-pytorch.git)。训练之后，先转onnx，再转ncnn。通过写的测试代码可以正常进行预测。

**依赖库：**

- opencv-3.4.10
- protobuf-3.4.0
- ncnn

## 转

测试权重下载链接：链接：https://pan.baidu.com/s/1cikC-Fkl-M_BWW-nM4POiA?pwd=1234 
提取码：1234

测试结果：[![pkgDIOK.jpg](https://s21.ax1x.com/2024/07/02/pkgDIOK.jpg)](https://imgse.com/i/pkgDIOK)

注：编写的ncnn前处理、后处理的推理代码理论上可以支持[bubbliiiing](https://github.com/bubbliiiing)这位大神的以下几个仓库，因为Pytorch代码的预处理和后处理代码完全一致。但是值得注意的是，有的语义分割模型可能存在不支持onnx算法（PSPNet)，这里仅测试过DeepLabV3+（分割结果还存在一些问题）和Unet两个模型。

[![pkgrCTg.jpg](https://s21.ax1x.com/2024/07/02/pkgrCTg.jpg)](https://imgse.com/i/pkgrCTg)

## 合

最后感谢[bubbliiiing](https://github.com/bubbliiiing)这位大神提供的Unet的Pytorch源码。

- [unet-pytorch](https://github.com/bubbliiiing/unet-pytorch.git)


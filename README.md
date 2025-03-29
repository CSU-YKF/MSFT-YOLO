# MSFT-YOLO

**MSFT-YOLO** 是一个基于 YOLOv5-6.0 的改进模型，参考了论文 *"MSFT-YOLO: Improved YOLOv5 Based on Transformer for Detecting Defects of Steel Surface"*，旨在提升钢表面缺陷检测的精度和实时性。本项目在 YOLOv5 的基础上实现了 Transformer 增强的主干网络（TRANS 模块）和 BiFPN 特征融合网络，并支持多阶段训练策略。

## 环境配置与数据集准备

### Python 环境
本项目以 Python 3.8 为基础环境，推荐使用 Conda 创建虚拟环境以确保依赖隔离：

```bash
conda create -n yolov5 python=3.8
conda activate yolov5
```

安装项目依赖：

```bash
pip install -r requirements.txt
```

**注意**：PyTorch 需要根据你的 GPU 型号和 CUDA 版本安装匹配的版本。例如，若使用 CUDA 12.1，可运行：

```bash
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu121
```

请根据你的硬件配置（如 NVIDIA 显卡型号和 CUDA 版本）调整安装命令，详情参考 [PyTorch 官网](https://pytorch.org/get-started/locally/)。

### 数据集
本项目使用 **NEU-DET** 数据集，包含 6 类钢表面缺陷（crazing、inclusion、patches、pitted_surface、rolled-in_scale、scratches）。数据集需按以下方式准备：

1. **存放路径**：
   将数据集放置在项目根目录下的 `./dataset/` 文件夹中。
   ```
   ./dataset/
   ├── images/
   │   ├── train/
   │   └── val/
   └── labels/
       ├── train/
       └── val/
   ```

2. **格式要求**：
   数据需转换为 YOLO 格式，即每张图片对应一个 `.txt` 标签文件，包含类别索引和归一化坐标（`class x_center y_center width height`）。

3. **划分比例**：
   按 80% 训练集和 20% 验证集划分，无测试集（80/20/0）。你可以使用脚本或手动划分。

4. **配置文件**：
   数据集配置位于 `./data/NEU-DET.yaml`，内容如下：
   ```yaml
   train: ./dataset/images/train/
   val: ./dataset/images/val/
   nc: 6
   names: ['crazing', 'inclusion', 'patches', 'pitted_surface', 'rolled-in_scale', 'scratches']
   ```

## 模型修改

### 主干网络（Backbone）改进
论文在 YOLOv5 的 `CSPDarknet53` 主干网络末端添加了 **TRANS 模块**（Transformer 编码器），以捕获全局上下文信息，提升特征提取能力。

#### 理解 YOLOv5 的 Backbone
在 YOLOv5-6.0 中，主干网络定义于 `models/yolov5l.yaml`（本文以 YOLOv5l 为基线）。它由一系列卷积层（`Conv`）、CSP 瓶颈层（`C3`）和空间金字塔池化（`SPP`）块组成。为实现 MSFT-YOLO，我们复制 `yolov5l.yaml` 并重命名为 `yolomsft.yaml`。

#### 定义 TRANS 模块
TRANS 模块是一个 Transformer 编码器块，包含多头自注意力（MHSA）和前馈网络（FFN）。在 `models/common.py` 中添加以下代码（约 58-85 行）：

```python
class TRANS(nn.Module):
    def __init__(self, in_channels, num_heads=8, dropout=0.1):
        super(TRANS, self).__init__()
        self.d_model = in_channels
        assert self.d_model % num_heads == 0, f"d_model ({self.d_model}) must be divisible by num_heads ({num_heads})"
        self.proj = nn.Conv2d(in_channels, self.d_model, 1)
        self.attn = nn.MultiheadAttention(self.d_model, num_heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(self.d_model)
        self.ffn = nn.Sequential(
            nn.Linear(self.d_model, self.d_model * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.d_model * 4, self.d_model)
        )
        self.norm2 = nn.LayerNorm(self.d_model)
        self.proj_back = nn.Conv2d(self.d_model, in_channels, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        x = self.proj(x).flatten(2).permute(2, 0, 1)
        attn_output, _ = self.attn(x, x, x)
        x = self.norm1(x + attn_output)
        ffn_output = self.ffn(x)
        x = self.norm2(x + ffn_output)
        x = x.permute(1, 2, 0).view(b, self.d_model, h, w)
        return self.proj_back(x)
```

#### 修改 Backbone 配置
在 `models/yolomsft.yaml` 的 `backbone` 部分，SPP 块后添加 TRANS 模块：
```yaml
backbone:
  - [-1, 1, Conv, [64, 6, 2, 2]]    # 0-P1/2
  - [-1, 1, C3, [64]]              # 1
  - [-1, 1, Conv, [128, 3, 2]]     # 2-P2/4
  - [-1, 2, C3, [128]]             # 3
  - [-1, 1, Conv, [256, 3, 2]]     # 4-P3/8
  - [-1, 3, C3, [256]]             # 5
  - [-1, 1, Conv, [512, 3, 2]]     # 6-P4/16
  - [-1, 3, C3, [512]]             # 7
  - [-1, 1, Conv, [1024, 3, 2]]    # 8-P5/32
  - [-1, 1, SPP, [1024, [5, 9, 13]]]  # 9
  - [-1, 1, TRANS, [1024]]         # 10 (Add TRANS here)
```
- `[-1, 1, TRANS, [1024]]` 表示从上一层（SPP，1024 通道）输入，添加一层 TRANS 模块，输入和输出通道均为 1024。

#### 更新解析逻辑
在 `models/yolo.py` 的 `parse_model` 函数中，添加对 TRANS 模块的支持：
```python
if m in [Conv, GhostConv, Bottleneck, ..., C3Ghost, TRANS]:
    c1, c2 = ch[f], args[0]
    if c2 != no:  # if not output
        c2 = make_divisible(c2 * gw, 8)
    args = [c1, c2, *args[1:]]
```

### 瓶颈层（Neck）改进
论文用 **BiFPN** 替换了 YOLOv5 的 PANet，实现更高效的多尺度特征融合，支持可学习权重。

#### 理解 YOLOv5 的 Neck
在原始 `yolov5l.yaml` 中，Neck 部分（层 10-23）使用 PANet 融合 P3、P4 和 P5 尺度的特征。我们将用 BiFPN 替代这一部分。

#### 定义 BiFPN 模块
在 `models/common.py` 中添加 BiFPN 实现（约 88-128 行）：
```python
class BiFPN(nn.Module):
    def __init__(self, in_channels_list, out_channels):
        super(BiFPN, self).__init__()
        self.out_channels = out_channels
        self.conv_p3 = nn.Conv2d(in_channels_list[0], out_channels, 1)
        self.conv_p4 = nn.Conv2d(in_channels_list[1], out_channels, 1)
        self.conv_p5 = nn.Conv2d(in_channels_list[2], out_channels, 1)
        self.conv6_up = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        self.conv5_up = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        self.conv4_up = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        self.conv6_td = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        self.conv5_td = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        self.conv4_out = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        self.weights = nn.Parameter(torch.ones(6, 3))

    def forward(self, inputs):
        p3_in, p4_in, p5_in = inputs
        p3_in = self.conv_p3(p3_in)
        p4_in = self.conv_p4(p4_in)
        p5_in = self.conv_p5(p5_in)
        w = F.relu(self.weights)
        w = w / (w.sum(1, keepdim=True) + 1e-6)
        p5_td = self.conv6_td(p5_in)
        p4_td = self.conv5_up(p4_in + F.interpolate(p5_td, size=p4_in.shape[2:], mode='nearest'))
        p3_td = self.conv4_up(p3_in + F.interpolate(p4_td, size=p3_in.shape[2:], mode='nearest'))
        p3_out = self.conv4_out(w[0, 0] * p3_in + w[0, 1] * p3_td + w[0, 2] * F.interpolate(p4_td, size=p3_in.shape[2:], mode='nearest'))
        p4_out = self.conv5_td(w[1, 0] * p4_in + w[1, 1] * p4_td + w[1, 2] * F.interpolate(p5_td, size=p4_in.shape[2:], mode='nearest'))
        p5_out = self.conv6_up(w[2, 0] * p5_in + w[2, 1] * p5_td)
        return [p3_out, p4_out, p5_out]
```

#### 修改 Neck 配置
在 `models/yolomsft.yaml` 的 `head` 部分，用 BiFPN 替换 PANet，并从 P3、P4、P5（层 5、7、10）获取输入：
```yaml
head:
  - [[5, 7, 10], 1, BiFPN, [[256, 512, 1024], 256]]  # 11
  - [[-1], 1, Detect, [6, [[10, 13, 16, 30, 33, 23], [30, 61, 62, 45, 59, 119], [116, 90, 156, 198, 373, 326]], [256, 256, 256]]]  # 12
```
- `[5, 7, 10]` 表示从层 5（256 通道）、7（512 通道）和 10（1024 通道）输入。
- `[[256, 512, 1024], 256]` 表示输入通道列表和输出通道数。

#### 更新解析逻辑
在 `models/yolo.py` 的 `parse_model` 函数中，添加对 BiFPN 的支持：
```python
elif m is BiFPN:
    c1 = [ch[x] for x in f]  # Input channels from multiple layers
    c2 = args[1]  # Output channels
    args = [c1, c2, *args[2:]]
```

## 训练配置

### 超参数设置
在 `data/hyps/hyp.scratch.yaml` 中更新超参数（参考论文建议）：
```yaml
lr0: 0.02            # 初始学习率
lrf: 0.1             # 学习率衰减因子
momentum: 0.937      # SGD 动量
weight_decay: 0.0005 # 权重衰减
warmup_epochs: 3.0   # 预热周期
```

### 模型训练
运行以下命令开始训练：
```bash
python train.py 
    --img 640 
    --batch 16 
    --epochs 10 
    --data data/NEU-DET.yaml 
    --cfg models/yolomsft.yaml
    --weights '' 
    --hyp data/hyps/hyp.scratch.yaml 
    --device 0
```
- `--img 640`：输入图像尺寸 640x640。
- `--batch 16`：批次大小（根据显存调整）。
- `--epochs 10`：训练轮次（可根据需求增加）。
- `--device 0`：使用 GPU 0。

#### 多阶段训练
论文提出了一种多阶段训练策略，以减少假阳性并提升检测性能：

1. **第一阶段**：使用 6 个缺陷类别进行初始训练：
   ```bash
   python train.py --img 640 --batch 16 --epochs 10 --data data/NEU-DET.yaml --cfg models/yolomsft.yaml --weights '' --hyp data/hyps/hyp.scratch.yaml --device 0
   ```

2. **收集无缺陷样本**：
   在验证集上运行推理，收集假阳性样本（无缺陷区域）：
   ```bash
   python detect.py --weights runs/train/exp/weights/best.pt --source dataset/images/val/ --save-txt
   ```

3. **第二阶段**：
   - 更新 `data/NEU-DET.yaml`，添加“无缺陷”类别：
     ```yaml
     nc: 7
     names: ['crazing', 'inclusion', 'patches', 'pitted_surface', 'rolled-in_scale', 'scratches', 'defect-free']
     ```
   - 将收集的无缺陷样本加入数据集。
   - 使用第一阶段的权重继续训练：
     ```bash
     python train.py --img 640 --batch 2 --epochs 50 --data data/NEU-stage2.yaml --cfg models/yolomsft.yaml --weights checkpoints/yolomsft-stage1-best.pt --hyp data/hyps/hyp.scratch.yaml --device 0
     ```

## 验证与测试
训练完成后，验证模型性能：
```bash
python val.py --data data/NEU-DET.yaml --weights runs/train/exp/weights/best.pt --task val --img 640
```
预期结果：mAP@0.5:0.95 约为 75.2（参考论文），具体取决于硬件和训练轮次。

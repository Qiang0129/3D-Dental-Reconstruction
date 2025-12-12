下面是你这段内容整理成的 **Markdown 格式**（直接复制到 `README.md` 或 `REDME.md` 里即可）：

````markdown
## 一、Python 环境的安装

1. 首先看看环境有没有 `python3.8` 或者 `python3.10`（不要用 3.11，nnU-Net v2 不适配）
   ```bash
   pyenv versions
````

2. 创建一个 Python 环境

   ```bash
   pyenv virtualenv 3.10.12 teeth-nnunet
   ```

3. 激活环境（进入 python3.10.12 环境）

   ```bash
   pyenv activate teeth-nnunet
   ```

4. 检查当前环境的 Python 版本是否正确

   ```bash
   python --version
   ```

## 二、安装 PyTorch + nnU-Net

### 1）确认 CUDA 版本

查看自己的 CUDA 版本：

```bash
nvidia-smi
```

例如：我的 CUDA-version 是 **12.3**，那么建议安装 **PyTorch + CUDA 12.1** 版本。

安装 PyTorch：

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

安装完成后测试：

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda)"
```

正确的话会显示类似：

```
True 12.1
```

### 2）安装 nnU-Net v2

安装指定版本：

```bash
pip install nnunetv2==2.3.1
```

安装 nnU-Net 所需依赖（医学图像库等）：

```bash
pip install SimpleITK nibabel scikit-image opencv-python tqdm pandas
```

### 3）配置 nnU-Net 环境变量（Linux 示例）

一次运行下方代码：

```bash
export nnUNet_raw="/root/learn/nnUNet_raw"
export nnUNet_preprocessed="/root/learn/nnUNet_preprocessed"
export nnUNet_results="/root/learn/nnUNet_results"

echo 'export nnUNet_raw="/root/learn/nnUNet_raw"' >> ~/.bashrc
echo 'export nnUNet_preprocessed="/root/learn/nnUNet_preprocessed"' >> ~/.bashrc
echo 'export nnUNet_results="/root/learn/nnUNet_results"' >> ~/.bashrc

source ~/.bashrc
```

## 三、训练与推理流程

1. 构建 nnU-Net 能够识别的数据集（在项目根目录执行）

   ```bash
   python preprocessing\prepare_dataset.py
   ```

2. 数据预处理（原始图片转灰度图、标签轮廓线膨胀等）

   ```bash
   nnUNetv2_plan_and_preprocess -d 501 --verify_dataset_integrity
   ```

3. 开始训练（默认：学习率自动衰减、训练轮次 1000、五折交叉验证）

   ```bash
   nnUNetv2_train 501 2d 0 -device cuda
   ```

4. 测试训练好的模型分割效果

   > 说明：可能肉眼只能看到黑色背景，因为线条标签值为 1（灰色很接近黑色），背景为 0（黑→白：0→255）

   ```bash
   python inference/predict.py
   ```

5. 安装骨架化算法所需包（用于下游可视化：让轮廓线更细）

   ```bash
   pip install opencv-contrib-python
   ```

6. 可视化处理（将牙齿轮廓 1 转化为 255 白色）

   ```bash
   python inference/visualize_only.py
   ```

```
::contentReference[oaicite:0]{index=0}
```

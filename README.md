一、python环境的安装 
1、首先看看环境有没有python3.8 或者 python3.10 不要用3.11 nn-unetv2不适配 ---pyenv versions 
2、创建一个python环境 ---pyenv virtualenv 3.10.12 teeth-nnunet 
3、激活环境进去python3.10.12环境 ---pyenv activate teeth-nnunet 
4、然后检查一下当前环境的python版本是不是之前创建的版本 ---python --version (这条命令会打印当前环境中的python版本)

二、安装PyTorch + nnU-Net 
1、先查看自己的cuda版本 ---nvidia-smi 
2、我的CUDA-version是12.3 那么就需要安装PyTorch + CUDA 12.1版本 首先安装pytorch： ---pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 安装完成后测试： ---python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda)" (正确的话就会显示True 12.1)

3、安装nnU-Netv2 ---pip install nnunetv2==2.3.1 安装 nnU-Net 所需依赖（医学图像库）
---pip install SimpleITK nibabel scikit-image opencv-python tqdm pandas

确保环境变量正确配置： 一次运行下方代码： export nnUNet_raw="/root/learn/nnUNet_raw" export nnUNet_preprocessed="/root/learn/nnUNet_preprocessed" export nnUNet_results="/root/learn/nnUNet_results" echo 'export nnUNet_raw="/root/learn/nnUNet_raw"' >> ~/.bashrc echo 'export nnUNet_preprocessed="/root/learn/nnUNet_preprocessed"' >> ~/.bashrc echo 'export nnUNet_results="/root/learn/nnUNet_results"' >> ~/.bashrc source ~/.bashrc

1、构建nnunet能够识别的数据集：在项目的根目录执行命令： --- python preprocessing\prepare_dataset.py 
2、数据预处理（原始图片转为灰度图、标签数据轮廓线条的膨胀化处理） --- nnUNetv2_plan_and_preprocess -d 501 --verify_dataset_integrity 
3、开始训练（默认：学习率自动衰减、训练轮次为1000轮、五折交叉验证） --- nnUNetv2_train 501 2d 0 -device cuda 
4、测试训练好的模型的分割效果（这里的结果为肉眼只能看到黑色背景，线条的颜色为1（灰色，非常接近黑色，所以看不到），背景为0（黑——>白：0——>255）） ---python inference/predict.py 
5、安装骨架化算法的包（为下游的可视化做准备，也就是将轮廓线条变细） ---pip install opencv-contrib-python 
6、可视化处理（将牙齿轮廓1转化为255白色） --- python inference/visualize_only.py

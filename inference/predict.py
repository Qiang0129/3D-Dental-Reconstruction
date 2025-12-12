import os
import cv2
import numpy as np

# ========================= 配置区 ========================= #
DATASET_ID = 501
CONFIG = "2d"
FOLD = 0
TRAINER = "nnUNetTrainer"   # 如果你用的是自定义Trainer，比如MyTeethTrainer，就改成那个名字
DEVICE = "cuda"

PROJECT_ROOT = r"E:\XiaoQiang\projects\teeth-boundary-nnunet"
MODEL_DIR = os.path.join(
    PROJECT_ROOT,
    f"nnUNet_results/Dataset{DATASET_ID}_TeethBoundary/{TRAINER}__nnUNetPlans__{CONFIG}/fold_{FOLD}"
)

CHECKPOINT_PATH = os.path.join(MODEL_DIR, "checkpoint_best.pth")

TEST_INPUT = os.path.join(PROJECT_ROOT, "inference/test_images")
PRED_OUTPUT = os.path.join(PROJECT_ROOT, "inference/predictions")
VIS_OUTPUT = os.path.join(PROJECT_ROOT, "inference/vis_results")

# ========================================================== #

# 确保目录存在
os.makedirs(TEST_INPUT, exist_ok=True)
os.makedirs(PRED_OUTPUT, exist_ok=True)
os.makedirs(VIS_OUTPUT, exist_ok=True)

# 预处理：将测试图像转换为 nnUNet 格式
print("\n 预处理测试图像...")
TEMP_INPUT = os.path.join(PROJECT_ROOT, "inference/test_images_nnunet_format")
os.makedirs(TEMP_INPUT, exist_ok=True)

# 清空临时目录
for f in os.listdir(TEMP_INPUT):
    os.remove(os.path.join(TEMP_INPUT, f))

for name in os.listdir(TEST_INPUT):
    if not name.lower().endswith(('.png', '.jpg', '.jpeg')):
        continue
    
    src_path = os.path.join(TEST_INPUT, name)
    stem = os.path.splitext(name)[0]
    
    # 转换为 nnUNet 命名格式: 原文件名_0000.png
    dst_name = f"{stem}_0000.png"
    dst_path = os.path.join(TEMP_INPUT, dst_name)
    
    # 读取图像并转换为灰度图(因为训练时使用的是灰度图)
    img = cv2.imread(src_path, cv2.IMREAD_GRAYSCALE)
    if img is not None:
        cv2.imwrite(dst_path, img)
        print(f"  {name} -> {dst_name} (转为灰度图)")

# 执行 nnUNet 推理
print("\n 开始 nnUNetv2 推理...")
print(f"使用模型路径：{CHECKPOINT_PATH}")

cmd = (
    f"nnUNetv2_predict "
    f"-i \"{TEMP_INPUT}\" "
    f"-o \"{PRED_OUTPUT}\" "
    f"-d {DATASET_ID} "
    f"-c {CONFIG} "
    f"-f {FOLD} "
    f"-tr {TRAINER} "      # 这里是 nnUNetTrainer
    f"-device {DEVICE} "
    f"-chk checkpoint_best.pth"  # 使用完整的 checkpoint 文件名
)

print(f"运行命令：\n{cmd}\n")
os.system(cmd)

# 可视化预测结果
print("开始生成可视化结果...")

for name in os.listdir(TEST_INPUT):
    if not name.lower().endswith(('.png', '.jpg', '.jpeg')):
        continue

    img_path = os.path.join(TEST_INPUT, name)
    stem, _ = os.path.splitext(name)

    # nnUNet 输出的预测文件名格式: 原文件名.png (不带_0000后缀)
    pred_path = os.path.join(PRED_OUTPUT, f"{stem}.png")
    
    if not os.path.exists(pred_path):
        print(f"跳过 {name},未找到对应预测结果: {os.path.basename(pred_path)}")
        continue

    # 读取原始图像(彩色)
    img = cv2.imread(img_path)
    # 读取预测掩码
    mask = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        print(f"读取预测掩膜失败:{pred_path}")
        continue

    # 将掩码值从1转换为255,以便可视化为白色
    mask_vis = (mask > 0).astype(np.uint8) * 255
    
    # 保存白色轮廓图
    mask_out_path = os.path.join(VIS_OUTPUT, f"{stem}_mask.png")
    cv2.imwrite(mask_out_path, mask_vis)
    
    # 创建红色叠加可视化
    color_mask = np.zeros_like(img)
    color_mask[mask > 0] = (0, 0, 255)  # 红色叠加
    
    vis = cv2.addWeighted(img, 0.7, color_mask, 0.3, 0)
    overlay_out_path = os.path.join(VIS_OUTPUT, f"{stem}_overlay.png")
    cv2.imwrite(overlay_out_path, vis)
    
    print(f"✓ {name}: 掩码图 + 叠加图")

print("\n推理与可视化全部完成！")
print(f"预测输出: {PRED_OUTPUT}")
print(f"可视化图: {VIS_OUTPUT}")

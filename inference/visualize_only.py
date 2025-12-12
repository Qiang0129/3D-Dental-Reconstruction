import os
import cv2
import numpy as np

PROJECT_ROOT = r"E:\XiaoQiang\projects\teeth-boundary-nnunet"
TEST_INPUT = os.path.join(PROJECT_ROOT, "inference/test_images")
PRED_OUTPUT = os.path.join(PROJECT_ROOT, "inference/predictions")
VIS_OUTPUT = os.path.join(PROJECT_ROOT, "inference/vis_results")

# ========== 线条粗细调节 ==========
LINE_THICKNESS = 5  # 调整此值: 1=单像素, 2=双像素, 3=三像素...
# ========== 叠加图颜色调节 ==========
OVERLAY_COLOR = (0, 255, 0)  # BGR格式: (蓝, 绿, 红)
# ==================================

os.makedirs(VIS_OUTPUT, exist_ok=True)
print("开始生成可视化结果...")

for name in os.listdir(TEST_INPUT):
    if not name.lower().endswith((".png", ".jpg", ".jpeg")):
        continue

    img_path = os.path.join(TEST_INPUT, name)
    stem, _ = os.path.splitext(name)

    pred_path = os.path.join(PRED_OUTPUT, f"{stem}.png")
    if not os.path.exists(pred_path):
        print(f"跳过 {name}，未找到预测结果")
        continue

    # 读取原图和预测掩码
    img = cv2.imread(img_path)
    mask = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        print(f"读取掩膜失败: {pred_path}")
        continue

    # 转换为二值图(0/255)
    mask_bin = (mask > 0).astype(np.uint8) * 255

    # 使用Zhang-Suen骨架化算法提取单像素中心线
    skeleton = cv2.ximgproc.thinning(mask_bin, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN)
    
    # 如果需要加粗线条,进行膨胀操作
    if LINE_THICKNESS > 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (LINE_THICKNESS, LINE_THICKNESS))
        skeleton = cv2.dilate(skeleton, kernel, iterations=1)
    
    # 保存白色轮廓图(骨架化单线条)
    cv2.imwrite(os.path.join(VIS_OUTPUT, f"{stem}_edge.png"), skeleton)
    
    # 创建叠加图(骨架化)
    overlay = img.copy()
    overlay[skeleton > 0] = OVERLAY_COLOR
    vis = cv2.addWeighted(img, 0.7, overlay, 0.3, 0)
    cv2.imwrite(os.path.join(VIS_OUTPUT, f"{stem}_overlay.png"), vis)
    
    print(f"✓ {name}: 骨架化轮廓线已生成")

print(f"\n可视化完成！输出: {VIS_OUTPUT}")
print("- *_edge.png: 白色骨架化轮廓线(单像素宽)")
print(f"- *_overlay.png: {OVERLAY_COLOR}叠加图")

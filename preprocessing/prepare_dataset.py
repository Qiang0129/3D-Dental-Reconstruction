import os
import cv2
import numpy as np
import json
from tqdm import tqdm
from skimage.morphology import dilation, disk

# ========================= 配置区 ========================= #

RAW_IMG_DIR = "./dataset/raw_images"      # 原始牙齿照片路径
RAW_LABEL_DIR = "./dataset/raw_label"     # 标注（轮廓）路径

DATASET_ID = 501
DATASET_NAME = "TeethBoundary"

DILATE_RADIUS = 8  # 标签膨胀半径（8像素）

# nnU-Net 的三个环境变量之一，必须已经 setx 过
NNUNET_RAW = os.getenv("nnUNet_raw")

# ============================================================ #


def ensure_dir(path):
    """创建文件夹"""
    if not os.path.exists(path):
        os.makedirs(path)


def binarize_and_dilate(label_path):
    """
    读取轮廓图 → 二值化 → 膨胀 → 输出 0/1 标签图
    自动处理：黑底黑线 / 白底黑线 / 黑底白线
    """
    img = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise RuntimeError(f"无法读取标签文件: {label_path}")

    # ================= 自动识别线条颜色 ================= #
    # 均值越大说明越白，越小说明是黑色图
    mean_val = np.mean(img)

    if mean_val > 200:
        # 几乎全白（极少数情况）
        mask = (img < 128).astype(np.uint8)
    elif mean_val > 128:
        # 白底黑线
        mask = (img < 128).astype(np.uint8)
    else:
        # 黑底黑线（你的情况）
        mask = (img < 50).astype(np.uint8)

    # ================= 膨胀线条（让轮廓更粗） ================= #
    structure = disk(DILATE_RADIUS)
    mask = dilation(mask, structure)

    # 转换成 0/1 掩膜（nnU-Net 要求）
    mask = (mask > 0).astype(np.uint8)

    return mask


def main():
    print(f"当前 nnUNet_raw 路径: {NNUNET_RAW}")
    if NNUNET_RAW is None:
        raise RuntimeError("环境变量 nnUNet_raw 未设置！请先运行 setx nnUNet_raw '路径'")

    # 输出目录
    dataset_dir = os.path.join(NNUNET_RAW, f"Dataset{DATASET_ID}_{DATASET_NAME}")
    imagesTr = os.path.join(dataset_dir, "imagesTr")
    labelsTr = os.path.join(dataset_dir, "labelsTr")
    imagesVa = os.path.join(dataset_dir, "imagesVa")
    imagesTs = os.path.join(dataset_dir, "imagesTs")

    ensure_dir(imagesTr)
    ensure_dir(labelsTr)
    ensure_dir(imagesTs)

    imgs = sorted(os.listdir(RAW_IMG_DIR))
    lbls = sorted(os.listdir(RAW_LABEL_DIR))
    assert len(imgs) == len(lbls), "原图与标签数量不一致！"

    N = len(imgs)
    print(f"==== 开始构建 nnU-Net v2 数据集，共 {N} 组 ====")

    training_entries = []

    for idx, img_name in tqdm(enumerate(imgs), total=N):
        img_path = os.path.join(RAW_IMG_DIR, img_name)
        lbl_path = os.path.join(RAW_LABEL_DIR, lbls[idx])

        # 读取为灰度图（单通道）
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise RuntimeError(f"无法读取原图: {img_path}")

        mask = binarize_and_dilate(lbl_path)

        cid = f"case_{idx:03d}"
        img_out = os.path.join(imagesTr, f"{cid}_0000.png")
        lbl_out = os.path.join(labelsTr, f"{cid}.png")
        visual_out = os.path.join(imagesVa, f"{cid}.png")

        # 保存图像与标签
        cv2.imwrite(img_out, img)
        cv2.imwrite(lbl_out, mask )  # 保存时仍为可视化友好的 0/1   
        cv2.imwrite(visual_out, mask * 255)  # 可视化标签，0/255    
        training_entries.append({
            "image": f"./imagesTr/{cid}_0000.png",
            "label": f"./labelsTr/{cid}.png"
        })

    print("原图与标签保存完成")

    # ================== 写入标准 nnU-Net v2 dataset.json ================== #
    dataset_json = {
        "labels": {
            "background": 0,
            "tooth_boundary": 1
        },
        "channel_names": {
            "0": "grayscale"
        },
        "file_ending": ".png",
        "numTraining": N,
        "regions_class_order": [1],
        "training": training_entries,
        "test": []
    }

    json_path = os.path.join(dataset_dir, "dataset.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(dataset_json, f, indent=4)

    print(f"dataset.json 已生成: {json_path}")
    print("数据准备完成，可以运行：")
    print("    nnUNetv2_plan_and_preprocess -d 501 --verify_dataset_integrity")


if __name__ == "__main__":
    main()

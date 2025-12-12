import os
import sys

# 如果项目中有自定义 Trainer，确保可以被导入
sys.path.append(os.path.dirname(__file__))

from nnunetv2.run.run_training import run_training

# 训练入口
if __name__ == "__main__":
    dataset_id = 501
    configuration = "2d"
    fold = 0

    # 运行训练
    run_training(
        dataset_name_or_id=dataset_id,
        configuration=configuration,
        fold=fold,
        tr="MyTeethTrainer",     # 使用自定义 Trainer
        p="nnUNetPlans",          # 默认配置
        device="cuda",            # GPU 训练
        continue_training=False,  # 是否从 checkpoint 继续
        num_gpus=1,             # 使用单 GPU
        pretrained_weights=None
    )

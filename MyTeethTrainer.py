from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


class MyTeethTrainer(nnUNetTrainer):
    """
    自定义 Trainer，用于牙齿轮廓分割。
    控制训练轮数、验证频率、保存间隔。
    """

    def __init__(self, plans, configuration, fold, **kwargs):
        super().__init__(plans, configuration, fold, **kwargs)
        self.num_epochs = 200      # 训练 200 轮（默认是 1000）
        self.save_every = 20       # 每 20 轮保存模型
        self.val_every = 20        # 每 20 轮验证一次

    def maybe_update_lr(self, epoch):
        """学习率衰减策略"""
        if epoch in [50, 100, 150]:
            old_lr = self.optimizer.param_groups[0]['lr']
            new_lr = old_lr * 0.5
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = new_lr
            self.print_to_log_file(f"Epoch {epoch}: LR {old_lr:.6f} → {new_lr:.6f}")

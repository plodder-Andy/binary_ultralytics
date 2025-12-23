# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""
ReActNet-YOLO 训练脚本（支持知识蒸馏）

使用方法:
    python train_reactnet.py --model yolov8n-react.yaml --data coco.yaml --epochs 150
    python train_reactnet.py --model yolov8n-react.yaml --data coco.yaml --epochs 150 \
        --teacher yolov8n.pt --kd_alpha 0.5 --kd_temp 2.0

配置文件使用:
    python train_reactnet.py --cfg reactnet_hyp.yaml
"""

from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from ultralytics import YOLO
from ultralytics.data import build_yolo_dataset
from ultralytics.nn.modules import Conv
from ultralytics.utils import LOGGER, TQDM, colorstr
from ultralytics.utils.loss import DistributionLoss
from ultralytics.utils.plotting import plot_results
from ultralytics.utils.torch_utils import unwrap_model


class KDTrainer:
    """
    知识蒸馏训练器 for ReActNet-YOLO

    特点:
    1. 支持二值化激活训练 (BinaryConv/BinaryC2f)
    2. 可选知识蒸馏 (使用全精度教师模型)
    3. 集成 ReActNet 训练超参数
    """

    def __init__(self, overrides: dict | None = None):
        """
        初始化训练器

        Args:
            cfg: 配置文件路径或字典
            overrides: 参数覆盖字典
        """
        # 直接从 overrides 中提取参数，不使用 get_cfg 验证
        self.model_path = overrides.get('model', 'yolov8n-react.yaml')
        self.data_path = overrides.get('data', '')
        self.epochs = overrides.get('epochs', 150)
        self.batch_size = overrides.get('batch', 32)
        self.img_size = overrides.get('imgsz', 640)
        self.device = str(overrides.get('device', '0'))

        # 学习率参数
        self.lr0 = overrides.get('lr0', 0.001)
        self.lrf = overrides.get('lrf', 0.01)
        self.momentum = overrides.get('momentum', 0.9)
        self.weight_decay = overrides.get('weight_decay', 1e-5)

        # 蒸馏参数
        self.teacher_path = overrides.get('teacher', None)
        self.kd_alpha = overrides.get('kd_alpha', 0.0)
        self.kd_temp = overrides.get('kd_temp', 1.0)

        # 其他参数
        self.save_dir = overrides.get('save_dir', 'runs/train')
        self.project = overrides.get('project', 'runs/train')
        self.name = overrides.get('name', 'exp')
        self.cos_lr = overrides.get('cos_lr', False)
        self.warmup_epochs = overrides.get('warmup_epochs', 3.0)
        self.amp = overrides.get('amp', True)
        self.val = overrides.get('val', True)
        self.save_period = overrides.get('save_period', -1)
        self.plots = overrides.get('plots', True)

        # 初始化组件
        self.model = None
        self.teacher_model = None
        self.optimizer = None
        self.scheduler = None
        self.scaler = torch.cuda.amp.GradScaler()
        self.train_loader = None
        self.val_loader = None

        # 损失函数
        self.kd_criterion = DistributionLoss()

        LOGGER.info(f"{colorstr('bold', 'ReActNet-YOLO KDTrainer')}")
        LOGGER.info(f"  - Model: {self.model_path}")
        LOGGER.info(f"  - Data: {self.data_path}")
        LOGGER.info(f"  - Epochs: {self.epochs}")
        LOGGER.info(f"  - Batch: {self.batch_size}")
        LOGGER.info(f"  - LR: {self.lr0}")
        if self.teacher_path:
            LOGGER.info(f"  - Teacher: {self.teacher_path}")
            LOGGER.info(f"  - KD Alpha: {self.kd_alpha}, Temp: {self.kd_temp}")

    def setup_model(self):
        """初始化模型"""
        LOGGER.info("-" * 60)
        LOGGER.info("Setting up model...")

        # 加载学生模型
        self.model = YOLO(self.model_path)
        self.model.to(self.device)

        # 冻结部分层（可选）
        freeze = getattr(self.args, 'freeze', None)
        if freeze:
            if isinstance(freeze, int):
                freeze = list(range(freeze))
            for i, param in enumerate(self.model.parameters()):
                if i in freeze:
                    param.requires_grad = False
            LOGGER.info(f"Froze first {freeze[-1]+1 if isinstance(freeze, list) else freeze} layers")

        # 加载教师模型
        if self.teacher_path:
            self._setup_teacher_model()

        LOGGER.info("Model setup complete.")

    def _setup_teacher_model(self):
        """设置教师模型"""
        LOGGER.info(f"Loading teacher model: {self.teacher_path}")
        self.teacher_model = YOLO(self.teacher_path)
        self.teacher_model.fuse()
        self.teacher_model.info()
        self.teacher_model.to(self.device)
        self.teacher_model.eval()

        # 冻结教师模型
        for param in self.teacher_model.parameters():
            param.requires_grad = False

        LOGGER.info("Teacher model loaded and frozen.")

    def setup_dataloader(self):
        """构建数据加载器"""
        LOGGER.info("-" * 60)
        LOGGER.info("Setting up dataloaders...")

        # 训练数据集
        train_dataset = build_yolo_dataset(
            self.args, self.data_path, self.batch_size, self.data, mode='train', stride=32
        )
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.args.workers,
            pin_memory=True,
            collate_fn=train_dataset.collate_fn,
        )

        # 验证数据集
        if self.val:
            val_dataset = build_yolo_dataset(
                self.args, self.data_path, self.batch_size, self.data, mode='val', stride=32
            )
            self.val_loader = DataLoader(
                val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.args.workers,
                pin_memory=True,
                collate_fn=val_dataset.collate_fn,
            )

        LOGGER.info(f"Train batches: {len(self.train_loader)}")

    def setup_optimizer(self):
        """设置优化器和学习率调度器"""
        LOGGER.info("-" * 60)
        LOGGER.info("Setting up optimizer...")

        # 优化器
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.lr0,
            weight_decay=self.weight_decay,
        )

        # 学习率调度器
        if self.cos_lr:
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.epochs,
                eta_min=self.lr0 * self.lrf,
            )
        else:
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=30,
                gamma=0.1,
            )

        LOGGER.info(f"Optimizer: AdamW, LR: {self.lr0}, WD: {self.weight_decay}")

    def _get_kd_loss(self, student_scores, images):
        """计算蒸馏损失"""
        if self.teacher_model is None or self.kd_alpha == 0:
            return torch.tensor(0.0, device=self.device)

        with torch.no_grad():
            teacher_output = self.teacher_model(images)
            # 提取分类分数
            if isinstance(teacher_output, (list, tuple)):
                teacher_output = teacher_output[0]

        # 应用温度缩放
        student_logits = student_scores / self.kd_temp
        teacher_logits = teacher_output / self.kd_temp

        # 计算KL散度
        kd_loss = self.kd_criterion(student_logits, teacher_logits)
        kd_loss = kd_loss * (self.kd_temp ** 2)

        return kd_loss

    def train_one_epoch(self, epoch: int) -> dict:
        """训练一个epoch"""
        self.model.train()
        pbar = TQDM(self.train_loader, desc=f"Epoch {epoch+1}/{self.epochs}")

        total_loss = 0
        total_box_loss = 0
        total_cls_loss = 0
        total_dfl_loss = 0
        total_kd_loss = 0

        for batch_idx, batch in enumerate(pbar):
            images = batch["img"].to(self.device)
            targets = batch["bboxes"].to(self.device)
            batch_idx = batch["batch_idx"].to(self.device)
            cls = batch["cls"].to(self.device)

            # 混合精度
            with autocast(self.amp):
                # 前向传播
                results = self.model(images)
                loss, loss_items = results[0], results[1]

                # 蒸馏损失
                kd_loss = torch.tensor(0.0, device=self.device)
                if self.teacher_model is not None:
                    # 提取分类分数
                    student_scores = self._extract_class_scores(results)
                    kd_loss = self._get_kd_loss(student_scores, images)

                # 总损失
                total_batch_loss = loss + self.kd_alpha * kd_loss

            # 反向传播
            self.scaler.scale(total_batch_loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()

            # 统计
            total_loss += loss.item()
            total_kd_loss += kd_loss.item() if isinstance(kd_loss, torch.Tensor) else kd_loss

            # 更新进度条
            loss_dict = {"loss": loss.item(), "kd": kd_loss.item() if isinstance(kd_loss, torch.Tensor) else 0}
            pbar.set_postfix(**{k: f"{v:.4f}" for k, v in loss_dict.items()})

        # 学习率调度
        self.scheduler.step()

        avg_loss = total_loss / len(self.train_loader)
        avg_kd_loss = total_kd_loss / len(self.train_loader)

        return {"loss": avg_loss, "kd_loss": avg_kd_loss}

    def _extract_class_scores(self, results):
        """从模型输出中提取分类分数"""
        # YOLO 输出结构: [loss, loss_items, predictions]
        # predictions 是列表，每个元素是一个检测头的输出
        if len(results) >= 3:
            predictions = results[2]
            if isinstance(predictions, list):
                # 取第一个检测头的分类输出
                m = self.model.model[-1]
                pred_distri, pred_scores = torch.cat(
                    [p.view(p.shape[0], m.no, -1) for p in predictions], 2
                ).split((m.reg_max * 4, m.nc), 1)
                return pred_scores
        return None

    @torch.no_grad()
    def validate(self) -> dict:
        """验证"""
        if not self.val:
            return {}

        self.model.eval()
        metrics = self.model.val(
            data=self.data_path,
            batch=self.batch_size,
            imgsz=self.img_size,
            device=self.device,
            save_json=False,
        )

        return metrics

    def save_model(self, epoch: int, metrics: dict | None = None):
        """保存模型"""
        save_path = Path(self.save_dir) / self.name / f"epoch_{epoch+1}.pt"
        save_path.parent.mkdir(parents=True, exist_ok=True)

        self.model.save(save_path)
        LOGGER.info(f"Saved model to {save_path}")

    def train(self):
        """开始训练"""
        LOGGER.info("=" * 60)
        LOGGER.info(f"Starting training for {self.epochs} epochs...")
        LOGGER.info("=" * 60)

        # 初始化
        self.setup_model()
        self.setup_dataloader()
        self.setup_optimizer()

        best_mAP = 0
        history = []

        for epoch in range(self.epochs):
            # 训练
            train_metrics = self.train_one_epoch(epoch)

            # 验证
            if self.val and (epoch + 1) % 10 == 0:
                val_metrics = self.validate()
                mAP50 = val_metrics.box.map50 if hasattr(val_metrics, 'box') else 0
                mAP = val_metrics.box.map if hasattr(val_metrics, 'box') else 0

                LOGGER.info(f"Validation: mAP50={mAP50:.4f}, mAP={mAP:.4f}")

                if mAP > best_mAP:
                    best_mAP = mAP
                    self.save_model(epoch, val_metrics)
            elif self.save_period > 0 and (epoch + 1) % self.save_period == 0:
                self.save_model(epoch)

            # 记录历史
            history.append({
                'epoch': epoch + 1,
                'train_loss': train_metrics['loss'],
                'kd_loss': train_metrics['kd_loss'],
            })

            LOGGER.info(f"Epoch {epoch+1}/{self.epochs} - Loss: {train_metrics['loss']:.4f}")

        LOGGER.info("=" * 60)
        LOGGER.info("Training complete!")

        # 保存最终模型
        self.save_model(self.epochs - 1)

        return history


def parse_args():
    """解析命令行参数"""
    import argparse

    parser = argparse.ArgumentParser(description='ReActNet-YOLO Training')

    # 基本参数
    parser.add_argument('--model', type=str, default='yolov8n-react.yaml', help='模型配置文件')
    parser.add_argument('--data', type=str, required=True, help='数据集配置文件')
    parser.add_argument('--epochs', type=int, default=150, help='训练轮数')
    parser.add_argument('--batch', type=int, default=32, help='批次大小')
    parser.add_argument('--imgsz', type=int, default=640, help='图像尺寸')

    # 学习率参数
    parser.add_argument('--lr0', type=float, default=0.001, help='初始学习率')
    parser.add_argument('--lrf', type=float, default=0.01, help='最终学习率比例')
    parser.add_argument('--momentum', type=float, default=0.9, help='动量')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='权重衰减')
    parser.add_argument('--cos_lr', action='store_true', help='使用余弦学习率')
    parser.add_argument('--warmup_epochs', type=float, default=2.0, help='热身轮数')

    # 蒸馏参数
    parser.add_argument('--teacher', type=str, default=None, help='教师模型路径')
    parser.add_argument('--kd_alpha', type=float, default=0.5, help='蒸馏损失权重')
    parser.add_argument('--kd_temp', type=float, default=2.0, help='蒸馏温度')

    # 其他参数
    parser.add_argument('--device', type=str, default='0', help='设备')
    parser.add_argument('--workers', type=int, default=8, help='数据加载线程数')
    parser.add_argument('--project', type=str, default='runs/train', help='项目名称')
    parser.add_argument('--name', type=str, default='exp', help='实验名称')
    parser.add_argument('--val', action='store_true', default=True, help='验证')
    parser.add_argument('--save_period', type=int, default=-1, help='保存周期')
    parser.add_argument('--amp', action='store_true', default=True, help='混合精度')
    parser.add_argument('--cfg', type=str, default=None, help='配置文件路径')

    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()

    # 构建配置字典
    overrides = {
        'model': args.model,
        'data': args.data,
        'epochs': args.epochs,
        'batch': args.batch,
        'imgsz': args.imgsz,
        'lr0': args.lr0,
        'lrf': args.lrf,
        'momentum': args.momentum,
        'weight_decay': args.weight_decay,
        'cos_lr': args.cos_lr,
        'warmup_epochs': args.warmup_epochs,
        'teacher': args.teacher,
        'kd_alpha': args.kd_alpha,
        'kd_temp': args.kd_temp,
        'device': args.device,
        'workers': args.workers,
        'project': args.project,
        'name': args.name,
        'val': args.val,
        'save_period': args.save_period,
        'amp': args.amp,
    }

    # 如果指定了配置文件，从yaml读取并合并
    if args.cfg:
        from ultralytics.cfg import get_cfg
        cfg_args = get_cfg(args.cfg, overrides)
        # 将配置对象转为字典
        cfg_dict = {k: v for k, v in vars(cfg_args).items() if not k.startswith('_')}
        # 合并到 overrides（命令行参数优先级更高）
        cfg_dict.update(overrides)
        trainer = KDTrainer(overrides=cfg_dict)
    else:
        trainer = KDTrainer(overrides=overrides)

    trainer.train()


if __name__ == "__main__":
    main()

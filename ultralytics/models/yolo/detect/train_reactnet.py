# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""
ReActNet-YOLO 训练脚本（支持知识蒸馏和多卡训练）

使用方法:
    # 单卡训练
    python main.py --model yolov8n-react.yaml --data coco.yaml --epochs 150
    # 多卡训练 (4张卡)
    torchrun --nproc_per_node=4 main.py --model yolov8n-react.yaml --data coco.yaml --epochs 150 --batch 64
"""

from __future__ import annotations

import os
import math
import random
from pathlib import Path
from typing import Any

# 禁用自动下载警告
os.environ['AUTODOWNLOAD'] = '0'

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.cuda.amp import autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from ultralytics import YOLO
from ultralytics.data import build_yolo_dataset
from ultralytics.nn.modules import Conv
from ultralytics.utils import LOGGER, TQDM, colorstr
from ultralytics.utils.loss import DistributionLoss
from ultralytics.utils.plotting import plot_results
from ultralytics.utils.torch_utils import unwrap_model
from ultralytics.utils import YAML


def init_distributed(local_rank: int):
    """初始化分布式训练"""
    if local_rank == -1:
        return False

    # 检查是否已经初始化（避免重复初始化）
    if dist.is_initialized():
        return True

    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl', init_method='env://')
    return True


def cleanup():
    """清理分布式环境"""
    if dist.is_initialized():
        dist.destroy_process_group()


class KDTrainer:
    """
    知识蒸馏训练器 for ReActNet-YOLO

    特点:
    1. 支持二值化激活训练 (BinaryConv/BinaryC2f)
    2. 可选知识蒸馏 (使用全精度教师模型)
    3. 支持多卡分布式训练 (DDP)
    4. 集成 ReActNet 训练超参数
    """

    def __init__(self, overrides: dict | None = None, local_rank: int = -1):
        """
        初始化训练器

        Args:
            cfg: 配置文件路径或字典
            overrides: 参数覆盖字典
            local_rank: 本地进程排名（用于DDP）
        """
        self.local_rank = local_rank
        self.rank = local_rank
        self.world_size = 1

        # 初始化分布式
        self.distributed = init_distributed(local_rank)
        if self.distributed:
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
            if self.rank == 0:
                LOGGER.info(f"Distributed training enabled: {self.world_size} GPUs")

        # 直接从 overrides 中提取参数，不使用 get_cfg 验证
        self.model_path = overrides.get('model', 'yolov8n-react.yaml')
        self.data_path = overrides.get('data', '')
        self.epochs = overrides.get('epochs', 150)
        self.batch_size = overrides.get('batch', 32)
        self.img_size = overrides.get('imgsz', 640)

        # 正确处理设备字符串
        device = overrides.get('device', '0')
        if device == '0':
            device = f'cuda:{local_rank}' if local_rank >= 0 else ('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.device = device

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
        self.workers = overrides.get('workers', 8)

        # 加载数据配置（用于 build_yolo_dataset）- 直接使用 YAML 读取为字典
        self.data = YAML.load(self.data_path)

        # 创建 args 对象供 build_yolo_dataset 使用，包含所有必要的超参数
        # 使用 get_cfg 加载默认配置并合并
        from ultralytics.cfg import get_cfg, DEFAULT_CFG_DICT
        # 创建自定义配置，覆盖默认的 data 路径（避免下载 coco8）
        custom_cfg = dict(DEFAULT_CFG_DICT)
        custom_cfg['data'] = self.data_path
        self.args = get_cfg(custom_cfg, {
            'imgsz': self.img_size,
            'batch': self.batch_size,
            'device': self.device,
            'workers': self.workers,
            'amp': self.amp,
            'fraction': 1.0,
            'task': 'detect',
        })
        # 添加训练必需的属性
        self.args.stride = 32
        self.args.rect = False
        self.args.cache = None
        self.args.single_cls = False
        self.args.mode = 'cache'
        self.args.classes = None

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

        if self.rank == 0:
            LOGGER.info(f"{colorstr('bold', 'ReActNet-YOLO KDTrainer')}")
            LOGGER.info(f"  - Model: {self.model_path}")
            LOGGER.info(f"  - Data: {self.data_path}")
            LOGGER.info(f"  - Epochs: {self.epochs}")
            LOGGER.info(f"  - Batch: {self.batch_size} ({self.batch_size // max(1, self.world_size)} per GPU)")
            LOGGER.info(f"  - LR: {self.lr0}")
            if self.teacher_path:
                LOGGER.info(f"  - Teacher: {self.teacher_path}")
                LOGGER.info(f"  - KD Alpha: {self.kd_alpha}, Temp: {self.kd_temp}")

    def setup_model(self):
        """初始化模型"""
        if self.rank == 0:
            LOGGER.info("-" * 60)
            LOGGER.info("Setting up model...")

        # 确保分布式进程组已初始化
        self._ensure_distributed_init()

        # 加载学生模型
        self.model = YOLO(self.model_path)
        self.model.to(self.device)

        # 多卡分布式包装
        if self.distributed:
            self.model = DDP(self.model, device_ids=[self.local_rank], output_device=self.local_rank)

        # 加载教师模型（仅主进程）
        if self.teacher_path and self.rank == 0:
            self._setup_teacher_model()

        if self.rank == 0:
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
        if self.rank == 0:
            LOGGER.info("-" * 60)
            LOGGER.info("Setting up dataloaders...")

        # 确保分布式进程组已初始化（用于DistributedSampler）
        self._ensure_distributed_init()

        # 从数据配置中获取训练和验证路径
        train_img_path = self.data.get('train', '')
        val_img_path = self.data.get('val', '')

        # 训练数据集
        train_dataset = build_yolo_dataset(
            self.args, train_img_path, self.batch_size, self.data, mode='train', stride=32
        )

        # 使用分布式采样器
        sampler = DistributedSampler(
            train_dataset,
            num_replicas=self.world_size,
            rank=self.rank,
            shuffle=True,
            drop_last=True,
            seed=0,
        ) if self.distributed else None

        # 计算实际 batch size per GPU
        batch_per_gpu = self.batch_size // max(1, self.world_size)

        self.train_loader = DataLoader(
            train_dataset,
            batch_size=batch_per_gpu,
            shuffle=(sampler is None),
            sampler=sampler,
            num_workers=self.args.workers,
            pin_memory=True,
            collate_fn=train_dataset.collate_fn,
            drop_last=True,
        )

        # 验证数据集（仅主进程）
        if self.val and val_img_path and self.rank == 0:
            val_dataset = build_yolo_dataset(
                self.args, val_img_path, self.batch_size, self.data, mode='val', stride=32
            )
            self.val_loader = DataLoader(
                val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.args.workers,
                pin_memory=True,
                collate_fn=val_dataset.collate_fn,
            )

        if self.rank == 0:
            LOGGER.info(f"Train batches: {len(self.train_loader)}")

    def setup_optimizer(self):
        """设置优化器和学习率调度器"""
        if self.rank == 0:
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

        if self.rank == 0:
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

    def _ensure_distributed_init(self):
        """确保分布式进程组已初始化"""
        if self.distributed and not dist.is_initialized():
            if self.rank == 0:
                LOGGER.info("Initializing distributed process group...")
            torch.cuda.set_device(self.local_rank)
            dist.init_process_group(backend='nccl', init_method='env://')
            # 同步所有进程
            dist.barrier()

    def train_one_epoch(self, epoch: int) -> dict:
        """训练一个epoch"""
        # 确保分布式进程组已初始化（YOLO内部trainer可能会用到）
        self._ensure_distributed_init()
        self.model.train()

        # 设置分布式采样器 epoch
        if hasattr(self.train_loader, 'sampler'):
            self.train_loader.sampler.set_epoch(epoch)

        pbar = TQDM(self.train_loader, desc=f"Epoch {epoch+1}/{self.epochs}") if self.rank == 0 else None

        total_loss = 0
        total_kd_loss = 0
        num_batches = 0

        for batch_idx, batch in enumerate(self.train_loader):
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
                if self.teacher_model is not None and self.rank == 0:
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
            num_batches += 1

            # 更新进度条（仅主进程）
            if pbar is not None:
                loss_dict = {"loss": loss.item(), "kd": kd_loss.item() if isinstance(kd_loss, torch.Tensor) else 0}
                pbar.set_postfix(**{k: f"{v:.4f}" for k, v in loss_dict.items()})

        # 同步所有进程的损失
        if self.distributed:
            total_loss = torch.tensor(total_loss, device=self.device)
            total_kd_loss = torch.tensor(total_kd_loss, device=self.device)
            num_batches = torch.tensor(num_batches, device=self.device)

            dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
            dist.all_reduce(total_kd_loss, op=dist.ReduceOp.SUM)
            dist.all_reduce(num_batches, op=dist.ReduceOp.SUM)

            total_loss = total_loss.item() / self.world_size
            total_kd_loss = total_kd_loss.item() / self.world_size
            num_batches = int(num_batches.item() / self.world_size)

        # 学习率调度
        self.scheduler.step()

        avg_loss = total_loss / num_batches
        avg_kd_loss = total_kd_loss / num_batches

        return {"loss": avg_loss, "kd_loss": avg_kd_loss}

    def _extract_class_scores(self, results):
        """从模型输出中提取分类分数"""
        # YOLO 输出结构: [loss, loss_items, predictions]
        # predictions 是列表，每个元素是一个检测头的输出
        if len(results) >= 3:
            predictions = results[2]
            if isinstance(predictions, list):
                # 取第一个检测头的分类输出
                m = self.model.module.model[-1] if hasattr(self.model, 'module') else self.model.model[-1]
                pred_distri, pred_scores = torch.cat(
                    [p.view(p.shape[0], m.no, -1) for p in predictions], 2
                ).split((m.reg_max * 4, m.nc), 1)
                return pred_scores
        return None

    @torch.no_grad()
    def validate(self) -> dict:
        """验证"""
        if not self.val or self.rank != 0:
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
        """保存模型（仅主进程）"""
        if self.rank != 0:
            return

        save_path = Path(self.save_dir) / self.name / f"epoch_{epoch+1}.pt"
        save_path.parent.mkdir(parents=True, exist_ok=True)

        # 保存非 DDP 模型
        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
        model_to_save.save(save_path)
        LOGGER.info(f"Saved model to {save_path}")

    def train(self):
        """开始训练"""
        if self.rank == 0:
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

            # 验证（仅主进程）
            if self.val and (epoch + 1) % 10 == 0:
                val_metrics = self.validate()
                if val_metrics:
                    mAP50 = val_metrics.box.map50 if hasattr(val_metrics, 'box') else 0
                    mAP = val_metrics.box.map if hasattr(val_metrics, 'box') else 0

                    if self.rank == 0:
                        LOGGER.info(f"Validation: mAP50={mAP50:.4f}, mAP={mAP:.4f}")

                    if mAP > best_mAP:
                        best_mAP = mAP
                        self.save_model(epoch, val_metrics)
            elif self.save_period > 0 and (epoch + 1) % self.save_period == 0:
                self.save_model(epoch)

            # 记录历史（仅主进程）
            if self.rank == 0:
                history.append({
                    'epoch': epoch + 1,
                    'train_loss': train_metrics['loss'],
                    'kd_loss': train_metrics['kd_loss'],
                })
                LOGGER.info(f"Epoch {epoch+1}/{self.epochs} - Loss: {train_metrics['loss']:.4f}")

        if self.rank == 0:
            LOGGER.info("=" * 60)
            LOGGER.info("Training complete!")

            # 保存最终模型
            self.save_model(self.epochs - 1)

        # 清理分布式环境
        cleanup()

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

    # 获取本地进程排名
    local_rank = int(os.environ.get('LOCAL_RANK', -1))

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
        trainer = KDTrainer(overrides=cfg_dict, local_rank=local_rank)
    else:
        trainer = KDTrainer(overrides=overrides, local_rank=local_rank)

    trainer.train()


if __name__ == "__main__":
    main()

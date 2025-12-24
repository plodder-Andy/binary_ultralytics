# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""
ReActNet-YOLO 训练脚本 - 修复版 (Ver2)

基于 DetectionTrainer 实现知识蒸馏训练
使用响应蒸馏：在检测头输出层面进行知识蒸馏
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from typing import Any

from ultralytics.models.yolo.detect.train import DetectionTrainer
from ultralytics.utils import DEFAULT_CFG_DICT, LOGGER, RANK
from ultralytics.utils.loss import v8DetectionLoss
from ultralytics.utils.torch_utils import unwrap_model


class v8DetectionLossWithKD(v8DetectionLoss):
    """带知识蒸馏的 YOLOv8 检测损失类。

    在原始检测损失基础上添加响应蒸馏损失，
    通过对齐学生和教师模型的分类分数来实现知识迁移。
    """

    def __init__(self, model, tal_topk: int = 10, teacher_model=None, kd_alpha: float = 0.5, kd_temp: float = 2.0):
        """初始化带 KD 的检测损失。

        Args:
            model: 学生模型
            tal_topk: Task-Aligned Assigner 的 topk 参数
            teacher_model: 教师模型（用于蒸馏）
            kd_alpha: 蒸馏损失权重
            kd_temp: 蒸馏温度
        """
        super().__init__(model, tal_topk)
        self.teacher_model = teacher_model
        self.kd_alpha = kd_alpha
        self.kd_temp = kd_temp
        self.kd_loss_fn = nn.KLDivLoss(reduction='batchmean')

    def set_teacher(self, teacher_model):
        """设置教师模型。"""
        self.teacher_model = teacher_model

    def __call__(self, preds: Any, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """计算检测损失 + 蒸馏损失。"""
        # 先计算原始检测损失
        det_loss, loss_items = super().__call__(preds, batch)

        # 如果没有教师模型，直接返回原始损失
        if self.teacher_model is None:
            return det_loss, loss_items

        # 计算蒸馏损失
        kd_loss = self._compute_kd_loss(preds, batch)

        # 总损失 = 检测损失 + alpha * 蒸馏损失
        total_loss = det_loss + self.kd_alpha * kd_loss

        # 更新 loss_items（添加 kd_loss 用于日志）
        # loss_items 是 [box, cls, dfl]，我们不修改它以保持兼容性
        # kd_loss 会在训练循环中单独记录

        return total_loss, loss_items

    def _compute_kd_loss(self, preds: Any, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """计算响应蒸馏损失。

        在检测头的分类分数层面进行蒸馏。
        """
        # 提取学生模型的特征
        feats = preds[1] if isinstance(preds, tuple) else preds

        # 获取学生的分类分数
        student_pred_scores = torch.cat(
            [xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2
        ).split((self.reg_max * 4, self.nc), 1)[1]  # 只取分类分数部分

        # 获取教师模型的预测
        with torch.no_grad():
            teacher_preds = self.teacher_model(batch["img"])
            teacher_feats = teacher_preds[1] if isinstance(teacher_preds, tuple) else teacher_preds

            # 确保教师模型输出格式正确
            if isinstance(teacher_feats, (list, tuple)) and len(teacher_feats) > 0:
                # 获取教师的分类分数
                teacher_no = self.no  # 假设教师和学生有相同的输出通道数
                try:
                    teacher_pred_scores = torch.cat(
                        [xi.view(teacher_feats[0].shape[0], teacher_no, -1) for xi in teacher_feats], 2
                    ).split((self.reg_max * 4, self.nc), 1)[1]
                except Exception:
                    # 如果教师模型结构不同，尝试其他方式
                    return torch.tensor(0.0, device=self.device)
            else:
                return torch.tensor(0.0, device=self.device)

        # 确保形状匹配
        if student_pred_scores.shape != teacher_pred_scores.shape:
            # 如果空间维度不同，进行插值对齐
            if student_pred_scores.shape[-1] != teacher_pred_scores.shape[-1]:
                min_len = min(student_pred_scores.shape[-1], teacher_pred_scores.shape[-1])
                student_pred_scores = student_pred_scores[..., :min_len]
                teacher_pred_scores = teacher_pred_scores[..., :min_len]

        # 计算 KL 散度损失
        # 对分类分数应用温度缩放
        student_logits = student_pred_scores / self.kd_temp
        teacher_logits = teacher_pred_scores / self.kd_temp

        # Softmax over classes (dim=1)
        student_log_probs = F.log_softmax(student_logits, dim=1)
        teacher_probs = F.softmax(teacher_logits, dim=1)

        # KL 散度损失
        kd_loss = self.kd_loss_fn(student_log_probs, teacher_probs) * (self.kd_temp ** 2)

        return kd_loss


class ReActNetTrainer(DetectionTrainer):
    """ReActNet-YOLO 训练器，支持知识蒸馏。

    继承 DetectionTrainer，复用所有检测训练功能，
    通过自定义损失类实现响应蒸馏。
    """

    def __init__(self, cfg=None, overrides: dict | None = None, _callbacks=None):
        """初始化训练器。"""
        if overrides is None:
            overrides = {}
        else:
            overrides = overrides.copy()  # 创建副本，避免修改原始字典

        # 提取自定义 KD 参数，避免被 check_dict_alignment 检查
        self._kd_temp = overrides.pop('kd_temp', None)
        self._kd_alpha = overrides.pop('kd_alpha', None)
        self._teacher_path = overrides.pop('teacher', None)

        # 加载 reactnet_hyp.yaml 中的自定义超参数并合并到 overrides
        reactnet_hyp_path = Path(__file__).parent.parent.parent.parent / "cfg" / "reactnet_hyp.yaml"
        if reactnet_hyp_path.exists():
            from ultralytics.utils import YAML
            reactnet_cfg = YAML.load(reactnet_hyp_path)
            # 将 reactnet_hyp.yaml 中的配置作为默认值，overrides 优先
            for k, v in reactnet_cfg.items():
                if k not in overrides and v is not None:
                    # 跳过 KD 相关参数，因为已经单独处理了
                    if k not in ('teacher', 'kd_alpha', 'kd_temp'):
                        overrides[k] = v

        # 使用 DEFAULT_CFG_DICT 作为基础配置（包含所有必需参数）
        super().__init__(cfg=DEFAULT_CFG_DICT, overrides=overrides, _callbacks=_callbacks)

        # 恢复自定义参数到 args
        self.args.kd_temp = self._kd_temp if self._kd_temp is not None else 2.0
        self.args.kd_alpha = self._kd_alpha if self._kd_alpha is not None else 0.5
        self.args.teacher = self._teacher_path

        # 蒸馏相关属性
        self.kd_temp = self.args.kd_temp
        self.kd_alpha = self.args.kd_alpha
        self.teacher_model = None

        # 检测 torchrun 启动的分布式环境，修正 world_size
        import os
        if RANK != -1 and self.world_size == 1:
            # torchrun 启动，但 world_size 未正确设置
            world_size_env = os.getenv("WORLD_SIZE")
            if world_size_env:
                self.world_size = int(world_size_env)
                LOGGER.info(f"检测到 torchrun 分布式环境: RANK={RANK}, WORLD_SIZE={self.world_size}")

    def get_model(self, cfg=None, weights=None, verbose=True):
        """返回检测模型并加载教师模型。"""
        model = super().get_model(cfg, weights, verbose)

        # 加载教师模型
        teacher_path = getattr(self.args, 'teacher', None)
        if teacher_path:
            from ultralytics import YOLO
            LOGGER.info(f"加载教师模型: {teacher_path}")
            teacher_yolo = YOLO(teacher_path)
            self.teacher_model = teacher_yolo.model
            self.teacher_model.to(self.device)
            self.teacher_model.eval()

            # 冻结教师模型
            for param in self.teacher_model.parameters():
                param.requires_grad = False
            LOGGER.info(f"教师模型已加载并冻结 (kd_alpha={self.kd_alpha}, kd_temp={self.kd_temp})")

        return model

    def get_loss(self, model):
        """返回带知识蒸馏的损失函数。"""
        return v8DetectionLossWithKD(
            model=model,
            teacher_model=self.teacher_model,
            kd_alpha=self.kd_alpha,
            kd_temp=self.kd_temp,
        )

    def _setup_train(self):
        """设置训练，使用自定义损失函数。"""
        super()._setup_train()

        # 替换损失函数为带 KD 的版本
        if self.teacher_model is not None:
            model = unwrap_model(self.model)
            self.loss = self.get_loss(model)
            # 确保教师模型在正确的设备上
            self.teacher_model.to(self.device)
            LOGGER.info("已启用知识蒸馏损失")

    def get_validator(self):
        """返回验证器。"""
        self.loss_names = "box_loss", "cls_loss", "dfl_loss"
        return super().get_validator()

    def _setup_ddp(self):
        """初始化分布式训练参数，修复 LOCAL_RANK 问题。"""
        import os
        from datetime import timedelta
        from torch import distributed as dist

        # 直接从环境变量读取，而不是使用模块导入时的值
        local_rank = int(os.getenv("LOCAL_RANK", 0))
        rank = int(os.getenv("RANK", 0))
        world_size = int(os.getenv("WORLD_SIZE", 1))

        LOGGER.info(f"DDP 初始化: LOCAL_RANK={local_rank}, RANK={rank}, WORLD_SIZE={world_size}")

        # 使用 LOCAL_RANK 来设置 CUDA 设备
        torch.cuda.set_device(local_rank)
        self.device = torch.device("cuda", local_rank)
        os.environ["TORCH_NCCL_BLOCKING_WAIT"] = "1"
        dist.init_process_group(
            backend="nccl" if dist.is_nccl_available() else "gloo",
            timeout=timedelta(seconds=10800),  # 3 hours
            rank=rank,
            world_size=world_size,
        )


# ========== 便捷使用接口 ==========

def train(
    model: str = "yolov8s-react.yaml",
    data: str = "coco.yaml",
    epochs: int = 150,
    teacher: str | None = None,
    kd_alpha: float = 0.5,
    kd_temp: float = 2.0,
    batch: int = 16,
    device: str = "0",
    **kwargs
):
    """便捷训练入口。

    使用示例:
        # 普通训练
        train(model="yolov8s-react.yaml", data="coco.yaml")

        # 蒸馏训练
        train(model="yolov8s-react.yaml", data="coco.yaml",
              teacher="yolov8s.pt", kd_alpha=0.5)
    """
    overrides = {
        'model': model,
        'data': data,
        'epochs': epochs,
        'batch': batch,
        'device': device,
        'teacher': teacher,
        'kd_alpha': kd_alpha,
        'kd_temp': kd_temp,
        **kwargs
    }

    trainer = ReActNetTrainer(overrides=overrides)
    trainer.train()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='ReActNet-YOLO 训练')
    parser.add_argument('--model', type=str, default='yolov8s-react.yaml', help='模型配置')
    parser.add_argument('--data', type=str, required=True, help='数据集配置')
    parser.add_argument('--epochs', type=int, default=150, help='训练轮数')
    parser.add_argument('--batch', type=int, default=16, help='批次大小')
    parser.add_argument('--teacher', type=str, default=None, help='教师模型路径')
    parser.add_argument('--kd_alpha', type=float, default=0.5, help='蒸馏权重')
    parser.add_argument('--kd_temp', type=float, default=2.0, help='蒸馏温度')
    parser.add_argument('--device', type=str, default='0', help='设备')

    args = parser.parse_args()

    train(
        model=args.model,
        data=args.data,
        epochs=args.epochs,
        batch=args.batch,
        teacher=args.teacher,
        kd_alpha=args.kd_alpha,
        kd_temp=args.kd_temp,
        device=args.device,
    )

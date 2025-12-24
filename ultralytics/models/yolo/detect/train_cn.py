# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import math
import random
from copy import copy
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from ultralytics.data import build_dataloader, build_yolo_dataset
from ultralytics.engine.trainer import BaseTrainer
from ultralytics.models import yolo
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils import DEFAULT_CFG, LOGGER, RANK
from ultralytics.utils.patches import override_configs
from ultralytics.utils.plotting import plot_images, plot_labels
from ultralytics.utils.torch_utils import torch_distributed_zero_first, unwrap_model


class DetectionTrainer(BaseTrainer):
    """基于检测模型训练器的扩展类。

    该训练器专门用于目标检测任务，处理训练YOLO模型进行目标检测的特定需求，
    包括数据集构建、数据加载、预处理和模型配置。

    属性:
        model (DetectionModel): 正在训练的YOLO检测模型。
        data (dict): 包含数据集信息的字典，包括类别名称和类别数量。
        loss_names (tuple): 训练中使用的损失组件名称 (box_loss, cls_loss, dfl_loss)。

    方法:
        build_dataset: 为训练或验证构建YOLO数据集。
        get_dataloader: 构造并返回指定模式的数据加载器。
        preprocess_batch: 预处理一批图像，包括缩放和转换为浮点类型。
        set_model_attributes: 根据数据集信息设置模型属性。
        get_model: 返回YOLO检测模型。
        get_validator: 返回用于模型评估的验证器。
        label_loss_items: 返回带有标签的训练损失项字典。
        progress_string: 返回格式化的训练进度字符串。
        plot_training_samples: 绘制带有标注的训练样本。
        plot_training_labels: 创建YOLO模型的带标签训练图。
        auto_batch: 根据模型内存需求计算最优批次大小。

    示例:
        >>> from ultralytics.models.yolo.detect import DetectionTrainer
        >>> args = dict(model="yolo11n.pt", data="coco8.yaml", epochs=3)
        >>> trainer = DetectionTrainer(overrides=args)
        >>> trainer.train()
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides: dict[str, Any] | None = None, _callbacks=None):
        """初始化用于训练YOLO目标检测模型的DetectionTrainer对象。

        参数:
            cfg (dict, optional): 包含训练参数的默认配置字典。
            overrides (dict, optional): 默认配置的参数字典覆盖。
            _callbacks (list, optional): 在训练期间执行的回调函数列表。
        """
        super().__init__(cfg, overrides, _callbacks)

    def build_dataset(self, img_path: str, mode: str = "train", batch: int | None = None):
        """为训练或验证构建YOLO数据集。

        参数:
            img_path (str): 包含图像的文件夹路径。
            mode (str): 'train'模式或'val'模式，用户可以为每种模式自定义不同的增强。
            batch (int, optional): 批次大小，这用于'rect'模式。

        返回:
            (Dataset): 为指定模式配置的YOLO数据集对象。
        """
        gs = max(int(unwrap_model(self.model).stride.max() if self.model else 0), 32)
        return build_yolo_dataset(self.args, img_path, batch, self.data, mode=mode, rect=mode == "val", stride=gs)

    def get_dataloader(self, dataset_path: str, batch_size: int = 16, rank: int = 0, mode: str = "train"):
        """构造并返回指定模式的数据加载器。

        参数:
            dataset_path (str): 数据集路径。
            batch_size (int): 每批图像数量。
            rank (int): 分布式训练进程排名。
            mode (str): 'train'为训练数据加载器，'val'为验证数据加载器。

        返回:
            (DataLoader): PyTorch数据加载器对象。
        """
        assert mode in {"train", "val"}, f"模式必须为'train'或'val'，而不是{mode}。"
        with torch_distributed_zero_first(rank):  # 仅在DDP情况下初始化数据集*.cache一次
            dataset = self.build_dataset(dataset_path, mode, batch_size)
        shuffle = mode == "train"
        if getattr(dataset, "rect", False) and shuffle:
            LOGGER.warning("'rect=True'与DataLoader的shuffle不兼容，已设置shuffle=False")
            shuffle = False
        return build_dataloader(
            dataset,
            batch=batch_size,
            workers=self.args.workers if mode == "train" else self.args.workers * 2,
            shuffle=shuffle,
            rank=rank,
            drop_last=self.args.compile and mode == "train",
        )

    def preprocess_batch(self, batch: dict) -> dict:
        """预处理一批图像，包括缩放和转换为浮点类型。

        参数:
            batch (dict): 包含批次数据的字典，其中包含'img'张量。

        返回:
            (dict): 归一化图像的预处理批次。
        """
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(self.device, non_blocking=self.device.type == "cuda")
        batch["img"] = batch["img"].float() / 255
        if self.args.multi_scale:
            imgs = batch["img"]
            sz = (
                random.randrange(int(self.args.imgsz * 0.5), int(self.args.imgsz * 1.5 + self.stride))
                // self.stride
                * self.stride
            )  # 尺寸
            sf = sz / max(imgs.shape[2:])  # 缩放因子
            if sf != 1:
                ns = [
                    math.ceil(x * sf / self.stride) * self.stride for x in imgs.shape[2:]
                ]  # 新形状（拉伸到gs倍数）
                imgs = nn.functional.interpolate(imgs, size=ns, mode="bilinear", align_corners=False)
            batch["img"] = imgs
        return batch

    def set_model_attributes(self):
        """根据数据集信息设置模型属性。"""
        # Nl = de_parallel(self.model).model[-1].nl  # 检测层数量（用于缩放超参数）
        # self.args.box *= 3 / nl  # 缩放到层数
        # self.args.cls *= self.data["nc"] / 80 * 3 / nl  # 缩放到类别和层数
        # self.args.cls *= (self.args.imgsz / 640) ** 2 * 3 / nl  # 缩放到图像大小和层数
        self.model.nc = self.data["nc"]  # 将类别数量附加到模型
        self.model.names = self.data["names"]  # 将类别名称附加到模型
        self.model.args = self.args  # 将超参数附加到模型
        # TODO: self.model.class_weights = labels_to_class_weights(dataset.labels, nc).to(device) * nc

    def get_model(self, cfg: str | None = None, weights: str | None = None, verbose: bool = True):
        """返回YOLO检测模型。

        参数:
            cfg (str, optional): 模型配置文件的路径。
            weights (str, optional): 模型权重的路径。
            verbose (bool): 是否显示模型信息。

        返回:
            (DetectionModel): YOLO检测模型。
        """
        model = DetectionModel(cfg, nc=self.data["nc"], ch=self.data["channels"], verbose=verbose and RANK == -1)
        if weights:
            model.load(weights)
        return model

    def get_validator(self):
        """返回用于YOLO模型验证的DetectionValidator。"""
        self.loss_names = "box_loss", "cls_loss", "dfl_loss"
        return yolo.detect.DetectionValidator(
            self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks
        )

    def label_loss_items(self, loss_items: list[float] | None = None, prefix: str = "train"):
        """返回带有标签的训练损失项张量字典。

        参数:
            loss_items (list[float], optional): 损失值列表。
            prefix (str): 返回字典中键的前缀。

        返回:
            (dict | list): 如果提供了loss_items则返回损失项字典，否则返回键列表。
        """
        keys = [f"{prefix}/{x}" for x in self.loss_names]
        if loss_items is not None:
            loss_items = [round(float(x), 5) for x in loss_items]  # 将张量转换为5位小数浮点数
            return dict(zip(keys, loss_items))
        else:
            return keys

    def progress_string(self):
        """返回格式化的训练进度字符串，包括轮次、GPU内存、损失、实例和大小。"""
        return ("\n" + "%11s" * (4 + len(self.loss_names))) % (
            "轮次",
            "GPU内存",
            *self.loss_names,
            "实例数",
            "尺寸",
        )

    def plot_training_samples(self, batch: dict[str, Any], ni: int) -> None:
        """绘制带有标注的训练样本。

        参数:
            batch (dict[str, Any]): 包含批次数据的字典。
            ni (int): 迭代次数。
        """
        plot_images(
            labels=batch,
            paths=batch["im_file"],
            fname=self.save_dir / f"train_batch{ni}.jpg",
            on_plot=self.on_plot,
        )

    def plot_training_labels(self):
        """创建YOLO模型的带标签训练图。"""
        boxes = np.concatenate([lb["bboxes"] for lb in self.train_loader.dataset.labels], 0)
        cls = np.concatenate([lb["cls"] for lb in self.train_loader.dataset.labels], 0)
        plot_labels(boxes, cls.squeeze(), names=self.data["names"], save_dir=self.save_dir, on_plot=self.on_plot)

    def auto_batch(self):
        """通过计算模型的内存占用获取最优批次大小。

        返回:
            (int): 最优批次大小。
        """
        with override_configs(self.args, overrides={"cache": False}) as self.args:
            train_dataset = self.build_dataset(self.data["train"], mode="train", batch=16)
        max_num_obj = max(len(label["cls"]) for label in train_dataset.labels) * 4  # 4用于马赛克增强
        del train_dataset  # 释放内存
        return super().auto_batch(max_num_obj)

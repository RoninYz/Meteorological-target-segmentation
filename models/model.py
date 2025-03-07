#!/usr/bin/env python
# -*- coding:utf-8 -*-
import pytorch_lightning as pl
import segmentation_models_pytorch as smp
import torch
from torch.optim import lr_scheduler


class ThiModel(pl.LightningModule):
    def __init__(self, arch, encoder_name, in_channels, out_classes, encoder_weights="imagenet", 
                 learning_rate=1e-4, batch_size=4, dataset_name=None,
                 calculate_membership='none', polynomial_dir=None, height_bands=None):
        super().__init__()
        # 保存所有参数 - 这些会被 save_hyperparameters() 方法捕获
        self.arch = arch
        self.encoder_name = encoder_name
        self.in_channels = in_channels
        self.out_classes = out_classes
        self.encoder_weights = encoder_weights
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.dataset_name = dataset_name
        
        # 添加隶属度计算相关参数
        self.calculate_membership = calculate_membership
        self.polynomial_dir = polynomial_dir
        self.height_bands = height_bands
        
        # 创建模型
        self.model = self._get_model()

        # 由于输入是5通道，我们不使用标准的图像预处理参数
        self.number_of_classes = out_classes

        # 使用Dice Loss作为损失函数
        self.loss_fn = smp.losses.DiceLoss(mode='multiclass')

        # 添加指标存储
        self.train_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []

        self.save_hyperparameters()

    def _get_model(self):
        return smp.create_model(
            self.arch,
            encoder_name=self.encoder_name,
            in_channels=self.in_channels,  # 5通道输入
            classes=self.out_classes,
            encoder_weights=self.encoder_weights,
        )

    def forward(self, x):
        # 数据预处理已经在数据集类中完成，这里直接使用模型
        return self.model(x)

    def shared_step(self, batch, stage):
        image, mask = batch

        # 确保图像维度正确
        assert image.ndim == 4  # [batch_size, channels, H, W]
        
        # 确保mask是long类型
        mask = mask.long()

        # 确保mask维度正确
        assert mask.ndim == 3  # [batch_size, H, W]

        # 预测mask
        logits_mask = self.forward(image)

        # 确保输出通道数正确
        assert logits_mask.shape[1] == self.number_of_classes

        # 确保logits是连续的
        logits_mask = logits_mask.contiguous()

        # 计算损失
        loss = self.loss_fn(logits_mask, mask)

        # 应用softmax得到概率
        prob_mask = logits_mask.softmax(dim=1)

        # 转换概率为预测类别
        pred_mask = prob_mask.argmax(dim=1)

        # 计算评估指标
        tp, fp, fn, tn = smp.metrics.get_stats(
            pred_mask, mask, mode="multiclass", num_classes=self.number_of_classes
        )

        return {
            "loss": loss,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
        }

    def training_step(self, batch, batch_idx):
        images, masks = batch
        # 确保数据类型正确
        images = images.float()
        masks = masks.long()
        
        # 前向传播
        outputs = self(images)
        loss = self.loss_fn(outputs, masks)
        
        # 计算并记录指标
        probs = outputs.softmax(dim=1)
        preds = probs.argmax(dim=1)
        
        tp, fp, fn, tn = smp.metrics.get_stats(
            preds, masks, mode='multiclass', num_classes=2
        )
        
        per_image_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction='micro-imagewise')
        dataset_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction='micro')
        
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_per_image_iou', per_image_iou, prog_bar=True)
        self.log('train_dataset_iou', dataset_iou, prog_bar=True)
        
        # 存储步骤输出
        self.train_step_outputs.append({
            'loss': loss,
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'tn': tn
        })
        
        return loss

    def validation_step(self, batch, batch_idx):
        images, masks = batch
        # 确保数据类型正确
        images = images.float()
        masks = masks.long()
        
        # 前向传播
        outputs = self(images)
        loss = self.loss_fn(outputs, masks)
        
        probs = outputs.softmax(dim=1)
        preds = probs.argmax(dim=1)
        
        tp, fp, fn, tn = smp.metrics.get_stats(
            preds, masks, mode='multiclass', num_classes=2
        )
        
        per_image_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction='micro-imagewise')
        dataset_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction='micro')
        
        self.log('valid_loss', loss, prog_bar=True)
        self.log('valid_per_image_iou', per_image_iou, prog_bar=True)
        self.log('valid_dataset_iou', dataset_iou, prog_bar=True)
        
        # 存储步骤输出
        self.validation_step_outputs.append({
            'loss': loss,
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'tn': tn
        })
        
        return loss

    def on_train_epoch_end(self):
        # 处理训练epoch结束
        self._shared_epoch_end(self.train_step_outputs, "train")
        self.train_step_outputs.clear()  # 清除存储的输出

    def on_validation_epoch_end(self):
        # 处理验证epoch结束
        self._shared_epoch_end(self.validation_step_outputs, "valid")
        self.validation_step_outputs.clear()  # 清除存储的输出

    def _shared_epoch_end(self, outputs, prefix):
        if not outputs:  # 如果输出列表为空，直接返回
            return
        
        # 继续原有的处理逻辑
        tp = torch.cat([x['tp'] for x in outputs])
        fp = torch.cat([x['fp'] for x in outputs])
        fn = torch.cat([x['fn'] for x in outputs])
        tn = torch.cat([x['tn'] for x in outputs])
        
        per_image_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction='micro-imagewise')
        dataset_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction='micro')
        
        metrics = {
            f'{prefix}_per_image_iou_epoch': per_image_iou,
            f'{prefix}_dataset_iou_epoch': dataset_iou,
        }
        self.log_dict(metrics, prog_bar=True)

    def test_step(self, batch, batch_idx):
        test_loss_info = self.shared_step(batch, "test")
        self.test_step_outputs.append(test_loss_info)
        return test_loss_info

    def on_test_epoch_end(self):
        self._shared_epoch_end(self.test_step_outputs, "test")
        self.test_step_outputs.clear()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), 
            lr=self.learning_rate,
            weight_decay=0.001
        )
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='max',
            factor=0.5,
            patience=5,
            verbose=True
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "valid_dataset_iou",
                "interval": "epoch",
            },
        } 
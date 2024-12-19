
import torchvision
import os

from ball_dataset import CocoDetection
from balloon_model import freeze_model, pt_model


        
from transformers import DetrImageProcessor

processor = DetrImageProcessor.from_pretrained("yiyiyiwufeng/detr-finetuned-balloon-v2")

train_dataset = CocoDetection(img_folder='/data1/lh/data/balloon/train', processor=processor)
val_dataset = CocoDetection(img_folder='/data1/lh/data/balloon/val', processor=processor, train=False)

# print("Number of training examples:", len(train_dataset))  # 61
# print("Number of validation examples:", len(val_dataset))  # 13

import numpy as np
import os
from PIL import Image, ImageDraw

# based on https://github.com/woctezuma/finetune-detr/blob/master/finetune_detr.ipynb
# image_ids = train_dataset.coco.getImgIds()
# # let's pick a random image
# image_id = image_ids[np.random.randint(0, len(image_ids))]
# print('Image n°{}'.format(image_id))
# image = train_dataset.coco.loadImgs(image_id)[0]
# image = Image.open(os.path.join('/data1/lh/data/balloon/train', image['file_name']))

# annotations = train_dataset.coco.imgToAnns[image_id]
# draw = ImageDraw.Draw(image, "RGBA")

cats = train_dataset.coco.cats
id2label = {k: v['name'] for k,v in cats.items()}

# for annotation in annotations:
#   box = annotation['bbox']
#   class_idx = annotation['category_id']
#   x,y,w,h = tuple(box)
#   draw.rectangle((x,y,x+w,y+h), outline='red', width=1)
#   draw.text((x, y), id2label[class_idx], fill='white')

# image.save('ball_example.jpg')


from torch.utils.data import DataLoader

def collate_fn(batch):
  pixel_values = [item[0] for item in batch]
  encoding = processor.pad(pixel_values, return_tensors="pt")
  labels = [item[1] for item in batch]
  batch = {}
  batch['pixel_values'] = encoding['pixel_values']
  batch['pixel_mask'] = encoding['pixel_mask']
  batch['labels'] = labels
  return batch

train_dataloader = DataLoader(train_dataset, collate_fn=collate_fn, batch_size=4, shuffle=True)
val_dataloader = DataLoader(val_dataset, collate_fn=collate_fn, batch_size=2)
batch = next(iter(train_dataloader))

# print(batch.keys())   # dict_keys(['pixel_values', 'pixel_mask', 'labels'])
pixel_values, target = train_dataset[0]
# print(pixel_values.shape, target)   # [torch.Size([3, 800, 1066]) {'size': tensor([ 800, 1066]), 'image_id': tensor([0]), 'class_labels': tensor([0]), 'boxes': tensor([[0.5955, 0.5811, 0.2202, 0.3561]]), 'area': tensor([3681.5083]), 'iscrowd': tensor([0]), 'orig_size': tensor([1536, 2048])}


#=================model==================
import pytorch_lightning as pl
from transformers import DetrForObjectDetection
import torch
import torch.nn as nn

class PromptedDetrModel(pl.LightningModule):
     def __init__(self, lr, lr_backbone, weight_decay):
         super().__init__()
         # replace COCO classification head with custom head
         # we specify the "no_timm" variant here to not rely on the timm library
         # for the convolutional backbone
         self.model = pt_model()
         freeze_model(self.model)
         self.lr = lr
         self.lr_backbone = lr_backbone
         self.weight_decay = weight_decay
     def forward(self, pixel_values, pixel_mask):
       outputs = self.model(pixel_values=pixel_values, pixel_mask=pixel_mask)

       return outputs

     def common_step(self, batch, batch_idx):
       pixel_values = batch["pixel_values"]
       pixel_mask = batch["pixel_mask"]
       labels = [{k: v.to(self.device) for k, v in t.items()} for t in batch["labels"]]

       outputs = self.model(pixel_values=pixel_values, pixel_mask=pixel_mask, labels=labels)

       loss = outputs.loss
       loss_dict = outputs.loss_dict

       return loss, loss_dict

     def training_step(self, batch, batch_idx):
        loss, loss_dict = self.common_step(batch, batch_idx)
        # logs metrics for each training_step,
        # and the average across the epoch
        self.log("training_loss", loss)
        for k,v in loss_dict.items():
          self.log("train_" + k, v.item())

        return loss

     def validation_step(self, batch, batch_idx):
        loss, loss_dict = self.common_step(batch, batch_idx)
        self.log("validation_loss", loss)
        for k,v in loss_dict.items():
          self.log("validation_" + k, v.item())

        return loss

     def configure_optimizers(self):
        param_dicts = [
              {"params": [p for n, p in self.named_parameters() if "backbone" not in n and p.requires_grad]},
              {
                  "params": [p for n, p in self.named_parameters() if "backbone" in n and p.requires_grad],
                  "lr": self.lr_backbone,
              },
        ]
        optimizer = torch.optim.AdamW(param_dicts, lr=self.lr,
                                  weight_decay=self.weight_decay)

        return optimizer

     def train_dataloader(self):
        return train_dataloader

     def val_dataloader(self):
        return val_dataloader

model = PromptedDetrModel(lr=1e-4, lr_backbone=1e-5, weight_decay=1e-4)

outputs = model(pixel_values=batch['pixel_values'], pixel_mask=batch['pixel_mask'])

print(outputs.logits.shape)   # torch.Size([4, 100, 2])

from pytorch_lightning import Trainer

trainer = Trainer(max_epochs=50,max_steps=300, gradient_clip_val=0.1)
trainer.fit(model)

# 加载模型从checkpoint文件，并传递初始化参数
# checkpoint_path = "lightning_logs/version_0/checkpoints/epoch=149-step=300.ckpt"
# model = Detr.load_from_checkpoint(checkpoint_path, lr=1e-4, lr_backbone=1e-5, weight_decay=1e-4)

# model.model.push_to_hub("yiyiyiwufeng/detr-prompt-finetuned-balloon-v4")
# processor.push_to_hub("yiyiyiwufeng/detr-prompt-finetuned-balloon-v4")



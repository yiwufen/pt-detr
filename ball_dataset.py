import torchvision
import os

class CocoDetection(torchvision.datasets.CocoDetection):
    def __init__(self, img_folder, processor, train=True):
        ann_file = os.path.join(img_folder, "custom_train.json" if train else "custom_val.json")
        super(CocoDetection, self).__init__(img_folder, ann_file)
        self.processor = processor

    def __getitem__(self, idx):
        # read in PIL image and target in COCO format
        # feel free to add data augmentation here before passing them to the next step
        img, target = super(CocoDetection, self).__getitem__(idx)

        # preprocess image and target (converting target to DETR format, resizing + normalization of both image and target)
        image_id = self.ids[idx]
        target = {'image_id': image_id, 'annotations': target}
        encoding = self.processor(images=img, annotations=target, return_tensors="pt")
        pixel_values = encoding["pixel_values"].squeeze() # remove batch dimension
        target = encoding["labels"][0] # remove batch dimension

        return pixel_values, target


from transformers import DetrImageProcessor

def balloon_data():
    processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")

    train_dataset = CocoDetection(img_folder='/data1/lh/data/balloon/train', processor=processor)
    val_dataset = CocoDetection(img_folder='/data1/lh/data/balloon/val', processor=processor, train=False)

    return train_dataset, val_dataset


# import numpy as np
# import os
# from PIL import Image, ImageDraw

# # based on https://github.com/woctezuma/finetune-detr/blob/master/finetune_detr.ipynb
# image_ids = train_dataset.coco.getImgIds()
# # let's pick a random image
# image_id = image_ids[np.random.randint(0, len(image_ids))]
# print('Image n°{}'.format(image_id))
# image = train_dataset.coco.loadImgs(image_id)[0]
# image = Image.open(os.path.join('/data1/lh/data/balloon/train', image['file_name']))

# annotations = train_dataset.coco.imgToAnns[image_id]
# draw = ImageDraw.Draw(image, "RGBA")

# cats = train_dataset.coco.cats
# id2label = {k: v['name'] for k,v in cats.items()}

# for annotation in annotations:
#   box = annotation['bbox']
#   class_idx = annotation['category_id']
#   x,y,w,h = tuple(box)
#   draw.rectangle((x,y,x+w,y+h), outline='red', width=1)
#   draw.text((x, y), id2label[class_idx], fill='white')

# image.save('balloon_sample.png')
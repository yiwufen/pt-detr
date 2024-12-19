# Data preparation
## COCO 2017
Download and extract COCO 2017 train and val images with annotations from http://cocodataset.org. We expect the directory structure to be the following:

```
path/to/coco/
  annotations/  # annotation json files
  train2017/    # train images
  val2017/      # val images
```

## balloon dataset
Download + preprocess data
Here we download the balloon dataset, hosted on Github.


### download, decompress the data
```
wget https://github.com/matterport/Mask_RCNN/releases/download/v2.1/balloon_dataset.zip
unzip balloon_dataset.zip
```
output:
```
--2023-12-19 14:11:08--  https://github.com/matterport/Mask_RCNN/releases/download/v2.1/balloon_dataset.zip
Resolving github.com (github.com)... 140.82.112.4
Connecting to github.com (github.com)|140.82.112.4|:443... connected.
HTTP request sent, awaiting response... 302 Found
Location: https://objects.githubusercontent.com/github-production-release-asset-2e65be/107595270/737339e2-2b83-11e8-856a-188034eb3468?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAIWNJYAX4CSVEH53A%2F20231219%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20231219T141108Z&X-Amz-Expires=300&X-Amz-Signature=089176e110aefe16a9a6a9cd7ec950f713f0f6a0ae5748a008e61a128b3e9fa1&X-Amz-SignedHeaders=host&actor_id=0&key_id=0&repo_id=107595270&response-content-disposition=attachment%3B%20filename%3Dballoon_dataset.zip&response-content-type=application%2Foctet-stream [following]
--2023-12-19 14:11:08--  https://objects.githubusercontent.com/github-production-release-asset-2e65be/107595270/737339e2-2b83-11e8-856a-188034eb3468?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAIWNJYAX4CSVEH53A%2F20231219%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20231219T141108Z&X-Amz-Expires=300&X-Amz-Signature=089176e110aefe16a9a6a9cd7ec950f713f0f6a0ae5748a008e61a128b3e9fa1&X-Amz-SignedHeaders=host&actor_id=0&key_id=0&repo_id=107595270&response-content-disposition=attachment%3B%20filename%3Dballoon_dataset.zip&response-content-type=application%2Foctet-stream
Resolving objects.githubusercontent.com (objects.githubusercontent.com)... 185.199.108.133, 185.199.111.133, 185.199.109.133, ...
Connecting to objects.githubusercontent.com (objects.githubusercontent.com)|185.199.108.133|:443... connected.
HTTP request sent, awaiting response... 200 OK
Length: 38741381 (37M) [application/octet-stream]
Saving to: ‘balloon_dataset.zip’

balloon_dataset.zip 100%[===================>]  36.95M  --.-KB/s    in 0.1s    

2023-12-19 14:11:09 (252 MB/s) - ‘balloon_dataset.zip’ saved [38741381/38741381]
```
The balloon dataset comes in the VIA annotation format. However, to prepare the annotations for the model, DetrImageProcessor expects them to be in COCO format. The following cell clones a Github repo that includes a function to convert them into COCO format.

```
git clone https://github.com/woctezuma/VIA2COCO
cd VIA2COCO/
git checkout fixes
```
output
```
Cloning into 'VIA2COCO'...
remote: Enumerating objects: 63, done.
remote: Counting objects: 100% (43/43), done.
remote: Compressing objects: 100% (23/23), done.
remote: Total 63 (delta 25), reused 36 (delta 20), pack-reused 20
Receiving objects: 100% (63/63), 16.41 KiB | 2.05 MiB/s, done.
Resolving deltas: 100% (32/32), done.
/content/VIA2COCO
Already on 'fixes'
Your branch is up to date with 'origin/fixes'.
```
```
import convert as via2coco

data_path = '/content/balloon/'

first_class_index = 0

for keyword in ['train', 'val']:

  input_dir = data_path + keyword + '/'
  input_json = input_dir + 'via_region_data.json'
  categories = ['balloon']
  super_categories = ['N/A']
  output_json = input_dir + 'custom_' + keyword + '.json'

  print('Converting {} from VIA format to COCO format'.format(input_json))

  coco_dict = via2coco.convert(
      imgdir=input_dir,
      annpath=input_json,
      categories=categories,
      super_categories=super_categories,
      output_file_name=output_json,
      first_class_index=first_class_index,
  )
```
output:
```    
Converting /content/balloon/train/via_region_data.json from VIA format to COCO format
Saving to /content/balloon/train/custom_train.json
Converting /content/balloon/val/via_region_data.json from VIA format to COCO format
Saving to /content/balloon/val/custom_val.json
```
%cd ..

/content

现在你拥有了一个coco格式的balloon数据集，位置在`/content`.

# train
ball_tuning.py 是微调detr，使用balloon数据集。
ball_prompt_tuning.py 是在微调模型的基础上，进行提示微调。
*_eval.py 是coco验证对于你训练的模型
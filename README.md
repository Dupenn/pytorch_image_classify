# pytorch_image_classify 图片多分类示例

## train
```python
python3 train.py
```
 
## val
```python
python3 inference.py models/model.ckpt data/train/1/3smile.jpg
```

## 数据要求

data 目录下分为train和val两个训练集，其中都包含1和0两个目录，分别存放true和false的数据集（目前采用9:1的数据比例存放）。
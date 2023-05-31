# MSFT-YOLO
Implemented some industrial product surface defect detection using improved yolov5.

- crazing：疯裂，这是一种表面现象，通常表现为钢材表面出现一系列细小的裂纹，这些裂纹通常是交叉分布的。
- inclusion：夹杂，这是一种由杂质或异种材料在钢材中形成的缺陷。
- patches：斑块，这可能是指材料表面的一种不均匀状态，这种状态可能会导致材料的部分区域出现颜色、质地或成分的变化。
- pitted surface：点蚀表面，这种缺陷出现在材料表面，形成一个个小坑，通常是由于腐蚀或磨损引起。
- rolled-in scale：轧入鳞片，这是一种由于轧制过程中，氧化物鳞片被压入钢材表面而形成的缺陷。
- scratches：划痕，这是指由于摩擦或刮擦等原因在钢材表面形成的线状痕迹。

## n-seg-5-29

```python train.py --weights weights/yolov5n-seg.pt --cfg models/yolov5n.yaml --data data/test.yaml --epoch 20 --batch-size 32```

| Class           | Images | Instances（样例数） | P(精度) | R（召回率） | mAP50 | mAP50-95 |
| --------------- | ------ | ------------------- | ------- | ----------- | ----- | -------- |
| all             | 360    | 809                 | 0.628   | 0.672       | 0.692 | 0.36     |
| crazing         | 360    | 164                 | 0.472   | 0.0874      | 0.318 | 0.0996   |
| inclusion       | 360    | 214                 | 0.555   | 0.841       | 0.722 | 0.37     |
| patches         | 360    | 175                 | 0.735   | 0.949       | 0.919 | 0.584    |
| pitted_surface  | 360    | 77                  | 0.788   | 0.724       | 0.754 | 0.441    |
| rolled-in_scale | 360    | 107                 | 0.6     | 0.598       | 0.593 | 0.257    |
| scratches       | 360    | 72                  | 0.615   | 0.833       | 0.848 | 0.41     |

## s-seg-5-31

YOLOv5s summary: 157 layers, 7026307 parameters, 0 gradients, 15.8 GFLOPs                                                     

| Class           | Images | Instances | P     | R      | mAP50 | mAP50-95 |
| --------------- | ------ | --------- | ----- | ------ | ----- | -------- |
| all             | 360    | 809       | 0.693 | 0.634  | 0.702 | 0.373    |
| crazing         | 360    | 164       | 0.831 | 0.0599 | 0.389 | 0.126    |
| inclusion       | 360    | 214       | 0.614 | 0.855  | 0.811 | 0.425    |
| patches         | 360    | 175       | 0.745 | 0.917  | 0.913 | 0.576    |
| pitted_surface  | 360    | 77        | 0.655 | 0.665  | 0.688 | 0.41     |
| rolled-in_scale | 360    | 107       | 0.552 | 0.551  | 0.567 | 0.243    |
| scratches       | 360    | 72        | 0.762 | 0.754  | 0.844 | 0.458    |


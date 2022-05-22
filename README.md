## YOLOV3目标检测模型在Pytorch当中的实现
---

## 目录

1. [性能情况 Performance](#性能情况)
2. [所需环境 Environment](#所需环境)
3. [训练步骤 How to train](#训练步骤)
4. [预测步骤 How to predict](#预测步骤)
5. [实验结果简介 results](#实验结果简介)
6. [参考资料 Reference](#参考资料)


## 性能情况
| 训练数据集 | 权值文件名称 | 测试数据集 | 输入图片大小 | mAP 0.5:0.95 | mAP 0.5 |
| :-----: | :-----: | :------: | :------: | :------: | :-----: |
| VOC-Train2017 | [yolo_weights.pth]| VOC-Val2017 | 416x416 | 38.0 | 67.2

## 所需环境
torch == 1.2.0  

## 训练步骤
1. 数据集的准备   
**使用VOC格式进行训练，训练前需要将VOC的数据集放在根目录**  
将标签文件放在VOCdevkit文件夹下的VOC2007文件夹下的Annotation中。   
将图片文件放在VOCdevkit文件夹下的VOC2007文件夹下的JPEGImages中。
另外需要在model_data1下建立一个cls_classes.txt，里面写自己所需要区分的类别。
2. 数据集的处理   
训练自己的数据集时，可以自己建立一个cls_classes.txt放在model_data下，里面写自己所需要区分的类别。例如：   
model_data/cls_classes.txt文件内容为：      
```python
car
...
```
然后修改voc_annotation.py中的classes_path，使其对应cls_classes.txt，并运行voc_annotation.py获得训练用的2007_train.txt和2007_val.txt。   
3. 开始网络训练   
**训练的参数较多，均在train.py中，其中最重要的部分是train.py里的classes_path。**  
**classes_path用于指向检测类别所对应的txt，这个txt和voc_annotation.py里面的txt一样**  
修改完classes_path后就可以运行train.py开始训练了，在训练多个epoch后，权值会生成在logs文件夹中。  

4. 训练结果预测  
训练结果预测需要用到两个文件，分别是yolo.py和predict.py。在yolo.py里面修改model_path以及classes_path。  
**model_path指向训练好的权值文件，应选择logs文件夹下epoch100所对应的权值文件。  
classes_path指向检测类别所对应的txt，即cls_classes.txt。**  
完成修改后就可以运行predict.py进行检测了。运行后输入图片路径即可检测。  
笔者发现该代码在Ubuntu系统运行时需要改成相对路径!在Windows中运行时需改成绝对路径!

## 预测步骤
1. 按照训练步骤训练。  
2. 在yolo.py文件里面，在如下部分修改model_path和classes_path使其对应训练好的文件；**model_path对应logs文件夹下面的权值文件，classes_path是model_path对应分的类**。  
```python
_defaults = {
    #--------------------------------------------------------------------------#
    #   使用自己训练好的模型进行预测一定要修改model_path和classes_path！
    #   model_path指向logs文件夹下的权值文件，classes_path指向model_data下的txt
    #   如果出现shape不匹配，同时要注意训练时的model_path和classes_path参数的修改
    #--------------------------------------------------------------------------#
    "model_path"        : 'logs/ep100-loss0.043-val_loss0.037.pth',
    "classes_path"      : 'model_data/cls_classes.txt',
    #---------------------------------------------------------------------#
    #   anchors_path代表先验框对应的txt文件，一般不修改。
    #   anchors_mask用于帮助代码找到对应的先验框，一般不修改。
    #---------------------------------------------------------------------#
    "anchors_path"      : 'model_data/yolo_anchors.txt',
    "anchors_mask"      : [[6, 7, 8], [3, 4, 5], [0, 1, 2]],
    #---------------------------------------------------------------------#
    #   输入图片的大小，必须为32的倍数。
    #---------------------------------------------------------------------#
    "input_shape"       : [416, 416],
    #---------------------------------------------------------------------#
    #   只有得分大于置信度的预测框会被保留下来
    #---------------------------------------------------------------------#
    "confidence"        : 0.5,
    #---------------------------------------------------------------------#
    #   非极大抑制所用到的nms_iou大小
    #---------------------------------------------------------------------#
    "nms_iou"           : 0.3,
    #---------------------------------------------------------------------#
    #   该变量用于控制是否使用letterbox_image对输入图像进行不失真的resize，
    #   在多次测试后，发现关闭letterbox_image直接resize的效果更好
    #---------------------------------------------------------------------#
    "letterbox_image"   : False,
    #-------------------------------#
    #   是否使用Cuda
    #   没有GPU可以设置成False
    #-------------------------------#
    "cuda"              : False,
}
```
3. 运行predict.py，输入需预测图片的地址，例如：  
```python
img/street.jpg
```
另外可以在predict.py里面进行设置可以进行fps测试和video视频检测。  

## 实验结果简介
笔者首先利用官网的VOC数据集进行训练，达到了很好的效果。后利用gazebo的仿真环境进行了数据集制作，得到了一批较为特殊的数据集，进行车辆识别训练，同样可以识别并且得到了较好的结果。具体实验结果见报告。

## 参考资料
https://github.com/qqwweee/pytorch-yolo3  
https://github.com/eriklindernoren/PyTorch-YOLOv3   
https://github.com/BobLiu20/YOLOv3_PyTorch

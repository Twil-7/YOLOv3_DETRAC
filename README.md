# YOLOv3_DETRAC

# 代码环境：

服务器GPU：用于15万张图片大型训练，YOLOv3模型所有层解冻。训练1个epoch需要22小时，总共训练了4个epoch。

python == 3.8

tensorflow == 2.4.1

keras == 2.4.3 

opencv-python == 4.5.3.56

# 运行

直接运行main.py

DETRAC数据集下载：https://blog.csdn.net/weixin_43653815/article/details/95514857?utm_term=detrac%E6%95%B0%E6%8D%AE%E9%9B%86&utm_medium=distribute.pc_aggpage_search_result.none-task-blog-2~all~sobaiduweb~default-0-95514857&spm=3001.4430

权重文件、视频效果下载：https://blog.csdn.net/Twilight737?spm=1018.2226.3001.5343&type=download


# 训练过程：

优化器 optimizer=Adam(lr=1e-3)

Epoch 1/10

2565/2565 [==============================] - 86006s 34s/step - loss: 154.7657 - val_loss: 73.8884

Epoch 2/10

2565/2565 [==============================] - 99136s 39s/step - loss: 30.2837 - val_loss: 60.6485

Epoch 3/10

2565/2565 [==============================] - 118117s 46s/step - loss: 26.0896 - val_loss: 65.0846

Epoch 4/10

2565/2565 [==============================] - 118138s 46s/step - loss: 24.5482 - val_loss: 63.6707

# 实验效果

虽然数据集看上去质量不佳，小目标过多，且resize后尺寸变形严重，但在超大规模数据量下，居然效果依旧很好，令人惊艳。

在测试集中，车辆检测精度几近100%，而且轮廓定位精准。

具体检测效果可见demo文件夹。

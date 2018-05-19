# FaceRecognization
学习人脸识别算法

# 数据
- 位置：`./data/PIE`
- 格式：`.mat`

|列名|说明|
|:----:|:----:|
|fea|图像数据，64\*64\*1|
|gnd|标签,int|
|isTest|0 or 1|

# 数据处理
- 1. 准备数据
- 2. 建立文件夹`./data/PIE/images`
- 3. 调整`./data_augumentation.py`第7行数据增强倍数`AUGUMENT_X`
- 4. 运行`./data_augumentation.py`
- 将在`./data/PIE/images`生成增强后的图片

# 训练&测试
- 1. 数据处理
- 2. 运行`./main.py`
  - 参数1：
    - train：训练
    - eval：测试
  - 参数2：
    - image：从图片读取数据
    - pickle：从图片读数据后会自动存为pickle，使用这个参数可以加快读取速度
  - 参数3：
    - <文件夹路径>：image时设置文件夹路径，建议存固态硬盘
    - pickle：pickle时占位，无意义
  - 示例：
    - `python ./main.py train image C:/images`
    - `python ./main.py eval pickle pickle`
- 3. 等待训练/测试完成
  - 训练时可以随时关闭，下次开启时自动继续训练


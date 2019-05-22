
# 基于TensorFlow训练花朵识别模型的源码和Demo
下面就通过对现有的 Google Inception-V3 模型进行 retrain ，对 5 种花朵样本数据的进行训练，来完成一个可以识别五种花朵的模型，并将新训练的模型进行测试部属，让大家体验一下完整的流程。

![花朵训练样本](https://img-blog.csdn.net/20180602195623764?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0FueW1ha2VfcmVu/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)


### 安装 TensorFlow （Mac 为例）

其他平台可以直接参考官网说明：[Installing TensorFlow](https://www.tensorflow.org/install/)

#### 首先检查系统是否安装了 Python

要安装 `TensorFlow` ，你的系统必须依据安装了以下任一 `Python` 版本：

*   **Python 2.7**
*   **Python 3.3+**

如果做数据处理较多的话，建议安装Anaconda， **Anaconda** 是一种Python语言的免费增值开源发行版 ，用于进行大规模数据处理, 预测分析, 和科学计算, 致力于简化包的管理和部署。Anaconda使用软件包管理系统Conda进行包管理。安装完成后输入shell下输入`python`即可查看Anaconda对应的Python 版本，我使用的是Python 2.7.14：
```
➜  ~ python
Python 2.7.14 |Anaconda, Inc.| (default, Dec  7 2017, 11:07:58)
[GCC 4.2.1 Compatible Clang 4.0.1 (tags/RELEASE_401/final)] on darwin
Type "help", "copyright", "credits" or "license" for more information.

```
如果你的系统还没有安装符合以上版本的 Python，现在安装。



#### 通过 pip 安装 TensorFlow

```
# Python 2
➜ pip install tensorflow
# Python 3
➜ pip3 install tensorflow 

```

#### 通过官方样例测试 TensorFlow 是否正常安装

进入 Python 环境后输入以下代码，当出现 `“Hello, TensorFlow!”` 时表明已经安装成功，可正常使用 TensorFlow 了。

```
➜ python
import tensorflow as tf
hello = tf.constant('Hello, TensorFlow!')
sess = tf.Session()
print(sess.run(hello))
Hello, TensorFlow!

```

### 准备训练样本

现在我们要训练花朵的识别模型，这是 Google 在TensorFlow里面提供的一个例子，其中包含了5类花朵的训练图片。可以新建个flower_demo文件夹，用于存放数据和训练的模型。

**下载并解压得到训练样本**

```
cd flower_demo
# 下载和解压花朵训练数据
curl -O http://download.tensorflow.org/example_images/flower_photos.tgz
tar xzf flower_photos.tgz

```

打开训练样本文件夹 flower_photos ，里面有 5 种类别的花：`daisy(雏菊), dandelion(蒲公英), roses(玫瑰), sunflowers(向日葵) , tulips(郁金香)`，总共3672张，每个类别的大概有 600-900 张训练样本图片,具体如下：

```
cd flower_photos
for dir in `find ./ -maxdepth 1 -type d`;do echo -n -e "$dir\t";find $dir -type f|wc -l ;done;
./	    3672
.//roses	     641
.//sunflowers	     699
.//daisy	     633
.//dandelion	     898
.//tulips	     799

```
### 开始训练

**下载训练模型使用的 retrain 脚本**
该脚本会自动下载 google Inception v3 模型相关文件，`retrain.py` 是 Google 提供的以ImageNet图片分类模型为基础模型，利用flower_photos数据迁移训练花朵识别模型的脚本。

```
 cd flower_demo
 curl -O https://raw.githubusercontent.com/tensorflow/tensorflow/r1.1/tensorflow/examples/image_retraining/retrain.py

```
**启动训练脚本，开始训练模型**

在运行 `retrain.py` 脚本时，需要配置一些运行命令参数，比如指定模型输入输出相关名称和其他训练要求的配置。其中`--how_many_training_steps=4000`配置代表训练迭代次数，默认值为4000，如果机器较差，可以适当减少这个值。

```
➜ cd flower_demo
➜ python3 retrain.py \
  --bottleneck_dir=bottlenecks \
  --how_many_training_steps=4000 \
  --model_dir=inception \
  --summaries_dir=training_summaries/basic \
  --output_graph=retrained_graph.pb \
  --output_labels=retrained_labels.txt \
  --image_dir=flower_photos

```
这里我们训练4000steps，时间不是很久，我在配备4.2 GHz Intel Core i7处理器的iMac上，不适用GPU大概就5分钟就能训练完成。模型训练完成后，可以看到测试集上`Final test accuracy = 92.1%`，也就是说我们训练的5类花朵识别模型，在测试集上已经有92%的识别准确率了。其中生成的 `retrained_labels.txt` 和 `retrained_graph.pb` 这两个是模型相关文件。
```
2018-06-02 15:47:00.266119: Step 3950: Train accuracy = 94.0%
2018-06-02 15:47:00.266159: Step 3950: Cross entropy = 0.135385
2018-06-02 15:47:00.327843: Step 3950: Validation accuracy = 93.0% (N=100)
2018-06-02 15:47:00.976543: Step 3960: Train accuracy = 94.0%
2018-06-02 15:47:00.976591: Step 3960: Cross entropy = 0.234760
2018-06-02 15:47:01.038559: Step 3960: Validation accuracy = 91.0% (N=100)
2018-06-02 15:47:01.667255: Step 3970: Train accuracy = 97.0%
2018-06-02 15:47:01.667372: Step 3970: Cross entropy = 0.167394
2018-06-02 15:47:01.731935: Step 3970: Validation accuracy = 87.0% (N=100)
2018-06-02 15:47:02.355780: Step 3980: Train accuracy = 96.0%
2018-06-02 15:47:02.355818: Step 3980: Cross entropy = 0.151201
2018-06-02 15:47:02.418314: Step 3980: Validation accuracy = 91.0% (N=100)
2018-06-02 15:47:03.042364: Step 3990: Train accuracy = 99.0%
2018-06-02 15:47:03.042402: Step 3990: Cross entropy = 0.094383
2018-06-02 15:47:03.103718: Step 3990: Validation accuracy = 91.0% (N=100)
2018-06-02 15:47:03.667861: Step 3999: Train accuracy = 99.0%
2018-06-02 15:47:03.667899: Step 3999: Cross entropy = 0.106797
2018-06-02 15:47:03.729215: Step 3999: Validation accuracy = 94.0% (N=100)
Final test accuracy = 92.1% (N=353)
```
### 测试训练完成后的模型

同样的，我们先下载测试模型的脚本 `label_image.py`，然后从flower_photos/daisy/文件夹下选择图片488202750_c420cbce61.jpg，测试我们训练后的模型的识别准确率，当然你也可以百度搜索一张5类花朵的任意一张图测试识别效果，从下图可以看出，我们训练的算法模型认为这张图属于`daisy`的概率高达98.9%.

```
➜ cd flower_demo
➜ curl -L https://goo.gl/3lTKZs > label_image.py
➜ python label_image.py flower_photos/daisy/488202750_c420cbce61.jpg

daisy (score = 0.98921)
sunflowers (score = 0.00948)
dandelion (score = 0.00088)
tulips (score = 0.00038)
roses (score = 0.00005)
```
![蒲公英测试图](https://img-blog.csdn.net/20180602200253465?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0FueW1ha2VfcmVu/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)
有人说`label_image.py`无法下载，代码如下：
```
import os, sys
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# change this as you see fit
image_path = sys.argv[1]

# Read in the image_data
image_data = tf.gfile.FastGFile(image_path, 'rb').read()

# Loads label file, strips off carriage return
label_lines = [line.rstrip() for line in tf.gfile.GFile("retrained_labels.txt")]

# Unpersists graph from file
with tf.gfile.FastGFile("retrained_graph.pb", 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    tf.import_graph_def(graph_def, name='')

with tf.Session() as sess:
    # Feed the image_data as input to the graph and get first prediction
    softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
    
    predictions = sess.run(softmax_tensor, {'DecodeJpeg/contents:0': image_data})
    
    # Sort to show labels of first prediction in order of confidence
    top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]
    
    for node_id in top_k:
        human_string = label_lines[node_id]
        score = predictions[0][node_id]
        print('%s (score = %.5f)' % (human_string, score))
```
我们随便从百度搜索一张蒲公英（dandelion）的图，保存到`test/WechatIMG383.jpg`，测试结果显示属于蒲公英的概率为99.59%.

```
python label_image.py test/WechatIMG383.jpg

dandelion (score = 0.99592)
sunflowers (score = 0.00359)
daisy (score = 0.00042)
tulips (score = 0.00005)
roses (score = 0.00001)
```
以上基本是模型训练和测试的全部过程，希望能让大家对深度学习的完整项目有个大致的了解。

**启动 TensorBoard**
TensorBoard 是 TensorFlow 自带的训练效果可视化的分析工具，我们可以利用此工具检测和分析模型的收敛情况，比如查看loss的下降、acc的提升和查看可视化的网络结构图等。在我们建的工程目录下，启动tensorboard的具体命令如下：

```
➜ cd flower_demo
➜ tensorboard --logdir training_summaries

```

启动 TensorBoard 会占用系统 `6006` 端口 ，再启动一个新的 TensorBoard 之前，必须要 kill 已在运行的 TensorBoard 任务。

```
➜ pkill -f "tensorboard

```
**启动浏览器查看 TensorBoard**

启动TensorBoard后，可以启动浏览器，在地址栏中输入 `localhost:6006` 来查看训练进度以及loss和准确度的变化，分析模型等。

![训练过程中loss和准确率的变化](https://img-blog.csdn.net/20180602200349392?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0FueW1ha2VfcmVu/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)


![花朵识别网络模型的后半部分](https://img-blog.csdn.net/20180602200413197?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0FueW1ha2VfcmVu/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

# TensorFlow Object Detection API

## TODO

- 将数据转成tf.record格式

## 配置

1. [Installation Guide](https://github.com/tensorflow/models/blob/master/object_detection/g3doc/installation.md)
2. 编译protobuf libraries
`protoc object_detection/protos/*.proto --python_out=.`
3. 设置环境变量
将`object_detection`所在父目录和`slim`目录添加到`PYTHONPATH`环境变量中：`C:\Users\19097\Desktop\object_detection;C:\Users\19097\Desktop\object_detection\slim;`
4. 测试配置是否成功
`python object_detection/builders/model_builder_test.py`

## Training on the Oxford-IIIT Pets Dataset

1. [下载数据集](http://www.robots.ox.ac.uk/~vgg/data/pets/)
2. 将数据转成TFRecord格式
	```
	python create_pet_tf_record.py --label_map_path=pet_label_map.pbtxt  --data_dir=. --output_dir=.
	```

## Downloading a COCO-pretrained Model for Transfer Learning

1. [下载模型](http://storage.googleapis.com/download.tensorflow.org/models/object_detection/faster_rcnn_resnet101_coco_11_06_2017.tar.gz)
2. 修改`samples/configs/faster_rcnn_resnet101_pets.config`文件中的`PATH_TO_BE_CONFIGURED`路径

## Train
```
	python object_detection\train.py
		--logtostderr
		--train_dir=./log
		--pipeline_config_path=C:\Users\19097\Desktop\object_detection\object_detection\samples\configs\faster_rcnn_resnet101_pets.config
```

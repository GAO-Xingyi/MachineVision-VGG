# coding=gbk
import tensorflow as tf
from tensorflow.python import keras
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.applications.vgg16 import VGG16

# 定义一个类，作为迁移模型
class TransferModel(object):

    def __init__(self):
    # 定义训练和测试图片的变化方法，标准化以及数据增强
        #实例化训练集与测试集的generator
        self.train_generator = ImageDataGenerator(rescale=1.0 / 255.0)
        self.test_generator = ImageDataGenerator(rescale=1.0 / 255.0)

        # 指定训练数据和测试的目录
        self.train_dir = "./data/train"
        self.test_dir = "./data/test"

        # 定义图片训练相关网络参数
        self.image_size = (224,224)
        self.batch_size = 32

        # 定义迁移学习的基类模型
        # 不包含全连接层（3个）VGG模型并且加载了参数
        self.base_model = VGG16(weights='imagenet',include_top=False)

    def get_local_data(self):
        """
        读取本地的图片数据及类别
        :return:训练数据和测试数据迭代器
        """
        # 使用flow_from_derectory读取数据
        train_gen = self.train_generator.flow_from_directory(self.train_dir,
                                                 target_size=self.image_size, # VGG要求大小（224，224，3）
                                                 batch_size=self.batch_size,
                                                 class_mode='binary',
                                                 shuffle=True) # 以乱序读取
        test_gen = self.test_generator.flow_from_directory(self.test_dir,
                                                           target_size=self.image_size,
                                                           batch_size=self.batch_size,
                                                           class_mode='binary',
                                                           shuffle=True)
        return train_gen, test_gen

    def refine_base_model(self):
        """
        微调VGG结构，5个blocks(5中分类)+全局平均池化+两个全连接层
        :return:
        """
        # 获取notop模型输出
        # [?, ?, ?, 512]
        x = self.base_model.outputs[0]
        # 于原输出后增加我们的模型结构
        # [?, ?, ?, 512]————>[?, 1 * 512]
        x = keras.layers.GlobalAveragePooling2D()(x)
        # 定义新的迁移模型
        x = keras.layers.Dense(1024, activation=tf.nn.relu)(x) # 1024神经元个数全连接层
        y_predict = keras.layers.Dense(5, activation=tf.nn.softmax)(x) # 全连接层分类模型

        # model定义新模型
        # 输入：VGG模型输入(整合了所有的模型，为最初始的输入)， 输出为:y_predict
        transfer_model = keras.models.Model(inputs=self.base_model.inputs, outputs=y_predict)

        return transfer_model

if __name__ == '__main__':
    tm = TransferModel()
    train_gen, test_gen = tm.get_local_data()
    # print(train_gen)
    # for data in train_gen:
    #     print(data)
    # print(tm.base_model.summary())
    model = tm.refine_base_model()
    print(model)

from tensorflow.keras import layers
from tensorflow.keras.models import Model
import tensorflow as tf

class inception(layers.Layer):
  def __init__(self,filter_1,filter_2,filter_3,filter_4,filter_5,filter_6):
    super(inception,self).__init__(name='')
    self.filter_1 = filter_1
    self.filter_2 = filter_2
    self.filter_3 = filter_3
    self.filter_4 = filter_4
    self.filter_5 = filter_5
    self.filter_6 = filter_6


    self.conv_1_5   =  tf.keras.layers.Conv2D(filters=self.filter_5,strides=1,kernel_size=(1,1),padding='SAME',activation='relu',kernel_initializer=tf.keras.initializers.he_normal())
    self.conv_1_6   =  tf.keras.layers.Conv2D(filters=self.filter_6,strides=1,kernel_size=(1,1),padding='SAME',activation='relu',kernel_initializer=tf.keras.initializers.he_normal())
    self.max_pool_1 =  tf.keras.layers.MaxPool2D(pool_size=(3,3),strides=1,padding='SAME')
    self.conv_2_1   =  tf.keras.layers.Conv2D(filters=self.filter_1,strides=1,kernel_size=(1,1),padding='SAME',activation='relu',kernel_initializer=tf.keras.initializers.he_normal())
    self.conv_2_2   =  tf.keras.layers.Conv2D(filters=self.filter_2,strides=1,kernel_size=(3,3),padding='SAME',activation='relu',kernel_initializer=tf.keras.initializers.he_normal())
    self.conv_2_3   =  tf.keras.layers.Conv2D(filters=self.filter_3,strides=1,kernel_size=(5,5),padding='SAME',activation='relu',kernel_initializer=tf.keras.initializers.he_normal())
    self.conv_2_4   =  tf.keras.layers.Conv2D(filters=self.filter_4,strides=1,kernel_size=(1,1),padding='SAME',activation='relu',kernel_initializer=tf.keras.initializers.he_normal())


  def call(self,network):
    input_1 = self.conv_1_5(network)
    input_2 = self.conv_1_6(network)
    input_3 = self.max_pool_1(network)
    input_4 = self.conv_2_1(network)
    input_1 = self.conv_2_2(input_1)
    input_2 = self.conv_2_3(input_2)
    input_3 = self.conv_2_4(input_2)
    output  = tf.keras.layers.concatenate([input_4,input_3,input_2,input_1],axis=-1)
    return output


class InceptionNet(Model):
    def __init__(self, no_classes, name = "InceptionNet"):
        super(InceptionNet,self).__init__(name=name)
        self.network_layer_1  =  tf.keras.layers.Conv2D(filters=64,kernel_size=(7,7),strides=2,padding='SAME',activation='relu',kernel_initializer=tf.keras.initializers.he_normal())
        self.network_layer_2  =  tf.keras.layers.MaxPool2D(pool_size=(3,3),strides=2)
        self.network_layer_3  =  tf.keras.layers.Conv2D(filters=64,kernel_size=(1,1),strides=1,padding='SAME',activation='relu',kernel_initializer=tf.keras.initializers.he_normal())
        self.network_layer_4  =  tf.keras.layers.Conv2D(filters=192,kernel_size=(3,3),strides=1,padding='SAME',activation='relu',kernel_initializer=tf.keras.initializers.he_normal())
        self.network_layer_5  =  tf.keras.layers.MaxPool2D(pool_size=(3,3),strides=2)
        self.network_layer_6  =  inception(64,128,32,32,96,16)
        self.network_layer_7  =  inception(128,192,96,64,128,32)
        self.network_layer_8  =  tf.keras.layers.MaxPool2D(pool_size=(3,3),strides=2)
        self.network_layer_9  = inception(192,208,48,64,128,32)
        self.network_layer_10 = inception(160,224,64,64,112,24)
        self.network_layer_11 = inception(128,256,64,64,128,24)
        self.network_layer_12 = inception(112,288,64,64,144,32)
        self.network_layer_13 = inception(256,320,128,128,160,32)
        self.network_layer_14 = tf.keras.layers.MaxPool2D(pool_size=(3,3),strides=2)
        self.network_layer_15 = inception(256,320,128,128,160,32)
        self.network_layer_16 = inception(384,384,128,128,192,48)        
        self.network_layer_17 = tf.keras.layers.GlobalAveragePooling2D(data_format="channels_last")
        self.network_layer_18 = tf.keras.layers.Dropout(0.4)
        self.network_layer_19 = tf.keras.layers.Flatten()
        self.network_layer_20 = tf.keras.layers.Dense(units=no_classes ,activation="softmax", kernel_initializer=tf.keras.initializers.TruncatedNormal())
    
    def call(self, inp):

        network = self.network_layer_1(inp)
        network = self.network_layer_2(network)
        
        network = tf.nn.local_response_normalization(network,depth_radius=2,alpha=0.00002,beta=0.75)
        network = self.network_layer_3(network)
        network = self.network_layer_4(network)

        network = tf.nn.local_response_normalization(network,depth_radius=2,alpha=0.00002,beta=0.75)
        network = self.network_layer_5(network)

        network = self.network_layer_6(network)
        network = self.network_layer_7(network)

        network = self.network_layer_8(network)
        network = self.network_layer_9(network)

        network = self.network_layer_10(network)
        network = self.network_layer_11(network)

        network = self.network_layer_11(network)
        network = self.network_layer_12(network)
        network = self.network_layer_13(network)
        network = self.network_layer_14(network)
        network = self.network_layer_15(network)
        network = self.network_layer_16(network)
        network = self.network_layer_17(network)
        network = self.network_layer_18(network)
        network = self.network_layer_19(network)
        network = self.network_layer_20(network)

        return network

import sys, os, math

import numpy as np
import tensorflow as tf

import cv2

from sklearn.metrics import precision_score, recall_score, accuracy_score
from sklearn.model_selection import train_test_split

def create_inputs():
    inputs = tf.placeholder(shape=[None, 300, 300, 3], dtype=tf.float32)
    labels = tf.placeholder(shape=[None], dtype=tf.float32)
    
    return inputs, labels

def create_conv_net(image_batch):
        
    #Preprocess data
    #Normalize image data. Since it is already from 0 to 255, it is not necessary to offset to min
    image_batch = image_batch / 255 
        
    conv1 = tf.layers.conv2d(image_batch, filters=8, kernel_size=[5,5], strides=[1, 1], padding='SAME',
                            kernel_initializer=None, activation=tf.nn.relu)
    conv1 = tf.layers.max_pooling2d(conv1, pool_size=[5,5], strides=[2,2], padding='SAME')
    #print(conv1.get_shape())
        
    conv2 = tf.layers.conv2d(conv1, filters=16, kernel_size=[3,3], strides=[1, 1], padding='SAME',
                            kernel_initializer=None, activation=tf.nn.relu)
    conv2 = tf.layers.max_pooling2d(conv2, pool_size=[3,3], strides=[2,2], padding='SAME')
    #print(conv2.get_shape())
    
    conv3 = tf.layers.conv2d(conv2, filters=32, kernel_size=[3,3], strides=[1, 1], padding='SAME',
                            kernel_initializer=None, activation=tf.nn.relu)
    conv3 = tf.layers.max_pooling2d(conv3, pool_size=[3,3], strides=[2,2], padding='SAME')
    #print(conv3.get_shape())
    
    conv4 = tf.layers.conv2d(conv3, filters=64, kernel_size=[3,3], strides=[1, 1], padding='SAME',
                            kernel_initializer=None, activation=tf.nn.relu)
    conv4 = tf.layers.max_pooling2d(conv4, pool_size=[3,3], strides=[2,2], padding='SAME')
    #print(conv4.get_shape())
    
    conv5 = tf.layers.conv2d(conv4, filters=128, kernel_size=[3,3], strides=[1, 1], padding='SAME',
                            kernel_initializer=None, activation=tf.nn.relu)
    conv5 = tf.layers.max_pooling2d(conv5, pool_size=[3,3], strides=[2,2], padding='SAME')
    #print(conv5.get_shape())

    flatten_layer = tf.layers.flatten(conv5)
    
    h1 = tf.layers.dense(flatten_layer, 5000, tf.nn.relu)
    h2 = tf.layers.dense(h1, 1000, tf.nn.relu)
    h3 = tf.layers.dense(h2, 256, tf.nn.relu)
    
    logits = tf.squeeze(tf.layers.dense(h3, 1), axis=1)
    
    outputs = tf.nn.sigmoid(logits)
        
    return logits, outputs

def create_optimizer(logits, labels, learning_rate):
    loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels)
    loss = tf.reduce_mean(loss)
    
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)
    
    return optimizer, loss

class CSGOModel:

    def __init__(self, learning_rate=0.001):
        tf.reset_default_graph()
        
        inputs, labels = create_inputs()
        logits, outputs = create_conv_net(inputs)
        optimizer, loss = create_optimizer(logits, labels, learning_rate)
        
        session = tf.Session()
        
        last_cp_file = tf.train.latest_checkpoint("checkpoints")

        if last_cp_file:
            print("Restoring last checkpoint...")
            saver = tf.train.Saver()
            saver.restore(session, last_cp_file)
        else:
            print("No checkpoint found. Creating fresh variables.")
            session.run(tf.global_variables_initializer())

        #Getting references
        self.inputs = inputs
        self.labels = labels
        self.logits = logits
        self.outputs = outputs
        self.optimizer = optimizer
        self.loss = loss

        self.session = session


    def partial_fit(self, x_train, y_train):
        _, loss_value = self.session.run([self.optimizer, self.loss], feed_dict={
            self.inputs: x_train,
            self.labels: y_train
        })

        return loss_value

    def predict(self, x_predict):
        predictions = self.session.run(self.outputs, feed_dict={ self.inputs: x_predict })
        return predictions

    def save(self):
        saver = tf.train.Saver()
        saver.save(self.session, "checkpoints/model.ckpt")



def get_image_batches(files_list, batch_size):
    for i in range(0, len(files_list), batch_size):
        x_batch, y_batch = list(), list()
        
        for filename in files_list[i:i+batch_size]:
            img = cv2.imread("train_imgs/" + filename)
            label = int(filename[0] == 'p') #If the first letter of the name is p of "pressed", return 1. otherwise 0

            x_batch.append(img)
            y_batch.append(label)

        x_batch = np.array(x_batch)
        yield x_batch, y_batch
        

def main(args):
    train_files = os.listdir("train_imgs")

    train_files, valid_files = train_test_split(train_files, train_size=0.8)
    print("Train files: {} \t Valid files: {}".format(len(train_files), len(valid_files)))

    notpressed_ratio = len([f for f in valid_files if 'not' in f]) / len(valid_files)
    print("Ratio of not pressed data: {}".format(notpressed_ratio))

    N_EPOCHS = 10

    BATCH_SIZE = 64

    n_batches = math.ceil(len(train_files) / BATCH_SIZE)

    model = CSGOModel()

    for e in range(N_EPOCHS):
        
        i_batch = 0
        
        for x_batch, y_batch in get_image_batches(train_files, BATCH_SIZE):
            
            i_batch += 1

            loss_value = model.partial_fit(x_batch, y_batch)
        
            print("Epoch {}/{} \t Batch {}/{} \t Loss: {}".format(e+1, N_EPOCHS, i_batch, n_batches, loss_value))
            
            
        print("Calculating performance...")
        acc_values = []
        recall_values = []
        precision_values = []

        for valid_x_batch, valid_y_batch in get_image_batches(valid_files, BATCH_SIZE):
    
            predictions = model.predict(valid_x_batch) > 0.5

            acc_values.append(accuracy_score(valid_y_batch, predictions))
            recall_values.append(recall_score(valid_y_batch, predictions))
            precision_values.append(precision_score(valid_y_batch, predictions))

        print("Accuracy: {} \t Recall: {} \t Precision: {}".format(
            np.mean(acc_values), np.mean(recall_values), np.mean(precision_values)))
        
    model.save()
    print("Model saved.")

if __name__ == "__main__":
    main(sys.argv[1:])
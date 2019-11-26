import argparse
import pickle
import sys
import numpy as np
import cifar10utils
import cifar100utils

import tensorflow as tf

cifar10_dataset_folder_path = 'cifar-10-batches-py'
save_model_path = './image_classification'


class AlexNet:
    def __init__(self, dataset, learning_rate):
        self.dataset = dataset
        self.learning_rate = learning_rate

        if self.dataset == 'cifar10':
            self.num_classes = 10
        elif self.dataset == 'cifar100':
            self.num_classes = 100

        # Weights for each layer
        self.conv_weights = {
            "c1_weights": tf.Variable(tf.truncated_normal([11, 11, 3, 96]),
                                      name="c1_weights"),
            "c2_weights": tf.Variable(tf.truncated_normal([5, 5, 48, 256]),
                                      name="c2_weights"),
            "c3_weights": tf.Variable(tf.truncated_normal([3, 3, 256, 384]),
                                      name="c3_weights"),
            "c4_weights": tf.Variable(tf.truncated_normal([3, 3, 192, 384]),
                                      name="c4_weights"),
            "c5_weights": tf.Variable(tf.truncated_normal([3, 3, 192, 256]),
                                      name="c5_weights"),
            "f1_weights": tf.Variable(tf.truncated_normal([6*6*256, 4096]),
                                      name="f1_weights"),
            "f2_weights": tf.Variable(tf.truncated_normal([4096, 4096]),
                                      name="f2_weights"),
            "f3_weights": tf.Variable(tf.truncated_normal([4096, self.num_classes]),
                                      name="f3_weights")
        }

        # Biases for each layer
        self.conv_biases = {
            "c1_biases": tf.Variable(tf.truncated_normal([96]), name="c1_biases"),
            "c2_biases": tf.Variable(tf.truncated_normal([256]), name="c2_biases"),
            "c3_biases": tf.Variable(tf.truncated_normal([384]), name="c3_biases"),
            "c4_biases": tf.Variable(tf.truncated_normal([384]), name="c4_biases"),
            "c5_biases": tf.Variable(tf.truncated_normal([256]), name="c5_biases"),
            "f1_biases": tf.Variable(tf.truncated_normal([4096]), name="f1_biases"),
            "f2_biases": tf.Variable(tf.truncated_normal([4096]), name="f2_biases"),
            "f3_biases": tf.Variable(tf.truncated_normal([self.num_classes]),
                                     name="f3_biases")
        }

        self.input = tf.placeholder(tf.float32, [None, 224, 224, 3], name='input')
        self.label = tf.placeholder(tf.int32, [None, self.num_classes], name='label')
        self.logits = self.loadmodel()

        # tf.identity is useful when you want to explicitly transport tensor between devices
        # (like, from GPU to a CPU). The op adds send/recv nodes to the graph, which make a
        # copy when the devices of the input and the output are different.
        self.model = tf.identity(self.logits)

        self.cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logits,
                                                                                       labels=self.label))

        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate,
                                                name='adam').minimize(self.cross_entropy)

        self.correctpred = tf.equal(tf.argmax(self.model, 1), tf.argmax(self.label, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correctpred, tf.float32), name='accuracy')


    def loadmodel(self):
        """
        implementing alexnet architecture
        """

        conv1 = tf.nn.conv2d(input=self.input, filter=self.conv_weights["c1_weights"],
                             strides=[1,4,4,1], padding='SAME', name='conv1')
        conv1 += self.conv_biases["c1_biases"]
        conv1 = tf.nn.relu(conv1)
        print("Convolutional Layer 1 shape: {}".format(conv1.get_shape()))
        lrn1 = tf.nn.local_response_normalization(conv1, bias=2, alpha=0.0001, beta=0.75)
        maxpool1 = tf.nn.max_pool2d(lrn1, ksize=[1,3,3,1], strides=[1,2,2,1], padding='VALID')
        print("Maxpool 1 shape: {}".format(maxpool1.get_shape()))

        conv2 = tf.nn.conv2d(input=maxpool1, filter=self.conv_weights["c2_weights"],
                             strides=[1,1,1,1], padding='SAME', name='conv2')
        conv2 += self.conv_biases["c2_biases"]
        conv2 = tf.nn.relu(conv2)
        print("Convolutional Layer 2 shape: {}".format(conv2.get_shape()))
        lrn2 = tf.nn.local_response_normalization(conv2, bias=2, alpha=0.0001, beta=0.75)
        maxpool2 = tf.nn.max_pool2d(lrn2, ksize=[1,3,3,1], strides=[1,2,2,1], padding='VALID')
        print("Maxpool 2 shape: {}".format(maxpool2.get_shape()))

        conv3 = tf.nn.conv2d(input=maxpool2, filter=self.conv_weights["c3_weights"],
                             strides=[1,1,1,1], padding='SAME', name='conv3')
        conv3 += self.conv_biases["c3_biases"]
        conv3 = tf.nn.relu(conv3)
        print("Convolutional Layer 3 shape: {}".format(conv3.get_shape()))

        conv4 = tf.nn.conv2d(input=conv3, filter=self.conv_weights["c4_weights"],
                             strides=[1,1,1,1], padding='SAME', name='conv4')
        conv4 += self.conv_biases["c4_biases"]
        conv4 = tf.nn.relu(conv4)
        print("Convolutional Layer 4 shape: {}".format(conv4.get_shape()))

        conv5 = tf.nn.conv2d(input=conv4, filter=self.conv_weights["c5_weights"],
                             strides=[1,1,1,1], padding='SAME', name='conv5')
        conv5 += self.conv_biases["c5_biases"]
        conv5 = tf.nn.relu(conv5)
        print("Convolutional Layer 5 shape: {}".format(conv5.get_shape()))
        maxpool5 = tf.nn.max_pool2d(conv5, ksize=[1,3,3,1], strides=[1,2,2,1], padding='VALID')
        print("Maxpool 5 shape: {}".format(maxpool5.get_shape()))

        # Flatten the output layer
        feature_map = tf.reshape(maxpool5, [-1, 6 *6 *256])
        print("Reshaped Layer shape: {}".format(feature_map.get_shape()))

        # Fully connected layer 1
        fc1 = tf.matmul(feature_map, self.conv_weights["f1_weights"]) + self.conv_biases["f1_biases"]
        fc1 = tf.nn.dropout(fc1, 0.5)
        print("Fully Connected Layer 1 shape: {}".format(fc1.get_shape()))

        # Fully connected layer 2
        fc2 = tf.matmul(fc1, self.conv_weights["f2_weights"]) + self.conv_biases["f2_biases"]
        fc2 = tf.nn.dropout(fc2, 0.5)
        print("Fully Connected Layer 2 shape: {}".format(fc2.get_shape()))

        # Fully connected layer 3
        fc3 = tf.matmul(fc2, self.conv_weights["f3_weights"]) + self.conv_biases["f3_biases"]
        print("Fully Connected Layer 3 shape: {}".format(fc3.get_shape()))
        return fc3

    def _train_cifar10(self, sess, input, label, optimizer, accuracy, epoch, batch,
                       batch_size, valid_set):
        tmpValidFeatures, valid_labels = valid_set

        for batchX, batchY in cifar10utils.load_preprocess_training_batch(batch,
                                                                          batch_size):
            opt = sess.run(optimizer, feed_dict={input: batchX, label: batchY})
            print(opt)
        print("Epoch {:>2}, CIFAR-10 Batch {}: ".format(epoch+1, batch), end=" ")

        # calculate the mean accuracy over all validation dataset
        acc = 0
        for validX, validY in cifar10utils.batch_features_labels(tmpValidFeatures,
                                                                 valid_labels, batch_size):
            acc += sess.run(accuracy, feed_dict={input: validX, label: validY})

        tmp_num = tmpValidFeatures.shape[0] / batch_size
        print('Validation Accuracy {:.6f}'.format(acc/tmp_num))


    def train(self, epochs, batch_size, valid_set, save_model_path):
        # tmpValidFeatures, valid_labels = valid_set
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            with tf.device("/gpu:0"):
                print("Initialze Global Variables....")
                sess.run(tf.global_variables_initializer())

                print("Start the training......")
                for epoch in range(epochs):
                    n_batches = 5

                    if self.dataset == 'cifar10':
                        for batch in range(1, n_batches + 1):
                            self._train_cifar10(sess, self.input, self.label,
                                                self.optimizer, self.accuracy, epoch, batch,
                                                batch_size, valid_set)



def parseargs(args):
    parser = argparse.ArgumentParser(description="Script for Running AlexNet")
    parser.add_argument('--dataset', help='imagenet or cifar10, cifar10 is the default',
                        default='cifar10')
    parser.add_argument('--dataset-path', help='location where the dataset is present',
                        default='none')
    parser.add_argument('--gpu-mode', help='single or multi', default='single')
    parser.add_argument('--learning-rate', help='learning-rate', default=0.00005)
    parser.add_argument('--epochs', default=20)
    parser.add_argument('--batch-size', default=64)

    return parser.parse_args(args)


def main():
    args = sys.argv[1:]
    args = parseargs(args)

    dataset = args.dataset
    dataset_path = args.dataset_path
    gpu_mode = args.gpu_mode
    learning_rate = args.learning_rate
    epochs = args.epochs
    batch_size = args.batch_size

    if dataset == 'cifar10' and dataset_path == 'none':
        cifar10utils.download(cifar10_dataset_folder_path)

    if dataset == 'cifar10':
        print('Preprocess and Save Data...')
        cifar10utils.preprocess_and_save_data(cifar10_dataset_folder_path)

        print('Load Features and Labels for valid dataset...')
        valid_features, valid_labels = pickle.load(open('preprocess_validation.p',
                                                        mode='rb'))

        print('Converting Valid images to fit into Imagenet size...')
        tmpValidfeatures = cifar10utils.convert_to_imagenet_size(valid_features[:200])

    else:
        sys.exit(0)

    alexNet = AlexNet(dataset, learning_rate)
    alexNet.train(epochs, batch_size, (tmpValidfeatures, valid_labels), save_model_path)


if __name__ == "__main__":
    main()
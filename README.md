# AlexNet-Tensorflow

An Implementation of AlexNet Convolutional Neural Network Architecture by Krizhevsky, Sutskever & Hinton using Tensorflow.

---

This is a simple implementation of the great paper **ImageNet Classification with Deep Convolutional Neural Networks** by **Alex Krizhevsky, Ilya Sutskever** and **Geoffrey Hinton**. Some key takeaways of this paper is stated here in addition to the implemented model in the *alexnet.py* model.

---

# Key Takeaways

## Dataset: ImageNet

The dataset used for this project is a subset of the main **ImageNet** dataset. This subset includes **1.2M** high-resolution images in **1000** categories. The main ImageNet dataset contains more than **15M** high-res images in about **22000**categories. **1.2M** images is a massive number of images which urges a model with **large learning capacity**.

## Capacity of CNNs

The capacity of CNNs is determined with their **Breadth** and **Depth**. Note that if the network size is large, then the low number of training samples can cause **overfitting**. It is also important to mention that most of the parameters of the model is layed in the **fully-connected** layers. In this specific example(this paper), each of the convolutional layers contains no more than **1%** of the model's parameters.

## Top-5 Error Rate

It is a fraction of test images for which the correct label is not among the five labels considered most probable by the model.

## ReLU vs Tanh

**Tanh** is known to be saturating in long time which cause in a longer training time and late convergence of gradients. Thus we'll use **ReLU** as a replacement.

## Lateral Inhibition

The capacity of an excited neuron to reduce the activity of its neighbours.

## Pooling Layers

They are used to summarize the outputs of neighbouring group of neurons in the same kernel map. If **S** is the number of **pooling pixels** and the  **neighbourhoods** are **z*z**:

```javascript
Local Pooling (s = z)
Overlapped Pooling (s < z): Less overfitting.
```

## Dropout

With dropout, we'll set the output of each hidden neuron to 0 with some given probability. So, every time an input is presented the neural network samples a different architecture but, all these architectures are sharing the weights. Using dropout, we'll have less **neuron co-adaptations**. Dropout **increases** the convergence time. Dropout is used in the first two fully connected layers.

## Fully Connected Layers

Fully-connected layers perform **high-level reasoning**. This is used to learn **non-linear** relationships between features. We use **FC Layers** to transform an **invariance feature space** to a **hypothesis learning problem**.

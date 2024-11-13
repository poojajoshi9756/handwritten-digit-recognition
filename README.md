HANDWRITTEN DIGIT RECOGNITION USING MNISET DATASET


This project demonstrates a neural network model using TensorFlow and Keras to classify handwritten digits from the MNIST dataset. The MNIST dataset contains images of digits (0-9) which are used for training and testing the model. This project also includes code for loading and processing custom images, making predictions, and visualizing the results.



Introduction
This project uses a simple deep neural network to recognize and classify handwritten digits. It is implemented with TensorFlow and Keras, and includes preprocessing steps to normalize the input data, define the model architecture, and make predictions on custom images.

Installation

To run this project, you need to have the following libraries installed:
- TensorFlow
- OpenCV (for reading custom images)
- Matplotlib (for visualizing results)
- NumPy
- Pandas

Install the required libraries with:
```bash
pip install tensorflow opencv-python-headless matplotlib numpy pandas
```

 Dataset

The MNIST dataset, a widely used dataset for image classification, is loaded directly from TensorFlow:
python
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()


The dataset is split into training and testing sets:
- Training set: 60,000 images
- Testing set: 10,000 images

Each image in the dataset is a 28x28 pixel grayscale image representing a handwritten digit from 0 to 9.

Model Architecture

The model is a sequential neural network that includes the following layers:
1. Flatten Layer - Converts the 28x28 pixel input into a 1D array of 784 values.
2. Dense Layer (128 units, ReLU activation) - Fully connected layer with 128 neurons and ReLU activation function.
3. Dense Layer (128 units, ReLU activation) - Another fully connected layer with 128 neurons and ReLU activation function.
4. Output Layer (10 units, Softmax activation) - The final layer has 10 neurons with softmax activation to classify each image as one of 10 possible digits (0-9).

 Code:

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(28,28)))
model.add(tf.keras.layers.Dense(units=128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(units=128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(units=10, activation=tf.nn.softmax))


Model Compilation
The model is compiled with:
- Optimizer: Adam
- Loss Function: Sparse Categorical Crossentropy
- Metric: Accuracy


model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])


 Training the Model

The model is trained on the normalized training dataset with 3 epochs:

model.fit(x_train, y_train, epochs=3)


 Evaluating the Model

After training, the model is evaluated on the test dataset:

loss, accuracy = model.evaluate(x_test, y_test)
print("Accuracy:", accuracy)
print("Loss:", loss)


 Predicting Custom Images

The model is also used to make predictions on custom images. Ensure that images are saved in the same directory as this script with filenames `1.png`, `2.png`, etc. These images should be 28x28 grayscale images.

 Code for Prediction

for x in range(1, 5):
    img = cv.imread(f'{x}.png')[:,:,0]
    img = np.invert(np.array([img]))
    prediction = model.predict(img)
    print("----------------")
    print("The predicted value is:", np.argmax(prediction))
    print("----------------")
    plt.imshow(img[0], cmap=plt.cm.binary)
    plt.show()




 Results

The model should achieve an accuracy of around 97-98% on the MNIST test dataset, depending on training and hyperparameters.

 Sample Prediction Output
The following output shows predictions on custom images:


The predicted value is: 3




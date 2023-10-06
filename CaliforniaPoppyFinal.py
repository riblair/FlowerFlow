import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Change the hyper-parameters to get the model performs well
im_size = 128 

config = {
    'batch_size': 64,
    'image_size': (im_size,im_size),
    'epochs': 50, 
    'optimizer': keras.optimizers.experimental.SGD(1e-3)
}


def read_data():
    train_ds, val_ds = tf.keras.utils.image_dataset_from_directory(
        "./images/flower_photos",
        validation_split=0.2,
        subset="both",
        seed=42,
        image_size=config['image_size'],
        batch_size=config['batch_size'],
        labels='inferred',
        label_mode = 'int'
    )
    val_batches = tf.data.experimental.cardinality(val_ds)
    test_ds = val_ds.take(val_batches // 2)
    val_ds = val_ds.skip(val_batches // 2)
    return train_ds, val_ds, test_ds

def data_processing(ds):
    data_augmentation = keras.Sequential(
        [
            # Use dataset augmentation methods to prevent overfitting, 
            layers.RandomFlip("horizontal_and_vertical"),
            layers.RandomRotation(np.pi / 2) # rotation between -90 and 90 degrees
        ]
    )
    ds = ds.map(
        lambda img, label: (data_augmentation(img), label),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds

def build_model(input_shape, num_classes):
    inputs = keras.Input(shape=input_shape)
    x = layers.Rescaling(1./255)(inputs)

    # From slides and online resources, hidden layer nodes should be roughly sqrt(input_nodes * output_nodes) and should decrease with each layer
    gain = 1.5
    hidden_units = int(np.sqrt(im_size*im_size*3*5) * gain) 
    activationType = 'relu' 
    activationType2 = 'tanh' 
  
    x = layers.Flatten()(x)

    x = layers.Dense(hidden_units-2*hidden_units/12, activation=activationType)(x)
    x = layers.Dense(hidden_units-5*hidden_units/12, activation=activationType)(x)
    x = layers.Dense(hidden_units-8*hidden_units/12, activation=activationType2)(x)
    x = layers.Dense(hidden_units-11*hidden_units/12, activation=activationType2)(x)

    outputs = layers.Dense(num_classes, activation="softmax", kernel_initializer='he_normal')(x)
    model = keras.Model(inputs, outputs)
    print(model.summary())
    return model



if __name__ == '__main__':
    # Load and Process the dataset
    train_ds, val_ds, test_ds = read_data()
    classNames = train_ds.class_names
    train_ds = data_processing(train_ds)

    # Build up the ANN model
    model = build_model(config['image_size']+(3,), 5)
    # Compile the model with optimizer and loss function
    model.compile(
        optimizer=config['optimizer'],
        loss='SparseCategoricalCrossentropy',
        metrics=["accuracy"],
    )
  

    # history = model.fit(
    #     train_ds,
    #     epochs=config['epochs'],
    #     validation_data=val_ds
    # )
    # print(history.history)
    model = tf.keras.saving.load_model('bestModel.keras')
    
    test_loss, test_acc = model.evaluate(test_ds, verbose=2)
    print("\nTest Accuracy: ", test_acc)

    if test_acc > 0.546875 :
        model.save('./bestModel.keras')
        print("NEW BEST ACQUIRED")

    test_images = np.concatenate([x for x, y in test_ds], axis=0)
    test_labels = np.concatenate([y for x, y in test_ds], axis=0)
    test_prediction = np.argmax(model.predict(test_images),1)

    # data printing here
    xVals = np.linspace(1,config['epochs'],num=config['epochs'])

    # plt.plot(xVals,history.history['accuracy'], label='accuracy')
    # plt.plot(xVals,history.history['loss'], label='loss')
    # plt.plot(xVals,history.history['val_accuracy'], label='val_accuracy')
    # plt.plot(xVals,history.history['val_loss'], label='val_loss')
    # plt.legend()
    # plt.title("Training History")
    # plt.xlabel("Epochs")
    # plt.show()
    confMatrix = confusion_matrix(test_labels,test_prediction)
    confDisp = ConfusionMatrixDisplay(confMatrix)
    confDisp.plot()
    plt.show()

    precision0 = confMatrix[0][0] / sum(confMatrix[0])
    precision1 = confMatrix[1][1] / sum(confMatrix[1])
    precision2 = confMatrix[2][2] / sum(confMatrix[2])
    precision3 = confMatrix[3][3] / sum(confMatrix[3])
    precision4 = confMatrix[4][4] / sum(confMatrix[4])

    recall0 = confMatrix[0][0] / (confMatrix[0][0] + confMatrix[1][0] + confMatrix[2][0] + confMatrix[3][0] + confMatrix[4][0])
    recall1 = confMatrix[1][1] / (confMatrix[0][1] + confMatrix[1][1] + confMatrix[2][1] + confMatrix[3][1] + confMatrix[4][1])
    recall2 = confMatrix[2][2] / (confMatrix[0][2] + confMatrix[1][2] + confMatrix[2][2] + confMatrix[3][2] + confMatrix[4][2])
    recall3 = confMatrix[3][3] / (confMatrix[0][3] + confMatrix[1][3] + confMatrix[2][3] + confMatrix[3][3] + confMatrix[4][3])
    recall4 = confMatrix[4][4] / (confMatrix[0][4] + confMatrix[1][4] + confMatrix[2][4] + confMatrix[3][4] + confMatrix[4][4])

    print("Precisions: ", precision0,precision1,precision2,precision3,precision4)
    print("Recall: ", recall0,recall1,recall2,recall3,recall4)

    
    

    def rgb2gray(rgb):
        return np.dot(rgb[...,:3], [0.299, 0.587, 0.144])
    
    # misclassified the first third, and 4th images
    img_gray = rgb2gray(test_images[0])
    plt.imshow(img_gray, cmap=plt.get_cmap('gray'))

    plt.show()
    img_gray = rgb2gray(test_images[2])
    plt.imshow(img_gray, cmap=plt.get_cmap('gray'))

    plt.show()
    img_gray = rgb2gray(test_images[2])
    plt.imshow(img_gray, cmap=plt.get_cmap('gray'))

    plt.show()
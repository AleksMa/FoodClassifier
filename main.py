import random

from imutils import paths
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
import tensorflow as tf
from tensorflow.python.framework.ops import disable_eager_execution

from config import *
from preprocess import *
from dataload import *
from resnet import *


def visual(X_train, Y_train):
    index = np.random.choice(np.arange(len(X_train)), 24, replace=False)
    _, axes = plt.subplots(nrows=4, ncols=6, figsize=(16, 9))
    for item in zip(axes.ravel(), X_train[index], Y_train[index]):
        axes, image, target = item
        axes.imshow(image)
        axes.set_xticks([])
        axes.set_yticks([])
        axes.set_title(target)
    plt.show()


def incorrect(X_test, Y_test, pred):
    food = ('Bread', 'Dessert', 'Meat', 'Soup')
    incorrect_predictions = []
    for i, (p, e) in enumerate(zip(pred, Y_test)):
        predicted, expected = np.argmax(p), np.argmax(e)
        if predicted != expected:
            incorrect_predictions.append((i, X_test[i], predicted, expected))
    print(len(incorrect_predictions))

    _, axes = plt.subplots(nrows=4, ncols=6, figsize=(16, 12))
    for item in zip(axes.ravel(), incorrect_predictions):
        axes, inc_pred = item
        axes.imshow(inc_pred[1])
        axes.set_xticks([])
        axes.set_yticks([])
        axes.set_title(f'p: {food[inc_pred[2]]}; e: {food[inc_pred[3]]}')
    plt.show()

    confusion = tf.math.confusion_matrix(Y_test.argmax(axis=1), pred.argmax(axis=1))
    print(confusion)


def visual_incorrect(Epochs, Hist):
    N = np.arange(0, Epochs)
    plt.style.use("ggplot")
    plt.figure()

    plt.plot(N, Hist.history["loss"], label="train_loss")
    plt.plot(N, Hist.history["val_loss"], label="val_loss")
    plt.plot(N, Hist.history["accuracy"], label="train_acc")
    plt.plot(N, Hist.history["val_accuracy"], label="val_acc")

    plt.title("Training Loss and Accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend()
    plt.show()


def main():
    disable_eager_execution()

    imagePaths = list(paths.list_images(DATASET_PATH))
    random.seed(42)
    random.shuffle(imagePaths)

    sp = Preprocessor(SIZE, SIZE)
    dsl = DatasetLoader(preprocessors=[sp])

    (data, labels) = dsl.load(imagePaths)
    data = data.astype('float32') / 255
    (trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.20, random_state=256)
    visual(trainX, trainY)

    lb = LabelBinarizer()
    trainY = lb.fit_transform(trainY)
    testY = lb.transform(testY)
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(rotation_range=60,
                                                            width_shift_range=0.4,
                                                            height_shift_range=0.2,
                                                            zoom_range=0.2,
                                                            horizontal_flip=True)

    model = ResnetBuilder.build_resnet((3, SIZE, SIZE), 4)
    model.summary()
    model.compile(loss="categorical_crossentropy", optimizer='adam', metrics=["accuracy"])

    H = model.fit(datagen.flow(trainX, trainY),
                epochs=EPOCHS,
                validation_data=(testX, testY),
                verbose=1)
    model.save_weights(MODEL_PATH)

    results = model.evaluate(testX, testY)
    print(results)

    predictions = model.predict(testX)
    for index, probability in enumerate(predictions[0]):
        print(f'{index}:{probability:.10%}')
    incorrect(testX, testY, predictions)
    visual_incorrect(EPOCHS, H)


if __name__ == '__main__':
    main()

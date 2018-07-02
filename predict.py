import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import model
from PIL import Image

def plot_image(image, trueLabel=None, predictedLabel=None):
    # add title with labels if given
    title = ''
    if trueLabel != None:
        title += 'True: {0}'.format(trueLabel)
    if predictedLabel != None:
        title += '\nPredicted: {0}'.format(predictedLabel)

    # plot image with smoothing
    plt.imshow(image, interpolation='bilinear')
    plt.title(title)
    plt.show()

def main(argv):
    # get model and image path
    if len(argv) < 2:
        print('requires image path argument')
        return
    elif len(argv) < 3:
        print('only one argument found: using default model directory')
        model_dir = model.DEFAULT_DIR
        img_dir = argv[1]
    else:
        model_dir = argv[1]
        img_dir = argv[2]

    img = Image.open(img_dir)
    # format data for model
    img_data = np.array(img.getdata(), np.uint8) / 255.0
    img_data = np.asarray(img_data, dtype=np.float32)
    img_data = np.reshape(img_data, [1, 32, 32, 3])

    # build model
    classifier = tf.estimator.Estimator(model_fn=model.model_fn, model_dir=model_dir)

    # input function for prediction
    predict_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={'x': img_data},
        y=None,
        batch_size=1,
        num_epochs=None,
        shuffle=False)

    # get prediction results
    predict_results = classifier.predict(input_fn=predict_input_fn)
    res = next(predict_results)
    predict_label = model.CLASSES[np.argmax(res['probabilities'])]
    # plot image with prediction
    plot_image(np.reshape(img_data, [32, 32, 3]), None, predict_label)

if __name__ == '__main__':
    tf.app.run()

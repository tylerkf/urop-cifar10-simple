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
    if len(argv) < 2:
        print('requires image path argument')
        return

    if len(argv) < 3:
        model_dir = model.DEFAULT_DIR
    else:
        model_dir = argv[2]

    img_dir = argv[1]
    img = Image.open(img_dir)
    # format data for model
    img_data = np.array(img.getdata(), np.uint8) / 255.0
    img_data = np.reshape(img_data, [1, 32, 32, 3])

    # build model
    classifier = tf.estimator.Estimator(model_fn=model.model_fn, model_dir=model_dir)

    predict_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={'x': img_data},
        y=None,
        batch_size=1,
        num_epochs=None,
        shuffle=False)

    predict_results = classifier.predict(input_fn=predict_input_fn)

    res = next(predict_results)
    predict_label = model.CLASSES[np.argmax(res['probabilities'])]
    plot_image(np.reshape(img_data, [32, 32, 3]), None, predict_label)

if __name__ == '__main__':
    tf.app.run()

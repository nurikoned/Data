{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "collapsed_sections": [
        "QRJ09uqJ4d3d",
        "EJbyG8Wh47vF"
      ],
      "authorship_tag": "ABX9TyMt3h0Bg/uHtB29BziAd4PP",
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/nurikoned/Data/blob/main/Deep_Learning_Hselper.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "# **FUNCTION TO PLOT LOSS CURVES**"
      ],
      "metadata": {
        "id": "QRJ09uqJ4d3d"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "import matplotlib.pyplot as plt\n",
        "def plot_loss_curves(history):\n",
        "  loss = history.history['loss']\n",
        "  val_loss = history.history['val_loss']\n",
        "  accuracy = history.history['accuracy']\n",
        "  val_accuracy = history.history['val_accuracy']\n",
        "\n",
        "  epochs = range(len(history.history['loss']))\n",
        "\n",
        "  plt.plot(epochs, loss, label='training_loss')\n",
        "  plt.plot(epochs, val_loss, label='val_loss')\n",
        "  plt.title('Loss')\n",
        "  plt.xlabel('Epochs')\n",
        "  plt.legend()\n",
        "\n",
        "  plt.figure()\n",
        "  plt.plot(epochs, accuracy, label='training_accuracy')\n",
        "  plt.plot(epochs, val_accuracy, label='val_accuracy')\n",
        "  plt.title('Accuracy')\n",
        "  plt.xlabel('Epochs')\n",
        "  plt.legend()"
      ],
      "metadata": {
        "id": "7ypzKT5w4jab"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "# **FUNCTION TO CREATE TL MODEL**"
      ],
      "metadata": {
        "id": "MqVpFRHG4nlS"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "IMAGE_SHAPE = (224,224)\n",
        "from tensorflow.keras import layers\n",
        "import tensorflow_hub as hub\n",
        "import tf_keras as keras\n",
        "def create_model(model_url, num_classes=10):\n",
        "  # Download the pretrained model and save it as a Keras layer\n",
        "  feature_extractor_layer = hub.KerasLayer(model_url,\n",
        "                                           trainable=False, # freeze the underlying patterns\n",
        "                                           name='feature_extraction_layer',\n",
        "                                           input_shape=IMAGE_SHAPE+(3,)) # define the input image shape\n",
        "\n",
        "  # Create our own model\n",
        "  model = keras.Sequential([\n",
        "    feature_extractor_layer, # use the feature extraction layer as the base\n",
        "    keras.layers.Dense(num_classes, activation='softmax', name='output_layer') # create our own output layer\n",
        "  ])\n",
        "\n",
        "  return model"
      ],
      "metadata": {
        "id": "5bFmLbLv4r3S"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "# **FUNCTION TO PLOT RANDOM IMAGE ON CNN**"
      ],
      "metadata": {
        "id": "EJbyG8Wh47vF"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "import matplotlib.pyplot as plt\n",
        "import matplotlib.image as mpimg\n",
        "import random\n",
        "\n",
        "def view_random_image(target_dir, target_class):\n",
        "  target_folder = target_dir + target_class\n",
        "  random_image = random.sample(os.listdir(target_folder), 1)\n",
        "  print(random_image)\n",
        "\n",
        "  img = mpimg.imread(target_folder + \"/\" + random_image[0])\n",
        "  plt.imshow(img)\n",
        "  plt.title(target_class)\n",
        "  plt.axis(\"off\")\n",
        "\n",
        "  print(f\"Image shape: {img.shape}\")\n",
        "\n",
        "  return img"
      ],
      "metadata": {
        "id": "ivjUrT6a4-8d"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "# **MAKE CONFUSION MATRIX**"
      ],
      "metadata": {
        "id": "ZF9v3soLh_bY"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "import numpy as np\n",
        "import matplotlib.pyplot as plt\n",
        "from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix\n",
        "import tensorflow as tf  # Ensure you have TensorFlow imported for tf.round()\n",
        "\n",
        "def plot_confusion_matrix(y_true, y_pred, normalize=True, figsize=(10, 10), cmap='Blues', title=\"Confusion Matrix\"):\n",
        "    \"\"\"\n",
        "    Plots a confusion matrix for the given true and predicted labels.\n",
        "\n",
        "    Args:\n",
        "    - y_true: Ground truth labels.\n",
        "    - y_pred: Predicted labels.\n",
        "    - normalize: Whether to normalize the confusion matrix (default=True).\n",
        "    - figsize: Tuple for figure size (default=(10, 10)).\n",
        "    - cmap: Colormap for the matrix (default='Blues').\n",
        "    - title: Title for the plot (default=\"Confusion Matrix\").\n",
        "    \"\"\"\n",
        "    # Create confusion matrix\n",
        "    cm = confusion_matrix(y_true, tf.round(y_pred))\n",
        "\n",
        "    # Normalize the confusion matrix if requested\n",
        "    if normalize:\n",
        "        cm = cm.astype(\"float\") / cm.sum(axis=1)[:, np.newaxis]\n",
        "\n",
        "    # Plot confusion matrix\n",
        "    disp = ConfusionMatrixDisplay(confusion_matrix=cm)\n",
        "    plt.figure(figsize=figsize)\n",
        "    disp.plot(cmap=cmap, values_format=\".2f\" if normalize else \"d\")  # Formatting numbers\n",
        "    plt.title(title)\n",
        "    plt.show()\n",
        "\n",
        "# Example usage:\n",
        "# plot_confusion_matrix(y_test, y_preds, normalize=True)\n"
      ],
      "metadata": {
        "id": "QGmCHoRQiDF5"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [],
      "metadata": {
        "id": "S14eRynh5AQc"
      }
    }
  ]
}
"""Train and evaluate one of the three vehicle classifiers.

    python scripts/train_classifier.py --model perceptron --epochs 20
    python scripts/train_classifier.py --model cnn --epochs 20
    python scripts/train_classifier.py --model vgg16 --epochs 20

Reproduces the numbers in docs/RESULTS.md. The transfer-learning runs want a
GPU; the perceptron and CNN are fine on CPU.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from tensorflow.keras.callbacks import ModelCheckpoint  # noqa: E402

from roadvision.data import label_to_onehot, load_vehicle_dataset, normalize  # noqa: E402
from roadvision.models import (  # noqa: E402
    build_cnn,
    build_perceptron,
    build_transfer_classifier,
)

BUILDERS = {
    "perceptron": build_perceptron,
    "cnn": build_cnn,
    "vgg16": lambda: build_transfer_classifier("VGG16"),
    "vgg19": lambda: build_transfer_classifier("VGG19"),
    "resnet50": lambda: build_transfer_classifier("ResNet50"),
    "densenet121": lambda: build_transfer_classifier("DenseNet121"),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", choices=sorted(BUILDERS), default="cnn")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--output", default=None,
                        help="Where to save the best checkpoint "
                             "(default: models/<model>.keras)")
    args = parser.parse_args()

    output = args.output or os.path.join("models", f"{args.model}.keras")
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)

    print("Loading the vehicle dataset...")
    (x_train, y_train), (x_test, y_test) = load_vehicle_dataset()
    x_train, x_test = normalize(x_train), normalize(x_test)
    y_train, y_test = label_to_onehot(y_train), label_to_onehot(y_test)
    print(f"  train {x_train.shape}  test {x_test.shape}")

    model = BUILDERS[args.model]()
    model.summary()

    # Keep the weights from the epoch with the lowest validation loss, not the
    # last epoch -- these models overfit well before training ends.
    checkpoint = ModelCheckpoint(output, monitor="val_loss", save_best_only=True,
                                 mode="auto", save_freq="epoch")

    model.fit(
        x_train, y_train,
        epochs=args.epochs,
        validation_data=(x_test, y_test),
        shuffle=True,
        callbacks=[checkpoint],
    )

    loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
    print(f"\nFinal test loss {loss:.4f} | accuracy {accuracy:.4f}")
    print(f"Best checkpoint saved to {output}")


if __name__ == "__main__":
    main()

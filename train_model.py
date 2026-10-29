"""Train a CNN for four-class brain MRI image classification.

Expected dataset structure:
data/
  Training/
    glioma/
    meningioma/
    notumor/
    pituitary/
  Testing/
    glioma/
    meningioma/
    notumor/
    pituitary/

Folder names are normalized and mapped to the four output classes.
"""
from pathlib import Path
import tensorflow as tf

SEED = 42
IMG_SIZE = (150, 150)
BATCH_SIZE = 32
EPOCHS = 12
DATA_DIR = Path("data")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

def find_split(root, candidates):
    for name in candidates:
        p = root / name
        if p.exists():
            return p
    return None

def main():
    train_dir = find_split(DATA_DIR, ["Training", "training", "train"])
    test_dir = find_split(DATA_DIR, ["Testing", "testing", "test"])
    if train_dir is None:
        raise FileNotFoundError(
            "Training folder not found. Put the dataset in data/Training/<class folders>. "
            "See README.md for dataset instructions."
        )

    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir, image_size=IMG_SIZE, batch_size=BATCH_SIZE,
        label_mode="int", validation_split=0.2, subset="training",
        seed=SEED, shuffle=True
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir, image_size=IMG_SIZE, batch_size=BATCH_SIZE,
        label_mode="int", validation_split=0.2, subset="validation",
        seed=SEED, shuffle=True
    )
    class_names = train_ds.class_names
    print("Class order:", class_names)

    # Keep class order in the model metadata so inference can use the same labels.
    test_metrics = None
    if test_dir:
        test_ds = tf.keras.utils.image_dataset_from_directory(
            test_dir, image_size=IMG_SIZE, batch_size=BATCH_SIZE,
            label_mode="int", shuffle=False
        )
        print("Test class order:", test_ds.class_names)
        if test_ds.class_names != class_names:
            print("WARNING: test folder class order differs from training folder.")

    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(AUTOTUNE)
    val_ds = val_ds.prefetch(AUTOTUNE)

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(*IMG_SIZE, 3)),
        tf.keras.layers.Rescaling(1.0 / 255),
        tf.keras.layers.Conv2D(32, 3, activation="relu"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(64, 3, activation="relu"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(128, 3, activation="relu"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(len(class_names), activation="softmax")
    ])
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=3, restore_best_weights=True
        ),
        tf.keras.callbacks.ModelCheckpoint(
            MODEL_DIR / "brain_tumor_cnn.keras",
            monitor="val_loss", save_best_only=True
        )
    ]
    history = model.fit(
        train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=callbacks
    )
    model.save(MODEL_DIR / "brain_tumor_cnn.keras")
    (MODEL_DIR / "class_names.txt").write_text("\n".join(class_names), encoding="utf-8")
    print("Saved model to models/brain_tumor_cnn.keras")

    if test_dir:
        print("Test evaluation:", model.evaluate(test_ds, verbose=1))

    print("Training finished. Use: streamlit run app.py")

if __name__ == "__main__":
    main()

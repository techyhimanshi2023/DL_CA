
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

IMG_SIZE = (150, 150)
BATCH_SIZE = 32

test_ds = tf.keras.utils.image_dataset_from_directory(
    "data/Testing",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

model = tf.keras.models.load_model("models/brain_tumor_cnn.keras")
class_names = test_ds.class_names

y_true = np.concatenate([y.numpy() for _, y in test_ds])
probabilities = model.predict(test_ds)
y_pred = np.argmax(probabilities, axis=1)

print(classification_report(
    y_true, y_pred, target_names=class_names, digits=4
))

cm = confusion_matrix(y_true, y_pred)
display = ConfusionMatrixDisplay(
    confusion_matrix=cm, display_labels=class_names
)
display.plot(cmap="Blues", xticks_rotation=45)
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=200)
plt.show()

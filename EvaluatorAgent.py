import tensorflow as tf
import h5py
import numpy as np
from utilities.Preprocess import preprocess

model_path = "model_full.keras"
model = tf.keras.models.load_model(model_path)
model.summary()

# Preprocess single record a01 (matches preprocessing.py logic)
out = preprocess(r"ecgdata\a01")
record_name = out["record"]
tensors = out["tensors"]
minutes = out["minutes"]

if tensors.shape[0] == 0:
    print(f"No valid windows found for record {record_name}")
else:
    preds = model.predict(tensors)

    print(f"\nRecord: {record_name}")
    print(f"Tensors shape: {tensors.shape}")
    print(f"Minutes evaluated: {len(minutes)}, skipped: {len(out['skipped'])}")

    # Model outputs 2-class softmax: [P(non-apnea), P(apnea)]
    for m, p in zip(minutes, preds):
        p = np.asarray(p).ravel()
        if p.size == 2:
            prob_apnea = float(p[1])
        else:
            prob_apnea = float(p[0]) if p.size == 1 else float(p.mean())
        print(f"Minute {m}: P(apnea)={prob_apnea:.4f}")

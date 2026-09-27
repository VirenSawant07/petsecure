"""Extract per-epoch Keras logs (as printed in the saved notebook outputs) for learning-curve figures."""
import json, re
R = "proj/audio_emotion_recognition_project/models/"
I = "proj/pet_emotion_without_venv/notebooks/"
pat = re.compile(r"accuracy: ([\d.]+) - loss: ([\d.]+) - val_accuracy: ([\d.]+) - val_loss: ([\d.]+)")
out = {}
for tag, nb, cell in [("audio_cnn_first", R + "02_cnn_spectrogram_model.ipynb", 1), ("audio_cnn_adam", R + "02_cnn_spectrogram_model.ipynb", 3),
                      ("audio_cnn_aug", R + "02_cnn_spectrogram_model.ipynb", 5), ("audio_cnn_final", R + "02_cnn_spectrogram_model.ipynb", 6),
                      ("audio_crnn_first", R + "03_crnn_model.ipynb", 1), ("audio_crnn_best", R + "03_crnn_model.ipynb", 3),
                      ("audio_transformer", R + "04_transformer_encoder_model.ipynb", 1), ("image_cnn", I + "02_cnn_training.ipynb", 3),
                      ("image_transfer", I + "03_transfer_learning_models.ipynb", 4)]:
    c = json.load(open(nb, encoding="utf-8"))["cells"][cell]
    txt = re.sub(r"\x1b\[[0-9;]*m", "", "".join("".join(o.get("text", "")) for o in c.get("outputs", [])))
    rows = [tuple(map(float, m.groups())) for m in pat.finditer(txt)]
    out[tag] = dict(acc=[r[0] for r in rows], loss=[r[1] for r in rows], val_acc=[r[2] for r in rows], val_loss=[r[3] for r in rows])
    print(tag, len(rows), "epochs; max val_acc", max(out[tag]["val_acc"]) if rows else None)
json.dump(out, open("results/notebook_logs.json", "w"))

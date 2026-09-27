# Data

The datasets are **not** stored in this repository (size and third-party licensing). To retrain or re-run the evaluation, place them like this:

```
data/
├── images/            3,750 dog photos
│   ├── angry/         1,250
│   ├── happy/         1,250
│   └── sad/           1,250
└── audio/             113 labelled dog vocalizations (mono WAV)
    ├── angry/         33
    ├── happy/         46
    └── sad/           34
```

The app itself does not need this folder: it ships with the trained models in `models/` and a few held-out examples in `examples/`.

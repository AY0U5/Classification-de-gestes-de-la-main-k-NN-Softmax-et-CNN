"""Chargement des modèles entraînés et prédiction sur une image.

Le prétraitement ci-dessous est volontairement IDENTIQUE à celui des notebooks :
  1. image RGB ;
  2. redimensionnement en 32 x 32 avec tf.image.resize (bilinéaire, antialias=True) ;
  3. arrondi en entiers 0..255 ;
  4. division par 255.
"""
import json
from pathlib import Path

import joblib
import numpy as np
from PIL import Image, ImageOps

IMG_SIZE = 32
MODELS_DIR = Path(__file__).parent / "models"
MODEL_NAMES = ["k-NN", "Softmax", "CNN"]


def models_available(models_dir=MODELS_DIR):
    """Retourne la liste des fichiers manquants (liste vide = tout est prêt)."""
    required = ["knn.joblib", "softmax.npz", "cnn_rps.keras"]
    return [f for f in required if not (Path(models_dir) / f).exists()]


def load_artifacts(models_dir=MODELS_DIR):
    """Charge les trois modèles, les noms de classes et les résultats du Test."""
    import tensorflow as tf  # import ici : plus rapide au démarrage si on n'en a pas besoin

    models_dir = Path(models_dir)
    knn = joblib.load(models_dir / "knn.joblib")
    sm = np.load(models_dir / "softmax.npz")
    cnn = tf.keras.models.load_model(models_dir / "cnn_rps.keras")

    results, class_names = {}, ["rock", "paper", "scissors"]
    res_path = models_dir / "resultats_modeles.json"
    if res_path.exists():
        results = json.loads(res_path.read_text(encoding="utf-8"))
        class_names = results.get("_class_names", class_names)

    return {"knn": knn, "W": sm["W"], "b": sm["b"], "cnn": cnn,
            "class_names": class_names, "results": results}


def preprocess(pil_image, center_crop=True):
    """PIL -> tableau float32 (32, 32, 3) dans [0, 1], comme pendant l'entraînement."""
    import tensorflow as tf

    img = ImageOps.exif_transpose(pil_image)          # corrige l'orientation des photos de téléphone
    if img.mode != "RGB":                             # RGBA / niveaux de gris -> RGB
        if img.mode in ("RGBA", "LA", "P"):
            fond = Image.new("RGB", img.size, (255, 255, 255))
            img = img.convert("RGBA")
            fond.paste(img, mask=img.split()[-1])
            img = fond
        else:
            img = img.convert("RGB")

    if center_crop:                                   # les images du dataset sont carrées
        w, h = img.size
        c = min(w, h)
        left, top = (w - c) // 2, (h - c) // 2
        img = img.crop((left, top, left + c, top + c))

    arr = np.asarray(img, dtype=np.uint8)
    small = tf.image.resize(arr, (IMG_SIZE, IMG_SIZE), antialias=True)
    small = tf.cast(tf.round(tf.clip_by_value(small, 0, 255)), tf.uint8).numpy()
    return small.astype("float32") / 255.0


def _softmax(z):
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def predict_all(art, x):
    """x : (32, 32, 3) normalisé. Retourne {nom_modèle: vecteur de 3 probabilités}."""
    x_flat = x.reshape(1, -1)
    probas = {
        "k-NN": art["knn"].predict_proba(x_flat)[0],                       # part des voisins par classe
        "Softmax": _softmax(x_flat[0].astype("float64") @ art["W"] + art["b"]),
        "CNN": art["cnn"].predict(x[None, ...], verbose=0)[0],
    }
    return {k: np.asarray(v, dtype="float64") for k, v in probas.items()}


def nearest_neighbors(art, x):
    """Les k images d'entraînement les plus proches (pour illustrer k-NN). None si indisponible."""
    knn = art["knn"]
    try:
        dist, idx = knn.kneighbors(x.reshape(1, -1))
        imgs = knn._fit_X[idx[0]].reshape(-1, IMG_SIZE, IMG_SIZE, 3)
        labels = np.asarray(knn._y)[idx[0]]
        return imgs, labels, dist[0]
    except Exception:
        return None

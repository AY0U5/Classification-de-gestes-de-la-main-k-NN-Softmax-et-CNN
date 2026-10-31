"""Plateforme web : Pierre – Feuille – Ciseaux (k-NN, Softmax, CNN).

Lancer avec :  streamlit run app.py
"""
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

import inference as inf

st.set_page_config(page_title="Pierre • Feuille • Ciseaux", page_icon="✌️", layout="wide")

EMOJI = {"rock": "✊", "paper": "✋", "scissors": "✌️"}
LABEL_FR = {"rock": "Pierre", "paper": "Feuille", "scissors": "Ciseaux"}


@st.cache_resource(show_spinner="Chargement des modèles…")
def get_artifacts():
    return inf.load_artifacts()


def fr(c):
    return f"{EMOJI.get(c, '')} {LABEL_FR.get(c, c)}"


# ------------------------------------------------------------------ en-tête
st.title("✊ ✋ ✌️  Reconnaissance de gestes de la main")
st.caption("Compare trois modèles entraînés dans le projet : **k-NN**, **classifieur linéaire Softmax** et **CNN**.")

missing = inf.models_available()
if missing:
    st.error("Modèles introuvables : " + ", ".join(missing))
    st.info("Exécute d'abord le **Notebook 3** jusqu'à la cellule 5.4 : elle crée le dossier `models/`. "
            "Copie ce dossier à côté de `app.py`, puis recharge cette page.")
    st.stop()

art = get_artifacts()
class_names = art["class_names"]

# ------------------------------------------------------------------ barre latérale
with st.sidebar:
    st.header("⚙️ Options")
    center_crop = st.checkbox("Recadrer l'image en carré (centre)", value=True,
                              help="Les images d'entraînement sont carrées. Sans recadrage, une photo "
                                   "rectangulaire est déformée lors du redimensionnement.")
    show_neighbors = st.checkbox("Montrer les voisins du k-NN", value=True)
    st.divider()
    st.markdown("**Conseils pour de bons résultats**\n"
                "- une seule main, bien centrée\n"
                "- fond uni, bon éclairage\n"
                "- geste net : poing, main ouverte, deux doigts")

tab_pred, tab_perf, tab_info = st.tabs(["🔍 Prédiction", "📊 Performances (Test)", "ℹ️ Comment ça marche"])

# ------------------------------------------------------------------ onglet prédiction
with tab_pred:
    source = st.radio("Source de l'image", ["📁 Importer un fichier", "📷 Prendre une photo"], horizontal=True)
    if source.startswith("📁"):
        fichier = st.file_uploader("Choisis une image (PNG, JPG, JPEG, WEBP)", type=["png", "jpg", "jpeg", "webp"])
    else:
        fichier = st.camera_input("Prends une photo de ta main")

    if fichier is None:
        st.info("Importe une image pour lancer l'analyse.")
    else:
        try:
            pil = Image.open(fichier)
            pil.load()
        except Exception:
            st.error("Impossible de lire ce fichier : est-ce bien une image valide ?")
            st.stop()

        x = inf.preprocess(pil, center_crop=center_crop)
        probas = inf.predict_all(art, x)

        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Image envoyée")
            st.image(pil, width="stretch")
        with c2:
            st.subheader("Ce que voient les modèles (32 × 32)")
            petite = Image.fromarray((x * 255).round().astype("uint8")).resize((256, 256), Image.NEAREST)
            st.image(petite, caption="Image réduite à 32×32 pixels (agrandie pour l'affichage)")

        st.divider()
        st.subheader("Résultats")
        cols = st.columns(3)
        votes = []
        for col, nom in zip(cols, inf.MODEL_NAMES):
            p = probas[nom]
            k = int(np.argmax(p))
            votes.append(k)
            with col:
                st.markdown(f"### {nom}")
                st.metric("Prédiction", fr(class_names[k]), f"confiance {p[k] * 100:.1f} %", delta_color="off")
                st.bar_chart(pd.DataFrame({"Probabilité": p}, index=[fr(c) for c in class_names]), height=200)
                if p[k] < 0.5:
                    st.caption("⚠️ Confiance faible : le modèle hésite.")

        if len(set(votes)) == 1:
            st.success(f"✅ Les trois modèles sont d'accord : **{fr(class_names[votes[0]])}**")
        else:
            majorite = max(set(votes), key=votes.count)
            if votes.count(majorite) >= 2:
                st.warning(f"⚠️ Désaccord entre les modèles. Majorité : **{fr(class_names[majorite])}**")
            else:
                st.warning("⚠️ Les trois modèles donnent trois réponses différentes.")

        st.caption("ℹ️ Les modèles ont été entraînés sur des images de synthèse (mains sur fond uni, 32×32). "
                   "Une photo réelle avec un fond chargé peut donner des résultats moins fiables.")

        if show_neighbors:
            voisins = inf.nearest_neighbors(art, x)
            if voisins is not None:
                imgs, labels, dist = voisins
                st.subheader("Les voisins les plus proches (k-NN)")
                st.caption("Le k-NN regarde ces images d'entraînement et vote : la classe majoritaire gagne.")
                cs = st.columns(len(imgs))
                for c, im, lab, d in zip(cs, imgs, labels, dist):
                    big = Image.fromarray((im * 255).round().astype("uint8")).resize((128, 128), Image.NEAREST)
                    c.image(big, caption=f"{fr(class_names[int(lab)])}\ndistance {d:.2f}")

# ------------------------------------------------------------------ onglet performances
with tab_perf:
    res = art["results"]
    if not res:
        st.info("Fichier `resultats_modeles.json` absent.")
    else:
        st.subheader("Résultats sur le jeu de Test (180 images)")
        lignes = []
        for nom in inf.MODEL_NAMES:
            r = res.get(nom)
            if r:
                lignes.append({"Modèle": nom,
                               "Accuracy": f"{r['accuracy'] * 100:.1f} %",
                               "F1 macro": f"{r['f1_macro']:.3f}",
                               "Entraînement (s)": f"{r['fit_s']:.2f}",
                               "Prédiction (s)": f"{r['predict_s']:.4f}"})
        st.table(lignes)
        st.bar_chart(pd.DataFrame({"accuracy": [res[n]["accuracy"] for n in inf.MODEL_NAMES if n in res]},
                                 index=[n for n in inf.MODEL_NAMES if n in res]))
        cfg = res.get("_config", {})
        if cfg:
            with st.expander("Hyperparamètres retenus"):
                st.json(cfg)

# ------------------------------------------------------------------ onglet explications
with tab_info:
    st.markdown("""
**Pipeline appliqué à ton image** (le même que pendant l'entraînement) :
1. conversion en RGB (et recadrage carré optionnel) ;
2. redimensionnement en **32 × 32** ;
3. normalisation : pixels divisés par **255** (valeurs entre 0 et 1) ;
4. puis selon le modèle :
   - **k-NN** et **Softmax** reçoivent l'image **aplatie** en vecteur de 3072 valeurs ;
   - le **CNN** reçoit l'image en **(32, 32, 3)**.

**Les trois modèles**
- **k-NN** : compare ton image aux 540 images d'entraînement et vote parmi les plus proches.
  *Sa « confiance » est la part de voisins qui votent pour la classe.*
- **Softmax** : un score linéaire par classe (`x·W + b`), transformé en probabilités.
- **CNN** : des filtres convolutifs apprennent les formes locales (doigts, contours).

**À savoir** : une « confiance » élevée ne garantit pas que la réponse est juste, surtout pour une image
très différente de celles d'entraînement.
""")

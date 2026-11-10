# Classification de gestes de la main : k-NN, Softmax et CNN

Projet universitaire de Machine Learning : comparaison de trois approches pour reconnaître les gestes **Pierre – Feuille – Ciseaux** à partir d'images, suivie d'une **application web** qui permet de tester les modèles sur ses propres images.

| Modèle | Idée | Entrée |
| --- | --- | --- |
| **k-NN** | vote des k images d'entraînement les plus proches | image aplatie `(3072,)` |
| **Classifieur linéaire Softmax** | un score linéaire par classe, écrit en NumPy | image aplatie `(3072,)` |
| **CNN** | filtres convolutifs appris sur l'image | tenseur `(32, 32, 3)` |

---

## Aperçu de l'application

<!-- Ajoute ici une capture d'écran : enregistre-la dans docs/apercu.png -->
![Aperçu de l'application](docs/apercu.png)

---

## Données

- **Dataset** : [Rock Paper Scissors](https://www.tensorflow.org/datasets/catalog/rock_paper_scissors) (TensorFlow Datasets), de Laurence Moroney, licence CC BY 2.0.
- **Classes** : `rock`, `paper`, `scissors`.
- **Protocole imposé par le projet** :
  - 300 images par classe, soit 900 images au total (sélection aléatoire, `SEED = 42`) ;
  - images RGB redimensionnées en **32 × 32** ;
  - pixels normalisés entre 0 et 1 (division par 255) ;
  - découpage **stratifié 60 % / 20 % / 20 %** : 540 Train, 180 Validation, 180 Test ;
  - **mêmes partitions** pour les trois modèles.
- Le jeu de Test n'est utilisé **qu'une seule fois**, pour l'évaluation finale. Les hyperparamètres sont choisis sur la Validation.

---

## Structure du dépôt

```
.
├── notebooks/
│   ├── 01_exploration_preparation.ipynb          # téléchargement, exploration, split
│   ├── 02_representation_preparation_modeles.ipynb  # flattening, one-hot, normalisation
│   └── 03_entrainement_comparaison_modeles.ipynb    # entraînement et évaluation finale
├── app/
│   ├── app.py              # interface web (Streamlit)
│   ├── inference.py        # chargement des modèles, prétraitement, prédiction
│   ├── requirements.txt
│   └── models/             # modèles entraînés (créés par le notebook 3)
├── README.md
└── .gitignore
```

---

## Installation

**Prérequis** : Python 3.10 à 3.12 (TensorFlow ne s'installe pas sur toutes les versions).

```bash
git clone https://github.com/<ton-compte>/<ton-depot>.git
cd <ton-depot>

python -m venv .venv
source .venv/bin/activate        # Windows : .venv\Scripts\activate

pip install -r app/requirements.txt
pip install jupyter              # pour exécuter les notebooks
```

> Sous macOS, le paquet à installer est `tensorflow` (le paquet `tensorflow-cpu` n'existe que pour Linux et Windows).

---

## Reproduire les expériences

1. Ouvre les notebooks dans l'ordre : `jupyter notebook notebooks/`.
2. Le **notebook 3** entraîne les trois modèles et, à la cellule 5.4, crée un dossier `models/` contenant :
   - `knn.joblib`
   - `softmax.npz`
   - `cnn_rps.keras`
   - `resultats_modeles.json`
3. Copie ce dossier dans `app/` (à côté de `app.py`).

Les résultats sont reproductibles grâce à la graine fixe `SEED = 42`. De petites différences peuvent apparaître selon la version de TensorFlow et le matériel.

---

## Lancer l'application web

```bash
cd app
streamlit run app.py
```

L'application s'ouvre sur <http://localhost:8501>. Elle permet de :

- envoyer une image (PNG, JPG, WEBP) ou prendre une photo avec la caméra ;
- voir l'image telle que les modèles la perçoivent (32 × 32) ;
- comparer la prédiction, la confiance et les probabilités des trois modèles ;
- voir les images d'entraînement les plus proches (voisins du k-NN) ;
- consulter les performances sur le jeu de Test.

Le prétraitement de l'application est **identique** à celui des notebooks : RGB → 32 × 32 (`tf.image.resize`, antialias) → arrondi → division par 255.

---

## Résultats

Résultats sur le **jeu de Test** (180 images). *À compléter avec tes valeurs de `resultats_modeles.json`.*

| Modèle | Accuracy | F1 macro | Entraînement (s) | Prédiction (s) |
| --- | --- | --- | --- | --- |
| k-NN | … | … | … | … |
| Softmax | … | … | … | … |
| CNN | … | … | … | … |

**Hyperparamètres retenus (choisis sur la Validation)** : k-NN : k = …, distance = … ; Softmax : taux d'apprentissage = …, régularisation = … ; CNN : … époques (arrêt anticipé).

### Discussion

*À rédiger : quel modèle est le meilleur, quelles classes sont confondues, ce que montrent les poids du Softmax, le compromis performance / temps de calcul.*

---

## Limites

- **Petit jeu de données** (900 images) et **faible résolution** (32 × 32).
- **Un seul découpage** Train / Validation / Test : avec 180 images de test, un écart de 1 à 2 points correspond à 2 ou 3 images et n'est pas forcément significatif.
- Les modèles ont appris sur des **images de synthèse** (mains générées sur fond uni). Les photos réelles avec un fond chargé peuvent être mal classées.
- La « confiance » affichée n'est pas une garantie de justesse, surtout pour une image très différente de celles d'entraînement. Pour le k-NN, c'est la part de voisins qui votent pour la classe.

---

## Pistes d'amélioration

- Augmentation de données pour le CNN (retournements, petites rotations).
- Validation croisée et plusieurs graines pour donner une moyenne ± écart-type.
- Autres architectures de CNN, ou résolution plus élevée.

---

## Références

- Dataset : L. Moroney, *Rock Paper Scissors Dataset*, licence [CC BY 2.0](https://creativecommons.org/licenses/by/2.0/) — [fiche TensorFlow Datasets](https://www.tensorflow.org/datasets/catalog/rock_paper_scissors).
- Bibliothèques : NumPy, scikit-learn, TensorFlow / Keras, Streamlit, Matplotlib.

## Auteur

Mahdaoui Ayoub — Master M2SI / INSEA — 2026
Bilal El Othmani — Master M2SI / INSEA — 2026
Yahya Ghallali — Master M2SI / INSEA — 2026

## Licence

Code distribué sous licence MIT (voir `LICENSE`). Le dataset reste soumis à sa propre licence (CC BY 2.0).

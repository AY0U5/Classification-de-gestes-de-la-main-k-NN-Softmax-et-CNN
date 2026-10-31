# Plateforme Pierre – Feuille – Ciseaux

Application web (Streamlit) qui applique les trois modèles du projet (k-NN, Softmax, CNN) à une image envoyée par l'utilisateur.

## Structure attendue

```
plateforme_rps/
├── app.py              # interface web
├── inference.py        # chargement des modèles + prétraitement + prédiction
├── requirements.txt
└── models/             # créé par le Notebook 3 (cellule 5.4)
    ├── knn.joblib
    ├── softmax.npz
    ├── cnn_rps.keras
    └── resultats_modeles.json
```

## Lancer en local

1. Exécute le **Notebook 3** jusqu'à la cellule **5.4**, qui crée le dossier `models/`.
2. Copie ce dossier `models/` à côté de `app.py`.
3. Installe les dépendances (idéalement dans un environnement virtuel) :
   ```
   pip install -r requirements.txt
   ```
4. Démarre l'application :
   ```
   streamlit run app.py
   ```
   Elle s'ouvre sur http://localhost:8501.

## Important : même prétraitement qu'à l'entraînement
`inference.preprocess` reproduit exactement les notebooks : RGB → 32×32 (`tf.image.resize`, antialias) → arrondi → division par 255.
Si tu changes le prétraitement dans les notebooks, change-le aussi ici.

## Limites à connaître
- Les modèles ont appris sur des images de synthèse (mains sur fond uni, 32×32). Les photos réelles avec un fond chargé peuvent être mal classées.
- La confiance affichée n'est pas une garantie : pour k-NN, c'est la part de voisins qui votent pour la classe.

## Mettre en ligne (optionnel)
Streamlit Community Cloud ou Hugging Face Spaces (type « Streamlit ») : envoie `app.py`, `inference.py`, `requirements.txt` et le dossier `models/` (le fichier `cnn_rps.keras` est petit, quelques Mo).

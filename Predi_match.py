import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# --- Chargement et nettoyage ---
df = pd.read_csv('final_dataset.csv')

colonnes_form = ['HM1','HM2','HM3','HM4','HM5','AM1','AM2','AM3','AM4','AM5']
colonnes_banni = ['FTHG','FTAG','Date','HomeTeam','AwayTeam','HTFormPtsStr','ATFormPtsStr','Unnamed: 0']

for col in colonnes_form:
    df[col] = df[col].map({'W': 3, 'D': 1, 'L': 0, 'M': 0})

df = df.drop(columns=colonnes_banni)

y = df['FTR']
X = df.drop(columns='FTR')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Random Forest de base ---
rf_base = RandomForestClassifier(n_estimators=500, random_state=42)
rf_base.fit(X_train, y_train)
y_pred_rf = rf_base.predict(X_test)
print(f"RF base accuracy : {accuracy_score(y_test, y_pred_rf):.4f}")

# --- GridSearchCV sur Random Forest ---
param_grid = {
    'n_estimators': [100, 300, 500],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)
grid_search.fit(X_train, y_train)

print(f"\nMeilleurs paramètres : {grid_search.best_params_}")
print(f"Meilleur score CV   : {grid_search.best_score_:.4f}")

rf_best = grid_search.best_estimator_
y_pred_best = rf_best.predict(X_test)
print(f"RF optimisé accuracy : {accuracy_score(y_test, y_pred_best):.4f}")

# --- Comparaison de modèles ---
modeles = {
    'Random Forest optimisé': rf_best,
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=300, random_state=42),
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42)
}

print("\n--- Comparaison des modèles ---")
for nom, modele in modeles.items():
    modele.fit(X_train, y_train)
    y_pred = modele.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    cv = cross_val_score(modele, X, y, cv=5).mean()
    print(f"{nom:30s} | Accuracy: {acc:.4f} | CV moyen: {cv:.4f}")

# --- Rapport complet sur le meilleur modèle ---
print("\n--- Rapport de classification (RF optimisé) ---")
print(classification_report(y_test, y_pred_best))

# --- Matrice de confusion ---
fig, ax = plt.subplots(figsize=(6, 5))
ConfusionMatrixDisplay.from_predictions(y_test, y_pred_best, ax=ax, colorbar=False)
ax.set_title("Matrice de confusion — Random Forest optimisé")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
plt.show()
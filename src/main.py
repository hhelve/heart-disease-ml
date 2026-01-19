import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

df = pd.read_csv("data/Heart_Disease_Prediction.csv")
df.columns = df.columns.str.replace(" ", "_")

# target binário correto
y = df["Heart_Disease"].map({"Absence": 0, "Presence": 1})
X = df.drop("Heart_Disease", axis=1)

colunas_categoricas = [
    "Sex",
    "Chest_pain_type",
    "FBS_over_120",
    "EKG_results",
    "Exercise_angina",
    "Slope_of_ST",
    "Thallium",
    "Number_of_vessels_fluro",
]

colunas_numericas = [
    "Age",
    "BP",
    "Cholesterol",
    "Max_HR",
    "ST_depression",
]

preprocessador = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), colunas_categoricas),
        ("num", StandardScaler(), colunas_numericas),
    ]
)

X_treino, X_teste, y_treino, y_teste = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

melhor_k = None
melhor_recall = 0

for k in range(3, 21, 2):
    pipeline = Pipeline([
        ("preprocessamento", preprocessador),
        ("modelo", KNeighborsClassifier(n_neighbors=k))
    ])

    pipeline.fit(X_treino, y_treino)
    preds = pipeline.predict(X_teste)

    report = classification_report(y_teste, preds, output_dict=True)
    recall_presence = report["1"]["recall"]

    if recall_presence > melhor_recall:
        melhor_recall = recall_presence
        melhor_k = k

pipeline_knn = Pipeline([
    ("preprocessamento", preprocessador),
    ("modelo", KNeighborsClassifier(n_neighbors=melhor_k))
])

pipeline_knn.fit(X_treino, y_treino)

THRESHOLD = 0.4

proba = pipeline_knn.predict_proba(X_teste)[:, 1]
pred_custom = (proba >= THRESHOLD).astype(int)

acc = accuracy_score(y_teste, pred_custom)
print(acc)

print(classification_report(y_teste, pred_custom))
print(confusion_matrix(y_teste, pred_custom))

age = int(input("Idade: "))
sex = int(input("Sexo (0=feminino, 1=masculino): "))
chest_pain = int(input("Tipo de dor no peito (1–4): "))
bp = int(input("Pressão arterial (mmHg): "))
chol = int(input("Colesterol: "))
fbs = int(input("Glicemia > 120 (0=nao, 1=sim): "))
ekg = int(input("EKG (0–2): "))
max_hr = int(input("Frequência cardíaca máxima: "))
ex_angina = int(input("Angina por exercício (0=nao, 1=sim): "))
st_dep = float(input("Depressão ST: "))
slope = int(input("Inclinação ST (0–2): "))
vessels = int(input("Nº vasos (0–3): "))
thallium = int(input("Thallium (0–3): "))

novo_paciente = {
    "Age": age,
    "Sex": sex,
    "Chest_pain_type": chest_pain,
    "BP": bp,
    "Cholesterol": chol,
    "FBS_over_120": fbs,
    "EKG_results": ekg,
    "Max_HR": max_hr,
    "Exercise_angina": ex_angina,
    "ST_depression": st_dep,
    "Slope_of_ST": slope,
    "Number_of_vessels_fluro": vessels,
    "Thallium": thallium,
}

novo_paciente_df = pd.DataFrame([novo_paciente])

proba_novo = pipeline_knn.predict_proba(novo_paciente_df)[0][1]
print(proba_novo * 100)
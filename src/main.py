import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report


df = pd.read_csv("../data/Heart_Disease_Prediction.csv")


df.columns = df.columns.str.replace(" ", "_")

y = df["Heart_Disease"]
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
    X, y, test_size=0.3, random_state=42
)

pipeline_knn = Pipeline(
    [
        ("preprocessamento", preprocessador),
        ("modelo", KNeighborsClassifier(n_neighbors=5)),
    ]
)


pipeline_knn.fit(X_treino, y_treino)
pred_knn = pipeline_knn.predict(X_teste)
acc = accuracy_score(y_teste, pred_knn) * 100
print(f"Knn accuracy: {acc:.2f}%")

age = int(input("Idade: "))
sex = int(input("Sexo (0=feminino, 1=masculino): "))
chest_pain = int(
    input(
        "Tipo de dor no peito (1 = Typical angina, 2 = Atypical angina, 3 = Non-anginal pain, 4 = Asymptomatic): "
    )
)
bp = int(input("Pressão arterial (mmHg): "))
chol = int(input("Colesterol: "))
fbs = int(input("Glicemia > 120 (0=nao, 1=sim): "))
ekg = int(
    input(
        "Resultado do EKG (0 = Normal, 1 = ST-T wave abnormality, 2 = Left ventricular hypertrophy): "
    )
)
max_hr = int(input("Frequência cardíaca máxima: "))
ex_angina = int(input("Angina por exercício (0=nao, 1=sim): "))
st_dep = float(input("Depressão ST: "))
slope = int(input("Inclinação ST (0–2): "))
vessels = int(input("Nº de vasos fluoroscopia (0–3): "))
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

predicao_novo_paciente = pipeline_knn.predict(novo_paciente_df)
predicao_novo_paciente = pipeline_knn.predict(novo_paciente_df)
proba = pipeline_knn.predict_proba(novo_paciente_df)

prob_doenca_pct = proba[0][1] * 100

print(f"Probabilidade de DOENÇA CARDÍACA: {prob_doenca_pct:.2f}%")

if prob_doenca_pct >= 70:
    print("Classificação: ALTO RISCO")
elif prob_doenca_pct >= 40:
    print("Classificação: RISCO MODERADO")
else:
    print("Classificação: BAIXO RISCO")

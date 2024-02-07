import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

st.title("Clasificador de Pinguinos")
st.write(
    "Esta app te ayuda a descubrir a qué especie pertenece un pingüino basándose en 6 características simples que proporcionas."
)

st.write(
    "Funciona de manera sencilla: introduces los datos del pingüino, y la app utilizará un modelo de ML pre-entrenado para predecir su especie. \
 Todo el proceso, desde el análisis hasta la predicción, se realiza dentro de la app, sin necesidad de conocimientos previos en el tema."
)

st.markdown("**¿Quieres entrenar tu propio modelo de ML?** :sparkles: 	:muscle:")

st.write(
    "En esta version se incluye la opción de cargar tu propio dataset y asi entrenar tu propio modelo de Machine Learning."
)

st.write("*Pd: Si no subes ningún dataset, la app utilizará un modelo pre-entrenado.*")

# El usuario sube su propio dataset para entrenar su modelo con esos datos,
# Si no sube un archivo, se utilizará un modelo ya entrenado con RandomForest.
penguin_file = st.file_uploader("Sube tu propio dataset de pinguinos")

if penguin_file is None:
    rf_pickle = open("random_forest_penguin.pickle", "rb")  # Se abre el archivo pickle
    map_pickle = open("output_penguin.pickle", "rb")
    rfc = pickle.load(
        rf_pickle
    )  # Se carga el archivo pickle a Python. Ahora es un modelo usable
    unique_penguin_mapping = pickle.load(map_pickle)
    rf_pickle.close()
    map_pickle.close()
    penguin_df = pd.read_csv("penguins.csv")

# Pre-procesamiento de los datos y entrenamiento del modelo
# Para esto vamos a utilizar el codigo ya creado en la APP previa (que utiliza el modelo de ML ya creado)
else:
    penguin_df = pd.read_csv("penguins.csv")
    penguin_df = penguin_df.dropna()
    output = penguin_df["species"]
    features = penguin_df[
        [
            "island",
            "bill_length_mm",
            "bill_depth_mm",
            "flipper_length_mm",
            "body_mass_g",
            "sex",
        ]
    ]
    features = pd.get_dummies(features)
    output, unique_penguin_mapping = pd.factorize(output)
    X_train, X_test, y_train, y_test = train_test_split(features, output, test_size=0.8)
    rfc = RandomForestClassifier(random_state=15)
    rfc.fit(X_train, y_train)
    y_pred = rfc.predict(X_test)
    score = accuracy_score(y_pred, y_test).round(2)
    st.write(
        f"Se entrenó un modelo RandomForest con estos datos, y tiene un score de {score}!\
        Utiliza los inputs de abajo para poner a prueba tu modelo :)"
    )

# Ahora necesitamos obtener las entradas (inputs) del usuario para realizar predicciones.
# En este punto vamos a realizar una mejora, ya que cada vez que el usuario agrega/cambia un input
# en la app, TODA la app de Streamlit se vuelve a ejecutar lo que produce un consumo excesivo de recursos.
# Para solucionar estyo, se utilizaran las funciones st.form() y st.submit_form_button()
# esto para "envolver" los inputs de nuestro usuario y permitir que el usuario cambie todas las entradas
# y envíe todo el formulario de una vez, en lugar de varias veces.

with st.form("user_inputs"):
    island = st.selectbox("Penguin Island", options=["Biscoe", "Dream", "Torgersen"])
    sex = st.selectbox("Sex", options=["Female", "Male"])
    bill_length = st.number_input("Bill Length (mm)", min_value=0)
    bill_depth = st.number_input("Bill Depth (mm)", min_value=0)
    flipper_length = st.number_input("Flipper Length (mm)", min_value=0)
    body_mass = st.number_input("Body Mass (g)", min_value=0)
    st.form_submit_button()

island_biscoe, island_dream, island_torgersen = 0, 0, 0
if island == "Biscoe":
    island_biscoe = 1
elif island == "Dream":
    island_dream = 1
elif island == "Torgersen":
    island_torgersen = 1
sex_female, sex_male = 0, 0
if sex == "Female":
    sex_female = 1
elif sex == "Male":
    sex_male = 1

new_prediction = rfc.predict(
    [
        [
            bill_length,
            bill_depth,
            flipper_length,
            body_mass,
            island_biscoe,
            island_dream,
            island_torgersen,
            sex_female,
            sex_male,
        ]
    ]
)
prediction_species = unique_penguin_mapping[new_prediction][0]
st.write(
    f"El modelo predice que tu pinguino pertenece a la especie {prediction_species}"
)
st.image("feature_importance.png")


fig, ax = plt.subplots()
ax = sns.displot(x=penguin_df["bill_length_mm"], hue=penguin_df["species"])
plt.axvline(bill_length)
plt.title("Bill Length by Species")
st.pyplot(ax)

fig, ax = plt.subplots()
ax = sns.displot(x=penguin_df["bill_depth_mm"], hue=penguin_df["species"])
plt.axvline(bill_depth)
plt.title("Bill Depth by Species")
st.pyplot(ax)

fig, ax = plt.subplots()
ax = sns.displot(x=penguin_df["flipper_length_mm"], hue=penguin_df["species"])
plt.axvline(flipper_length)
plt.title("Flipper Length by Species")
st.pyplot(ax)

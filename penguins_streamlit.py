# Este archivo tiene como función cargar el modelo y la función de mapeo.
import streamlit as st
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

st.title("Clasificador de Pinguinos")
st.write(
    "Esta app utiliza 6 inputs para predecir la especie del pinguino usando un modelo de ML construido del dataset de Palmer Penguins.\
        Usa el formulario de abajo para saber a que especie pertenece el pinguino."
)


rf_pickle = open(
    "random_forest_penguin.pickle", "rb"
)  # "rb" ya que vamos a leer bytes, no escribir
map_pickle = open("output_penguin.pickle", "rb")
rfc = pickle.load(rf_pickle)
unique_penguin_mapping = pickle.load(map_pickle)
rf_pickle.close()
map_pickle.close()

# Ya tenemos el Modelo y el Mapeo cargado en la App web.
# Ahora vamos a introduccion la funcionalidad que el usuario ingrese datos via web y el modelo haga predicciones
# Recordar que el modelo requeire las siguientes entradas para funcionar.
#           ["island", "bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g", "sex"]
####
# Para "island" y "sex" ya conocemos las opciones disponibles, por lo cual utilizaremos --->  st.selectbox()
# Para los otros datos solo tenemos que asegurarnos que el usuario ingrese números positivos y que el minimo sea cero, por lo que usaremos --->  st.number_input()

island = st.selectbox("Isla del pinguino", options=["Biscoe", "Dream", "Torgersen"])
sex = st.selectbox("Sexo", options=["Female", "Male"])

bill_length = st.number_input("Largo del pico (mm)", min_value=0)
bill_depth = st.number_input("Profundidad del pico (mm)", min_value=0)
flipper_length = st.number_input("Largo de la aleta", min_value=0)
body_mass = st.number_input("Masa corporal (gr)", min_value=0)

# Como el modelo no tiene una variable sex, sino que tiene dos (sex_demale y sex_male) lo mismo para island, se debe modificar la logica de los inputs
island_biscoe, island_dream, island_torgersen = (
    0,
    0,
    0,
)  # Se inicializan las 3 variables en 0
if island == "Biscoe":
    island_biscoe = 1
elif island == "Dream":
    islan_dream = 1
elif island == "Torgersen":
    island_torgersen = 1

sex_female, sex_male = 0, 0
if sex == "Female":
    sex_female = 1
elif sex == "Male":
    sex_male = 1

# Inputs del usuario
user_inputs = [
    island,
    sex,
    bill_length,
    bill_depth,
    flipper_length,
    body_mass,
]
st.write(f"El usuario ingreso los siguientes datos {user_inputs}")

# Todos los los inputs están en el formato correcto para el modelo de ML
# El ultimo paso es utilizar la función predict() en el modelo y mostrar los resultados.
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
    f"La predicción indica que tu pinguino corresponde a la especie {prediction_species}"
)

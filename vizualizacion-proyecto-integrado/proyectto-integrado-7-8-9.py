#parte 7

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
import plotly.express as px
# Lee los datos procesados
df = pd.read_csv('datos_procesados.csv')

plt.figure(figsize=(10,6))
plt.hist(df['age'], bins=20, color='skyblue')
plt.title('Distribución de Edades')
plt.xlabel('Edad')
plt.ylabel('Frecuencia')
plt.grid(axis='y', alpha=0.75)
plt.show()




# Obtener los valores para hombres y mujeres
valores_hombres = df[df['sex'] == True][['anaemia', 'diabetes', 'smoking', 'DEATH_EVENT']].sum().values
valores_mujeres = df[df['sex'] == False][['anaemia', 'diabetes', 'smoking', 'DEATH_EVENT']].sum().values

# Convertir los valores booleanos a números enteros
valores_hombres = [int(valor) for valor in valores_hombres]
valores_mujeres = [int(valor) for valor in valores_mujeres]

# Imprimir los valores de las variables
print(valores_hombres)
print(valores_mujeres)

# Configurar el gráfico de barras
etiquetas = ['Anemia', 'Diabetes', 'Fumador', 'Muertos']
x = np.arange(len(etiquetas))
ancho = 0.35

fig, ax = plt.subplots(figsize=(10, 6))
rectsTrue = ax.bar(x - ancho/2, valores_hombres, ancho, label='Hombres', color='blue')
rects2 = ax.bar(x + ancho/2, valores_mujeres, ancho, label='Mujeres', color='red')

# Agregar texto, etiquetas y título
ax.set_xlabel('Categorías')
ax.set_ylabel('Cantidad')
ax.set_title('Estadísticas de Salud por Género')
ax.set_xticks(x)
ax.set_xticklabels(etiquetas)
ax.legend()

plt.show()



#parte 8

# Lee los datos procesados
df = pd.read_csv('datos_descargados.csv')

valores_anémicos = df['anaemia'].value_counts()
valores_diabéticos = df['diabetes'].value_counts()
valores_fumadores = df['smoking'].value_counts()
valores_muertos = df['DEATH_EVENT'].value_counts()

print("Valor de la anaemia: ", valores_anémicos)
print("Valor de la diabetes: ", valores_diabéticos)
print("Valor del fumar: ", valores_fumadores)
print("Valor de las muertes: ", valores_muertos)


anemicos = 'si','no'
diabeticos =  'si','no'
fumadores =  'si','no'
muertos =  'si','no'

# Crear las gráficas de torta
fig, axs = plt.subplots(1, 4)
axs[0].pie(valores_anémicos, labels=anemicos, autopct='%1.1f%%', startangle=90)
axs[1].pie(valores_diabéticos, labels=diabeticos, autopct='%1.1f%%', startangle=90)
axs[2].pie(valores_fumadores, labels=fumadores, autopct='%1.1f%%', startangle=90)
axs[3].pie(valores_muertos, labels=muertos, autopct='%1.1f%%', startangle=90)

# Ajustar los títulos de las gráficas
axs[0].set_title('anemicos')
axs[1].set_title('Diabeticos')
axs[2].set_title('Fumadores')
axs[3].set_title('Muerte')

# Mostrar la gráfica
plt.show()


#parte 9

df = pd.read_csv("datos_procesados1.csv")

dfnew = df.drop(columns=["DEATH_EVENT", "categoria_edad"])

X = dfnew.values

np.savetxt("X.csv", X, delimiter=",")

y = df["DEATH_EVENT"]

y = y.ravel()

np.savetxt("y.csv", y, delimiter=",")

X_embedded = TSNE(
    n_components=3,
    learning_rate='auto',
    init='random',
    perplexity=3
).fit_transform(X)



y1 =df["DEATH_EVENT"].replace({1: "Fallecido", 0: "vivo"})
print(y1)

fig = px.scatter_3d(
    x=X_embedded[:, 0],
    y=X_embedded[:, 1],
    z=X_embedded[:, 2],
    color=y1,
    color_continuous_scale='inferno',
    opacity=0.8,
)

fig.show()


# Cargamos las dos tablas de datos
perfiles = pd.read_parquet('data/raw/flex_perfiles_usuario.parquet')
sesiones = pd.read_parquet('data/raw/flex_sesiones.parquet')


# Filtro en perfiles valores erroneos peso y altura. Corrijo formatos
perfiles.loc[perfiles["posicion"]=="manual","posicion"] = "Manual"

for col in ["altura", "peso"]:
    perfiles[col] = perfiles[col].str.replace(',', '.')

perfiles[["altura", "peso"]] = perfiles[["altura", "peso"]].apply(pd.to_numeric)

perfiles_filtrado = perfiles[(perfiles["activo"]==1) &(perfiles["peso"]<150) &
                             (perfiles["peso"]!=0 )& (perfiles["altura"]>100) & (perfiles["altura"]<220)]
perfiles_filtrado.reset_index(drop=True, inplace=True)

### Cruce posicion sexo
pd.crosstab(perfiles_filtrado["posicion"],perfiles_filtrado["sexo"])
# Los Manual de posicion son exactamente los manual de sexo

### Cruce presiones y sexo
df_pres_sex = pd.DataFrame(pd.crosstab(perfiles_filtrado["presiones"],perfiles_filtrado["sexo"],normalize=False))
# Elimino aquellas presiones con menos de 10 registros en hombres o mujeres
df_pres_sex = df_pres_sex[(df_pres_sex["Female"]>=10) | (df_pres_sex["Male"]>=10)]
# Vemos las más frecuentes para cada uno
df_pres_sex_fem = df_pres_sex.sort_values(by=['Female'], ascending=False)
df_pres_sex_male = df_pres_sex.sort_values(by=['Male'], ascending=False)

# Para pintarlo
grouped_df = perfiles_filtrado.groupby(["presiones","sexo"])
df_pres_sex_graph = pd.DataFrame(grouped_df.size().reset_index(name = "Group_Count"))
# Elimino aquellas presiones con menos de 20 registros en hombres o mujeres
df_pres_sex_graph = df_pres_sex_graph[df_pres_sex_graph["Group_Count"]>=25]
sns.barplot(y='Group_Count',x='presiones',hue='sexo',data=df_pres_sex_graph)
plt.show()



### Cruce presiones y posicion
df_pres_pos = pd.DataFrame(pd.crosstab(perfiles_filtrado["presiones"],perfiles_filtrado["posicion"],normalize=False))
# Elimino aquellas presiones con menos de 10 registros en lateral o supine
df_pres_pos = df_pres_pos[(df_pres_pos["Lateral"]>=10) | (df_pres_pos["Supine"]>=10)]
# Vemos las más frecuentes para cada uno
df_pres_pos_lat = df_pres_pos.sort_values(by=['Lateral'], ascending=False)
df_pres_pos_sup = df_pres_pos.sort_values(by=['Supine'], ascending=False)

# Para pintarlo
grouped_df2 = perfiles_filtrado.groupby(["presiones","posicion"])
df_pres_pos_graph = pd.DataFrame(grouped_df2.size().reset_index(name = "Group_Count"))
# Elimino aquellas presiones con menos de 20 registros en hombres o mujeres
df_pres_pos_graph = df_pres_pos_graph[df_pres_pos_graph["Group_Count"]>=25]
sns.barplot(y='Group_Count',x='presiones',hue='posicion',data=df_pres_pos_graph)
plt.show()


### Cruce presiones y altura
# Categorizamos altura
perfiles_filtrado["altura_cat"] = pd.cut(perfiles_filtrado['altura'], bins=[0, 145, 160, 170, 180, 190, 250], include_lowest=True,
       labels=['<1.45', '1.45-1.6', '1.6-1.7', '1.7-1.8', '1.8-1.9', '>1.9'])
# Cruce
df_pres_alt = pd.DataFrame(pd.crosstab(perfiles_filtrado["presiones"],perfiles_filtrado["altura_cat"],normalize=False))

# Para pintarlo
grouped_df3 = perfiles_filtrado.groupby(["presiones","altura_cat"])
df_pres_alt_graph = pd.DataFrame(grouped_df3.size().reset_index(name = "Group_Count"))
# Elimino aquellas presiones con menos de 20 registros en hombres o mujeres
df_pres_alt_graph = df_pres_alt_graph[df_pres_alt_graph["Group_Count"]>=25]
sns.barplot(y='Group_Count',x='presiones',hue='altura_cat',data=df_pres_alt_graph)
plt.show()


### SQR NECESITA LIMPIEZA
sesiones["sqr"].describe()
plt.hist(sesiones["sqr"], bins=100)
plt.show()

# Filtro: sqr>0 y <500
sesiones_filter = sesiones.loc[(sesiones["sqr"]<=500) & (sesiones["sqr"]>=0)]
plt.hist(sesiones_filter["sqr"], bins=100)
plt.show()

# Lo que deberia ser
sesiones_filter2 = sesiones.loc[(sesiones["sqr"]<=100) & (sesiones["sqr"]>=0)]
plt.hist(sesiones_filter2["sqr"], bins=100)
plt.show()

### CLUSTERING
from sklearn import preprocessing
from statistics import mode
import gower
from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as sch


df_5var = perfiles_filtrado[["presiones","posicion","altura","peso","sexo"]]
df_5var = df_5var[df_5var["sexo"]!="Manual"]
df_4var = df_5var[["posicion","altura","peso","sexo"]]
df_4var.loc[df_4var["posicion"] == "Lateral","posicion"] = 1
df_4var.loc[(df_4var["posicion"] == "Supine"),"posicion"] = 0
df_4var.loc[df_4var["sexo"] == "Female","sexo"] = 1
df_4var.loc[(df_4var["sexo"] == "Male"),"sexo"] = 0



df_4var[["altura","peso"]] = preprocessing.scale(df_4var[["altura","peso"]],with_mean=True, with_std=True)

# pip install gower


# Gower's distance
gd = gower.gower_matrix(df_4var, cat_features = [True,False,False,True])

# Hierarchical


dendrogram = sch.dendrogram(sch.linkage(gd, method='ward'))
plt.show()

model = AgglomerativeClustering(n_clusters=4, affinity='precomputed',linkage='average').fit(gd)
labels = model.labels_

df_5var["labels"] = labels
# Descripcion clusterss
df_5var.groupby('labels').size()


df_5var.groupby('labels').agg(
    {
        'altura': [min, max, 'mean', 'std'],
        'peso': [min, max, 'mean', 'std'],
        'sexo': [mode],
        'posicion': [mode, 'size']
    }
)

df_5var.groupby(['labels','presiones']).size()


## Heatmap para las presiones

# Df para los 4 clusters
cols = ['Pos1', 'Pos2', 'Pos3', 'Pos4', 'Pos5', 'Pos6',
        'Pos7', 'Pos8', 'Pos9', 'Pos10', 'Pos11', 'Pos12']
rows_heat = ['Pres0', 'Pres1', 'Pres2', 'Pres3', 'Pres4', 'Pres5']
list_result = []

for h in range(len(np.unique(labels))):
    df_pres = df_5var.loc[df_5var['labels'] == h,'presiones']
    rows = range(len(df_pres))
    df_pres_distr = pd.DataFrame(columns=cols, index=rows)

    # Separamos las presiones
    for j in range(len(df_pres)):
        print(j)
        pres_ejem = df_pres.iloc[j]
        pres_ejem_split = [pres_ejem[i:i + 1] for i in range(0, len(pres_ejem), 1)]
        df_pres_distr.iloc[j, :] = pres_ejem_split

    # Rellenamos para el heatmap
    df_press_heat = pd.DataFrame(columns=cols, index=rows_heat)
    k = 0
    for pos in cols:
        print(pos)
        to_fill = list(df_pres_distr.groupby(pos)[pos].size())
        if len(to_fill) != 6:
            to_fill = [sum(df_pres_distr[pos] == '0'),
                       sum(df_pres_distr[pos] == '1'),
                       sum(df_pres_distr[pos] == '2'),
                       sum(df_pres_distr[pos] == '3'),
                       sum(df_pres_distr[pos] == '4'),
                       sum(df_pres_distr[pos] == '5')]
        df_press_heat.iloc[:, k] = to_fill
        k += 1

    list_result.append(df_press_heat)

df_press_heat0 = list_result[0]
df_press_heat1 = list_result[1]
df_press_heat2 = list_result[2]
df_press_heat3 = list_result[3]


ax = sns.heatmap(df_press_heat0, linewidth=0.5)
plt.show()

ax = sns.heatmap(df_press_heat1, linewidth=0.5)
plt.show()

ax = sns.heatmap(df_press_heat2, linewidth=0.5)
plt.show()

ax = sns.heatmap(df_press_heat3, linewidth=0.5)
plt.show()


import numpy as np
# desemejanza rango
data = df_4var[["altura","peso"]]
rangos = np.array(data.max(axis=0)) - np.array(data.min(axis=0))


desem_rango = np.zeros((len(data),len(data)))

for i in range(len(data) - 1):
    for z in np.arange(i+1,len(data)):
        for j in range(len(data.columns)):
            desem_rango[i,z] += abs(data.iloc[i, j] - data.iloc[z, j]) / rangos[j]
        if (df_4var.loc[i,"sexo"] == df_4var.loc[z,"sexo"]):
            desem_rango[i, z] += 1
        if (df_4var.loc[i,"posicion"] == df_4var.loc[z,"posicion"]):
            desem_rango[i, z] += 1

desem_rango = desem_rango + desem_rango.T - np.diag(np.diag(desem_rango))
desem_rango.max()


import numpy as np
###########################################################################################
##############                  SIMILARIDAD ENTRE PRESIONES                 ###############
###########################################################################################

# pressure1=df_perfiles.loc[0,'presiones']
# pressure2=df_perfiles.loc[150,'presiones']

def pressures_similarity(pressure1,pressure2):
    ## Comprobamos que tengan la misma longitud
    if (len(pressure1) != len(pressure2)):
        raise ValueError("Las presiones han de tener la misma longitud")

    # Separamos las presiones en vectores
    pressure1_list = np.array([int(pressure1[i:i + 1]) for i in range(0, len(pressure1), 1)])
    pressure2_list = np.array([int(pressure2[i:i + 1]) for i in range(0, len(pressure2), 1)])

    ## Similaridad entre sus dos presiones
    dif_pressures = np.absolute(np.subtract(pressure1_list, pressure2_list)) / 5
    disimilarity = np.sum(dif_pressures * (1 / len(pressure1)))

    return disimilarity


###################################################################################################
##############      SIMILARIDAD ENTRE INDIVIDUOS: Sexo, Posicion, altura, peso      ###############
###################################################################################################

## parto de altura y peso estandarizado

# df_4var = df_5var[["posicion","altura","peso","sexo"]]
# df_4var.shape[0]
# df=df_4var
# weights=[1/4,1/4,1/4,1/4]

def perfiles_similarity(df,weights):
    import numpy as np

    ## Inicializo matriz
    n_filas= df.shape[0]
    dis_matrix = np.zeros((n_filas, n_filas))

    ## Rango de peso y altura para calcular similaridad
    rango_peso=max(df.loc[:,"peso"])-min(df.loc[:,"peso"])
    rango_altura=max(df.loc[:,"altura"])-min(df.loc[:,"altura"])

    for i in range(n_filas):
        # print(i)
        individuo_i=df.loc[i,]
        for j in  np.arange(i + 1,n_filas):
            individuo_j=df.loc[j,]
            ## Similaridad en peso
            dis_peso=abs(individuo_i['peso']-individuo_j['peso'])/rango_peso
            ## similaridad en altura
            dis_altura=abs(individuo_i['altura']-individuo_j['altura'])/rango_altura
            ## Similaridad en sexo
            dis_sexo=0 if individuo_i['sexo']==individuo_j['sexo'] else 1
            ## similaridad en posicion
            dis_posicion=0 if individuo_i['posicion']==individuo_j['posicion'] else 1

            ## desemejanza total
            vector_desemejanzas=np.array([dis_peso,dis_altura,dis_sexo,dis_posicion])
            dis=np.sum(np.multiply(vector_desemejanzas,np.array(weights)))

            ## relleno matriz
            dis_matrix[i, j] = dis

        ## relleno la diagonal de 1
        dis_matrix[i, i] = 0
    dis_matrix = dis_matrix + dis_matrix.T - np.diag(np.diag(dis_matrix))

    return dis_matrix





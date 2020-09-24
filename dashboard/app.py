import os
import sys

import numpy as np
import pandas as pd
import string
from dash.dependencies import Input, Output, State
import base64
import dash
import dash_bio
import dash_core_components as dcc
import dash_core_components as dcc
import dash_daq as daq
import dash_html_components as html
import dash_html_components as html
import plotly.express as px
import dash_bootstrap_components as dbc
import math
from scipy.stats import norm

if sys.version_info < (3, 7):
    import pickle5 as pickle
else:
    import pickle


DATAPATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model')
with open(os.path.join(DATAPATH, 'knn.p'), 'rb') as pickle_file:
    model = pickle.load(pickle_file)

perfiles_sqr = pd.read_parquet('./data/perfiles_sqr_filtrado.parquet')
perfiles_sqr.reset_index(drop=True, inplace=True) # reseteamos el índice
perfiles_sqr['IMC'] = perfiles_sqr['peso'] / (perfiles_sqr['altura']/100)**2
perfiles_sqr['IMC_cat'] = pd.cut(perfiles_sqr['IMC'], bins=[0, 25, 30, 50],
                                include_lowest=True,labels=['Normal', 'Overweight', 'Obese'])

# tubos
for i in range(6):
    perfiles_sqr[f'PresPos{i+1}'] = perfiles_sqr.presiones.apply(lambda x: x[i])


def plot_heatmap(df):
    fig = px.imshow(df,
                    labels=dict(x="Posición del tubo", y="Nivel de presión", color="Porcentaje de uso"),
                    x=['Pos. 1', 'Pos. 2', 'Pos. 3', 'Pos. 4', 'Pos. 5', 'Pos. 6'],
                    y=['Nivel 1', 'Nivel 2', 'Nivel 3', 'Nivel 4', 'Nivel 5', 'Nivel 6'])
    fig.update_xaxes(side="top")
    return fig

def presiones_df_heat(df):
    # Dado un df input, separar las presiones de la variable presiones de perfiles_usuario
    # df_pres: dataframe  de 12 columnas, una por cada posicion, 1 fila por id
    # df_completo_pres: join(df, df_pres)
    # df_pres_count: dataframe con 12 columnas (posiciones) y 6 filas (niveles de presion), conteo de valores
    # df_pres_prop: df_pres_count/total valores

    # Del df input nos quedamos con presiones
    df_pres = pd.Series(df)

    # Estructura para los df resultantes
    cols = ['PresPos1', 'PresPos2', 'PresPos3', 'PresPos4', 'PresPos5', 'PresPos6']
    rows_heat = ['NivPres1', 'NivPres2', 'NivPres3', 'NivPres4', 'NivPres5', 'NivPres6']
    rows = range(len(df_pres))
    df_pres_split = pd.DataFrame(columns=cols, index=rows)
    df_pres_count = pd.DataFrame(columns=cols, index=rows_heat)
    # Separamos las presiones
    for j in range(len(df_pres)):
        pres_j = df_pres.iloc[j]
        pres_j_split = [pres_j[i:i + 1] for i in range(0, len(pres_j), 1)]
        df_pres_split.iloc[j, :] = pres_j_split
    # Para la distribucion
    k = 0
    for pos in cols:
        # print(pos)
        to_fill = list(df_pres_split.groupby(pos)[pos].size())
        if len(to_fill) != 6:
            to_fill = [sum(df_pres_split[pos] == '0'),
                       sum(df_pres_split[pos] == '1'),
                       sum(df_pres_split[pos] == '2'),
                       sum(df_pres_split[pos] == '3'),
                       sum(df_pres_split[pos] == '4'),
                       sum(df_pres_split[pos] == '5')]
        df_pres_count.iloc[:, k] = to_fill
        k += 1

    # Dividimos df_pres_count por el total de observaciones para tener la proporcion
    df_pres_prop = round((df_pres_count / len(df_pres))*100,2)
    # Juntamos las presiones separadas al df original
    df_completo_pres = pd.concat([pd.Series(df), df_pres_split], axis=1, join='inner')

    return df_completo_pres, df_pres_count, df_pres_prop


def simula_sqi(mu, sigma, n_simul=7, alfa=0.4):
    '''Simula los n valores de SQI y proporciona el valor de SQI límite para levantar las alertas'''
    sqi_simulados = np.random.normal(mu, sigma, n_simul)

    p = norm.ppf((1 - alfa) / 2)
    valor_alerta = mu - p * sigma

    return sqi_simulados, valor_alerta

def model_n(sqi_simulados, valor_alerta, ventana, n_alerta):
    '''Devuelve el ínidice necesario para filtar los SQI simulados hasta que se levanta la alerta. Dependiendo
    de la ventana y el número de veces (n_alerta) que el SQI es inferior al valor de alerta (valor_alerta)'''
    n_simul = len(sqi_simulados)
    alarmas = np.array([sum((sqi_simulados < valor_alerta)[i:i+ventana]) for i in range(n_simul-ventana+1)]) > (n_alerta - 1)
    if np.sum(alarmas) > 0:
        aviso_cambio = np.argmax(alarmas) + ventana
    else:
        aviso_cambio = None
    return aviso_cambio

def nueva_configuracion(presiones, configuracion_cero):
    '''Devuelve la nueva configuración para un individuo dados los estadísticos de las presiones para su categoría'''

    # Dataframe con las posibles presiones nuevas
    presiones_posibles = presiones[presiones.frecuencia_relativa > 10].sort_values(by='media', ascending=False)
    for i in range(1, presiones_posibles.shape[0]):
        # Configuración con SQR más alto, para una frecuencia relativa mayor del 10%, distinto de la configuración actual
        nuevas_presiones = presiones_posibles.head(i).index[0]
        if nuevas_presiones != configuracion_cero:
            break
    return nuevas_presiones

def sqr_actual(perfiles_sqr, sexo, posicion, altura, peso, configuracion_cero):
    '''Se calcula la media y la desviación para la simulación. Además, el dataframe de estadísticos descriptivos
    de las configuraciones de presiones en su grupo'''
    # Se calcula el IMC
    IMC = peso / (altura / 100) ** 2
    # Cálculo del IMC categorizado
    if IMC < 25:
        IMC_cat = 'Normal'
    elif IMC < 30:
        IMC_cat = 'Overweight'
    else:
        IMC_cat = 'Obese'
    # Filtrado del grupo al que pertenece
    perfiles_filtrado = perfiles_sqr[
        (perfiles_sqr.sexo == sexo) & (perfiles_sqr.posicion == posicion) & (perfiles_sqr.IMC_cat == IMC_cat)]
    presiones = perfiles_filtrado[['presiones', 'sqr']].groupby('presiones').describe().loc[:, 'sqr']
    # Cálculo de estadísticos del SQR para cada configuración de presiones
    presiones = presiones.rename({'count': 'frecuencia_absoluta', 'mean': 'media', 'std': 'desviación'},
                                 axis='columns').round(2)
    presiones['frecuencia_absoluta'] = presiones['frecuencia_absoluta'].astype('int')
    presiones['frecuencia_relativa'] = round(100 * presiones['frecuencia_absoluta'] / perfiles_filtrado.shape[0], 2)
    try:
        media = presiones.loc[configuracion_cero]['media']
        desviacion = presiones.loc[configuracion_cero]['desviación']
    except:
        media = perfiles_filtrado['sqr'].mean()
        desviacion = perfiles_filtrado['sqr'].std()

    if math.isnan(desviacion):
        desviacion = perfiles_filtrado['sqr'].std()

    return presiones, round(media, 2), round(desviacion, 2)


def modelo_evolutivo_simulacion(perfiles_sqr, sexo, posicion, altura, peso, configuracion_cero):
    # Se calculan los parámetros de la distribución del SQR y el dataframe de la categoría
    presiones, mu, sigma = sqr_actual(perfiles_sqr, sexo, posicion, altura, peso, configuracion_cero)
    # Se simulan los valores y el valor de alerta
    sqi_simulados, valor_alerta = simula_sqi(mu, sigma)
    # Se calcula el primer día de alerta de cada modelo
    alarmas_modelos = [model_n(sqi_simulados, valor_alerta, *i) for i in [(3, 2), (5, 2), (7, 7)] if
                       model_n(sqi_simulados, valor_alerta, *i) != None]
    # Se comprueba si hay alguna alerta en algún modelo
    if alarmas_modelos != []:
        # Se filtran los valores SQR simulados hasta el primer día de alerta
        valores_prev_alerta = sqi_simulados[:min(alarmas_modelos)]
        # Se proporciona la configuración recomendada
        nuevas_presiones = nueva_configuracion(presiones, configuracion_cero)
    else:
        valores_prev_alerta = sqi_simulados
        nuevas_presiones = None

    return nuevas_presiones, valores_prev_alerta, valor_alerta, mu, sigma

sidebar = html.Div(
    [
        html.H2("Modelos", className="display-4",style={"fontSize":"40px"}),
        html.Hr(),
        dbc.Nav(
            [
                dbc.NavItem(dbc.NavLink("Modelo día 0", href="/page-1", id="page-1-link")),
                dbc.NavItem(dbc.NavLink("Modelo supervisado", href="/page-2",id="page-2-link")),
                dbc.NavItem(dbc.NavLink("Aprendizaje por refuerzo", href="/page-3",id="page-3-link"))
            ],
            vertical=True,
            pills=True,
        ),
    ],
            style={
            "position": "fixed",
            "top": 0,
            "left": 0,
            "bottom": 0,
            "width": "16rem",
            "padding": "2rem 1rem",
            "backgroundColor": "#f8f9fa",
        },
)



def modelo_dia_cero():
    return html.Div(id='app', className='app', children=[
        #Empieza Header
        html.Div(className='header', children=[
            html.H1(
                "ABACO: Modelo de día 0",
                style={"textAlign":"center"}
            )
        ]),
        #Fin de header

        #Empieza contenedor de contenido (segunda row html)
        html.Div(id="contenido", className='contenido', children=[
        html.Div(id='vp-control-tabs', className='control-tabs', children=[
            dcc.Tabs(id='vp-tabs', value='explicacion', children=[
                dcc.Tab(
                    label='Explicación',
                    value='explicacion',
                    children=html.Div(className='control-tab', children=[
                        html.Br(),
                        html.P(
                           """El modelo de día 0 es un modelo de predicción supervisada que respondé a la siguiente pregunta:
                        """),
                        html.P(
                            """¿Qué configuración inicial se ha de proporcionar a un cliente nuevo del que no se dispone de información previa relativa a la calidad del sueño?
                        """),

                        html.P(
                            """El cliente tipo de este modelo sería el comprador del colchón en su primer día de uso, o el cliente de un hotel en su primera noche de estancia en el mismo.
                    """),

                        html.P(
                            """Con el objetivo de construir el modelo de día 0 se emplean técnicas de clasificación no supervisada sobre las diferentes configuraciones de sensores. Se buscan patrones comunes en todas ellas. Se identifican mediante técnicas de Aprendizaje basado en Similaridades aquellos clientes de los que sí se disponga de información de calidad del sueño, con mayores semejanzas con respecto al nuevo cliente."""
                        )
                    ])
                ),
                dcc.Tab(
                    label='Predecir',
                    value='predecir',
                    children=html.Div(className='control-tab', children=[
                        html.Br(),
                        html.Br(),
                        html.Div(className="Sexo", children=[
                            html.Div(style={"display": "inline-block", "width":"60%"}, children=[
                            html.H4("Sexo",style={"display": "inline-block"}),]),
                            html.Div(style={"display": "inline-block", "width": "40%"}, children=[
                            dcc.Dropdown(
                                id='sexo',
                                options=[
                                    {'label': 'Masculino', 'value': '1'},
                                    {'label': 'Femenino', 'value': '0'},
                                ],
                                value='0'
                            ,style={"display": "inline-block","verticalAlign": "middle","width": "120px"})])
                        ], style={"width": "100%"}),
                        html.Div(className="altura", children=[
                            html.Div(style={"display": "inline-block", "width": "60%"}, children=[
                            html.H4("Altura (cm)" , style={"display":"inline-block"})]),
                            html.Div(style={"display": "inline-block", "width": "40%"}, children=[
                            dcc.Input(id="altura", type="number", placeholder="Altura (cm)",min=0, max=250, step=1,value="180",
                                      style={"display": "inline-block","verticalAlign": "middle","width": "120px"})])
                        ],style={"width":"100%"}),
                        html.Div(className="peso", children=[
                            html.Div(style={"display": "inline-block", "width": "60%"}, children=[
                            html.H4("Peso (Kg)",style={"display": "inline-block"})]),
                            html.Div(style={"display": "inline-block", "width": "40%"}, children=[
                            dcc.Input(id="peso", type="number", placeholder="Peso (Kg)", min=0, max=300, step=1,value="80",
                                      style={"display": "inline-block","verticalAlign": "middle","width": "120px"})])
                        ], style={"width": "100%"}),
                        html.Div(className="Posicion", children=[
                            html.Div(style={"display": "inline-block", "width": "60%"}, children=[
                            html.H4("Posición",style={"display": "inline-block"})]),
                            html.Div(style={"display": "inline-block", "width": "40%"}, children=[
                            dcc.Dropdown(
                                id='posicion',
                                options=[
                                    {'label': 'Lateral', 'value': '0'},
                                    {'label': 'Supine', 'value': '1'},
                                ],
                                value='0'
                            , style={"display": "inline-block","verticalAlign": "middle","width": "120px"})])
                        ]),
                        html.Div(children=[
                        html.Button('Configuración inicial Día 0', id='predecir',className="predecir", n_clicks=0,style={"marginTop": "50px","width":"70%","height": "70px","WebkitBorderRadius": "50px"})],
                        style={"textAlign":"center"})
                    ])
                )
            ])
        ]),
        html.Div(id='plot-content', className="plot-content", children=[
            html.H4("Configuración de los Vecinos más cercanos",style={"fontSize":"30px","textAlign":"center", "marginTop": "1px"}),
            dcc.Loading(className='dashbio-loading', children=html.Div(
                id='plot-div',
                children=dcc.Graph(
                    id='plot'
                ,style={"height":"550px"}),
            )),
            html.Div(id='results', className="results",
             children=[
                 html.Div(style={"display": "flex"},children=[
                     html.H2("SQI medio: ",style={"fontSize": "20px"}),
                             html.H2(id="resultado-output3",
                                     style={"fontSize": "20px", "textAlign": "center", "color": "green","margin-left":"5px"})
                 ]),
                 html.Div(style={"display": "flex"},children=[
                     html.H2("Configuración óptima recomendada: ",style={"fontSize": "20px"}),
                             html.H2(id="resultado-output",
                                     style={"fontSize": "20px", "textAlign": "center", "color": "green","margin-left":"5px"})
                 ]),
                 html.Div(style={"display": "flex"},children=[
                     html.H2("SQI esperado con la configuración recomendada: ",style={"fontSize": "20px"}),
                     html.H2(id="resultado-output2",
                             style={"fontSize": "20px", "textAlign": "center", "color": "green","margin-left":"5px"}),
                 ])

             ])
        ])
    ])
    ])



def modelo_supervisado():

    return html.Div(id='app2', className='app', children=[
        #Empieza Header
        html.Div(className='header', children=[
            html.H1(
                "ABACO: Modelo supervisado",
                style={"textAlign":"center"}
            )
        ]),
        #Fin de header

        #Empieza contenedor de contenido (segunda row html)
        html.Div(id="contenido2", className='contenido', children=[
        html.Div(id='vp-control-tabs2', className='control-tabs', children=[
            dcc.Tabs(id='vp-tabs2', value='predecir2', children=[
                dcc.Tab(
                    label='Predecir',
                    value='predecir2',
                    children=html.Div(className='control-tab', children=[
                        html.Br(),
                        html.Br(),
                        html.Div(className="Sexo", children=[
                            html.Div(style={"display": "inline-block", "width":"60%"}, children=[
                            html.H4("Sexo",style={"display": "inline-block"}),]),
                            html.Div(style={"display": "inline-block", "width": "40%"}, children=[
                            dcc.Dropdown(
                                id='sexo',
                                options=[
                                    {'label': 'Masculino', 'value': 'Male'},
                                    {'label': 'Femenino', 'value': 'Female'},
                                ],
                                value='Male'
                            ,style={"display": "inline-block","verticalAlign": "middle","width": "120px"})])
                        ], style={"width": "100%"}),
                        html.Div(className="altura", children=[
                            html.Div(style={"display": "inline-block", "width": "60%"}, children=[
                            html.H4("Altura (cm)" , style={"display":"inline-block"})]),
                            html.Div(style={"display": "inline-block", "width": "40%"}, children=[
                            dcc.Input(id="altura", type="number", placeholder="Altura (cm)",min=0, max=250, step=1,value="180",
                                      style={"display": "inline-block","verticalAlign": "middle","width": "120px"})])
                        ],style={"width":"100%"}),
                        html.Div(className="peso", children=[
                            html.Div(style={"display": "inline-block", "width": "60%"}, children=[
                            html.H4("Peso (Kg)",style={"display": "inline-block"})]),
                            html.Div(style={"display": "inline-block", "width": "40%"}, children=[
                            dcc.Input(id="peso", type="number", placeholder="Peso (Kg)", min=0, max=300, step=1,value="80",
                                      style={"display": "inline-block","verticalAlign": "middle","width": "120px"})])
                        ], style={"width": "100%"}),
                        html.Div(className="Posicion", children=[
                            html.Div(style={"display": "inline-block", "width": "60%"}, children=[
                            html.H4("Posición",style={"display": "inline-block"})]),
                            html.Div(style={"display": "inline-block", "width": "40%"}, children=[
                            dcc.Dropdown(
                                id='posicion',
                                options=[
                                    {'label': 'Lateral', 'value': 'Lateral'},
                                    {'label': 'Supine', 'value': 'Supine'},
                                ],
                                value='Lateral'
                            , style={"display": "inline-block","verticalAlign": "middle","width": "120px"})])
                        ]),
                        html.Div(children=[
                        html.Button('Obtener Configuración', id='predecir2',className="predecir", n_clicks=0,style={"marginTop": "50px","width":"70%","height": "70px","WebkitBorderRadius": "50px"})],
                        style={"textAlign":"center"})
                    ])
                )
            ])
        ]),
        html.Div(id='comun-content', className="comun-content", children=[

            html.Div(id='results2',
             children=[                     dcc.Loading(id="loading1",children=[

                 html.Div( className="results", children=[

                     html.H2("Configuración más común:",
                             style={"fontSize": "20px", "textAlign": "left", "marginTop": "10px"}),
                     html.H2(id="resultado-output5",style={"fontSize":"20px","textAlign":"center" ,"color":"green"}),
                     html.H2("SQI medio:",
                             style={"fontSize": "20px", "textAlign": "left", "marginTop": "40px"}),
                     html.H2(id="resultado-output6",style={"fontSize":"20px","textAlign":"center","color":"green"}),
                ],style={"marginTop":"40px","padding":"20px"}),
                 html.Div(className="results", children=[
                     html.H4("Configuración personalizada:",
                             style={"fontSize": "20px", "textAlign": "left", "marginTop": "10px"}),
                     html.H2(id="resultado-output7",
                             style={"fontSize": "20px", "textAlign": "center", "marginTop": "1px", "color": "green"}),
                     html.H4("SQI medio:",
                             style={"fontSize": "20px", "textAlign": "left", "marginTop": "40px"}),
                     html.H2(id="resultado-output8",
                             style={"fontSize": "20px", "textAlign": "center", "marginTop": "1px", "color": "green"})
                ],style={"marginTop":"60px","padding":"20px"})

             ])
                                            ])
        ]),
            html.Div(id='play-content', className="play-content", children=[
                html.H4("Personalización de presiones",
                        style={"fontSize": "30px", "textAlign": "center", "marginTop": "1px"}),



                dcc.Loading(className='dashbio-loading', children=html.Div(
                    id='sliders',
                    children=[
                        html.Div( className="labels",
                                 children=[
                        dcc.Slider(
                            className="slider_h",
                            id='slider_updatemode_1',
                            marks={i: "{}".format(i) for i in [1, 2, 3, 4,5,6]},
                            min=1,
                            max=6,
                            step=1,
                            updatemode='drag',
                            vertical=True
                        ),
                                     html.H2("Tubo 1", style={"fontSize": "13px", "marginTop":"10px"}),
                                 ]),
                        html.Div(className="labels",
                                 children=[
                                     dcc.Slider(
                                         className="slider_h",
                                         id='slider_updatemode_2',
                                         marks={i: "{}".format(i) for i in [1, 2, 3, 4, 5, 6]},
                                         min=1,
                                         max=6,
                                         step=1,
                                         updatemode='drag',
                                         vertical=True
                                     ),
                                     html.H2("Tubo 2", style={"fontSize": "13px", "marginTop":"10px"}),
                                 ]),
                        html.Div(className="labels",
                                 children=[
                                     dcc.Slider(
                                         className="slider_h",
                                         id='slider_updatemode_3',
                                         marks={i: "{}".format(i) for i in [1, 2, 3, 4, 5, 6]},
                                         min=1,
                                         max=6,
                                         step=1,
                                         updatemode='drag',
                                         vertical=True
                                     ),
                                     html.H2("Tubo 3", style={"fontSize": "13px", "marginTop":"10px"}),
                                 ]),
                        html.Div(className="labels",
                                 children=[
                                     dcc.Slider(
                                         className="slider_h",
                                         id='slider_updatemode_4',
                                         marks={i: "{}".format(i) for i in [1, 2, 3, 4, 5, 6]},
                                         min=1,
                                         max=6,
                                         step=1,
                                         updatemode='drag',
                                         vertical=True
                                     ),
                                     html.H2("Tubo 4", style={"fontSize": "13px", "marginTop": "10px"}),
                                 ]),
                        html.Div(className="labels",
                                 children=[
                                     dcc.Slider(
                                         className="slider_h",
                                         id='slider_updatemode_5',
                                         marks={i: "{}".format(i) for i in [1, 2, 3, 4, 5, 6]},
                                         min=1,
                                         max=6,
                                         step=1,
                                         updatemode='drag',
                                         vertical=True
                                     ),
                                     html.H2("Tubo 5", style={"fontSize": "13px", "marginTop":"10px"}),
                                 ]),
                        html.Div(className="labels",
                                 children=[
                                     dcc.Slider(
                                         className="slider_h",
                                         id='slider_updatemode_6',
                                         marks={i: "{}".format(i) for i in [1, 2, 3, 4, 5, 6]},
                                         min=1,
                                         max=6,
                                         step=1,
                                         updatemode='drag',
                                         vertical=True
                                     ),
                                     html.H2("Tubo 6", style={"fontSize": "13px", "marginTop":"10px"}),
                                 ]),
                    ]
                , style = {"height": "300px","display":"flex"}))

            ]),
    ])
    ])



def modelo_refuerzo():
    return html.Div(id='app3', className='app', children=[
        #Empieza Header
        html.Div(className='header', children=[
            html.H1(
                "ABACO: Aprendizaje por refuerzo",
                style={"textAlign":"center"}
            )
        ]),
        #Fin de header

        #Empieza contenedor de contenido (segunda row html)
        html.Div(id="contenido", className='contenido', children=[
        html.Div(id='vp-control-tabs', className='control-tabs', children=[
            dcc.Tabs(id='vp-tabs', value='predecir', children=[
                dcc.Tab(
                    label='Predecir',
                    value='predecir',
                    children=html.Div(className='control-tab', children=[
                        html.Br(),
                        html.Br(),
                        html.Div(className="Sexo", children=[
                            html.Div(style={"display": "inline-block", "width":"60%"}, children=[
                            html.H4("Sexo",style={"display": "inline-block"}),]),
                            html.Div(style={"display": "inline-block", "width": "40%"}, children=[
                            dcc.Dropdown(
                                id='sexo2',
                                options=[
                                    {'label': 'Masculino', 'value': '1'},
                                    {'label': 'Femenino', 'value': '0'},
                                ],
                                value='0'
                            ,style={"display": "inline-block","verticalAlign": "middle","width": "120px"})])
                        ], style={"width": "100%"}),
                        html.Div(className="altura", children=[
                            html.Div(style={"display": "inline-block", "width": "60%"}, children=[
                            html.H4("Altura (cm)" , style={"display":"inline-block"})]),
                            html.Div(style={"display": "inline-block", "width": "40%"}, children=[
                            dcc.Input(id="altura2", type="number", placeholder="Altura (cm)",min=0, max=250, step=1,value="180",
                                      style={"display": "inline-block","verticalAlign": "middle","width": "120px"})])
                        ],style={"width":"100%"}),
                        html.Div(className="peso", children=[
                            html.Div(style={"display": "inline-block", "width": "60%"}, children=[
                            html.H4("Peso (Kg)",style={"display": "inline-block"})]),
                            html.Div(style={"display": "inline-block", "width": "40%"}, children=[
                            dcc.Input(id="peso2", type="number", placeholder="Peso (Kg)", min=0, max=300, step=1,value="80",
                                      style={"display": "inline-block","verticalAlign": "middle","width": "120px"})])
                        ], style={"width": "100%"}),
                        html.Div(className="Posicion", children=[
                            html.Div(style={"display": "inline-block", "width": "60%"}, children=[
                            html.H4("Posición",style={"display": "inline-block"})]),
                            html.Div(style={"display": "inline-block", "width": "40%"}, children=[
                            dcc.Dropdown(
                                id='posicion2',
                                options=[
                                    {'label': 'Lateral', 'value': '0'},
                                    {'label': 'Supine', 'value': '1'},
                                ],
                                value='0'
                            , style={"display": "inline-block","verticalAlign": "middle","width": "120px"})])
                        ])
                    ])
                )
            ])
        ],style={"position":"relative"}),
        html.Div(className="comun-content", children=[
            html.Div(id='results2',
             children=[

                 html.Div( className="results", children=[

                     html.H2("Configuración recomendada:",
                             style={"fontSize": "20px", "textAlign": "left", "marginTop": "10px"}),
                     html.H2(id="resultado-output11",style={"fontSize":"20px","textAlign":"center" ,"color":"green"}),
                     html.H2("SQI esperado:",
                             style={"fontSize": "20px", "textAlign": "left", "marginTop": "40px"}),
                     html.H2(id="resultado-output12",style={"fontSize":"20px","textAlign":"center","color":"green"}),
                ],style={"marginTop":"40px","padding":"20px"})

                                            ]),
            html.Div(children=[
                html.Button('Simula como vas a dormir', id='simular', className="predecir", n_clicks=0,
                            style={"position": "absolute", "bottom": "4%", "marginTop": "50px", "width": "70%", "marginLeft":"10%",
                                   "height": "70px",
                                   "WebkitBorderRadius": "50px"})],
                style={"textAlign": "center","display": "flex"})

        ],style={"position":"relative"}),

            html.Div(id='play-content', className="play-content", children=[
                html.H4("Simulación (SQI)",
                        style={"fontSize": "30px", "textAlign": "center", "marginTop": "1px"}),

                dcc.Loading(className='dashbio-loading', children=html.Div(
                    id='simulacion',
                    children=[
                        html.Div(children=[
                            html.H2(id="resultado-output15",
                                    style={"fontSize": "20px"})
                        ],style={"margin-top":"25px"}),
                        html.Div(id='resultado-output14', className="results",
                                 children=[

                                 ],style={"position":"absolute","bottom":"4%","right":"4%","left":"4%"})
                    ])
                )

            ],style={"position":"relative","min-height":"500px"})

    ])
    ])



def callbacks(_app):
    @_app.callback(
        [Output(f"page-{i}-link", "active") for i in range(1, 4)],
        [Input("url", "pathname")],
    )
    def toggle_active_links(pathname):
        if pathname == "/":
            return True, False,False
        return [pathname == f"/page-{i}" for i in range(1, 4)]

    @_app.callback(Output("page-content", "children"), [Input("url", "pathname")])
    def render_page_content(pathname):
        if pathname in ["/", "/page-1"]:
            return modelo_dia_cero()
        elif pathname == "/page-2":
            return modelo_supervisado()
        elif pathname == "/page-3":
            return modelo_refuerzo()
        # If the user tries to reach a different page, return a 404 message
        return dbc.Jumbotron(
            [
                html.H1("404: Not found", className="text-danger"),
                html.Hr(),
                html.P(f"The pathname {pathname} was not recognised..."),
            ]
        )

    @_app.callback(
        [Output('resultado-output', 'children'),
         Output('resultado-output2', 'children'),
         Output('resultado-output3', 'children'),
        Output('plot', 'figure')],
        [
            Input('predecir', 'n_clicks')
        ],
    [dash.dependencies.State('sexo', 'value'),
     dash.dependencies.State('posicion', 'value'),
     dash.dependencies.State('altura', 'value'),
     dash.dependencies.State('peso', 'value')]
    )
    def get_prediction(predecir,sexo,posicion,altura,peso):
        results = model.predict(np.array([int(altura),int(peso),int(posicion),int(sexo)]),neighbours_index=True)
        target =model.target[results[1]]
        sqis = model.ref[results[1]]
        mean = np.mean(sqis[0])
        std = np.std(sqis[0])
        mean = np.round(mean,decimals=3)
        std = np.round(std, decimals=3)

        heatmap= presiones_df_heat(target[0])
        plot = plot_heatmap(heatmap[2])

        optima = [int(i) + 1 for i in results[0][0][0]]

        return '{}'.format(
            ''.join(map(str, optima))
        ),'{}'.format(
            round(results[0][0][1],2)), '{} ± {}'.format(round(mean,2),round(std,2)),  plot

    @_app.callback(
        [Output('resultado-output5', 'children'),
         Output('resultado-output6', 'children'),
         Output(component_id='slider_updatemode_1', component_property='value'),
         Output(component_id='slider_updatemode_2', component_property='value'),
         Output(component_id='slider_updatemode_3', component_property='value'),
         Output(component_id='slider_updatemode_4', component_property='value'),
         Output(component_id='slider_updatemode_5', component_property='value'),
         Output(component_id='slider_updatemode_6', component_property='value')],
        [
            Input('predecir2', 'n_clicks')
        ],
    [dash.dependencies.State('sexo', 'value'),
     dash.dependencies.State('posicion', 'value'),
     dash.dependencies.State('altura', 'value'),
     dash.dependencies.State('peso', 'value')]
    )
    def get_results(predecir,sexo,posicion,altura,peso):
        imc = int(peso) / (int(altura)/100)**2
        if (imc<25):
            imc = "Normal"
        elif (imc>25) and (imc<30):
            imc ="Overweight"
        else:
            imc = "Obese"

        perfiles_afines = perfiles_sqr[(perfiles_sqr["IMC_cat"] == imc) & (perfiles_sqr["sexo"] == sexo) & (perfiles_sqr["posicion"] == posicion)].groupby('presiones').describe().loc[:, 'sqr']
        perfiles_afines = perfiles_afines.rename({'count': 'frecuencia absoluta', 'mean': 'media', 'std': 'desviacion'},
                                             axis='columns').round(2)
        perfiles_afines['frecuencia absoluta'] = perfiles_afines['frecuencia absoluta'].astype('int')
        presiones_afines = perfiles_afines.sort_values(by='frecuencia absoluta', ascending=False).head(3)


        presion_comun= presiones_afines.head(1).index[0]
        presion_comun=[int(i)+1 for i in presion_comun]
        mean = presiones_afines.iloc[[0]]["media"].values[0]
        std = presiones_afines.iloc[[0]]["desviacion"].values[0]
        return presion_comun,'{} ± {}'.format(mean,std), int(presion_comun[0]),int(presion_comun[1]),int(presion_comun[2]),int(presion_comun[3]),int(presion_comun[4]),int(presion_comun[5])

    @_app.callback(
        [Output('resultado-output7', 'children'),
         Output('resultado-output8', 'children')],
    [dash.dependencies.Input('slider_updatemode_1', 'value'),
     dash.dependencies.Input('slider_updatemode_2', 'value'),
     dash.dependencies.Input('slider_updatemode_3', 'value'),
     dash.dependencies.Input('slider_updatemode_4', 'value'),
     dash.dependencies.Input('slider_updatemode_5', 'value'),
     dash.dependencies.Input('slider_updatemode_6', 'value'),
     ],
    [dash.dependencies.State('sexo', 'value'),
     dash.dependencies.State('posicion', 'value'),
     dash.dependencies.State('altura', 'value'),
     dash.dependencies.State('peso', 'value')]
    )
    def update_play(slider_updatemode_1,slider_updatemode_2,slider_updatemode_3,slider_updatemode_4,slider_updatemode_5,slider_updatemode_6,sexo,posicion,altura,peso):
        imc = int(peso) / (int(altura) / 100) ** 2
        if (imc < 25):
            imc = "Normal"
        elif (imc > 25) and (imc < 30):
            imc = "Overweight"
        else:
            imc = "Obese"

        if (slider_updatemode_1 is None) or (slider_updatemode_2 is None) or (slider_updatemode_3 is None) or (slider_updatemode_4 is None) or (slider_updatemode_5 is None) or (slider_updatemode_6 is None):
            return "",""


        perfiles_afines = perfiles_sqr[(perfiles_sqr["IMC_cat"] == imc) & (perfiles_sqr["sexo"] == sexo) & (
                    perfiles_sqr["posicion"] == posicion) & (perfiles_sqr["presiones"] ==
        '{}{}{}{}{}{}'.format(slider_updatemode_1-1,slider_updatemode_2-1,slider_updatemode_3-1,slider_updatemode_4-1,slider_updatemode_5-1,slider_updatemode_6-1))]

        mean = round(perfiles_afines["sqr"].mean(),2)
        std = round(perfiles_afines["sqr"].std(),2)

        if math.isnan(mean) or math.isnan(std):
            result = "No hay registros en la base de datos"
        else:
            result = '{} ± {}'.format(mean, std)

        return '{}{}{}{}{}{}'.format(slider_updatemode_1,slider_updatemode_2,slider_updatemode_3,slider_updatemode_4,slider_updatemode_5,slider_updatemode_6),result


    @_app.callback(
        [Output('resultado-output11', 'children'),
         Output('resultado-output12', 'children'),
         Output('resultado-output14', 'children'),
         Output('resultado-output15', 'children')],

        [Input('sexo2', 'value'),
         Input('posicion2', 'value'),
         Input('altura2', 'value'),
         Input('peso2', 'value'),
         Input("simular","n_clicks")]
    )
    def get_prediction_refuerzo(sexo,posicion,altura,peso,simular):

        results = model.predict(np.array([int(altura),int(peso),int(posicion),int(sexo)]),neighbours_index=True)
        target =model.target[results[1]]
        sqis = model.ref[results[1]]
        mean = np.mean(sqis[0])
        std = np.std(sqis[0])
        mean = np.round(mean,decimals=3)
        std = np.round(std, decimals=3)
        presiones = '{}'.format(
            ''.join(map(str, results[0][0][0])))
        if int(sexo) ==0:
            sexo="Female"
        else:
            sexo="Male"

        if int(posicion) == 0:
            posicion="Lateral"
        else:
            posicion="Supine"

        simulacion = modelo_evolutivo_simulacion(perfiles_sqr, sexo, posicion, int(altura), int(peso), results[0][0][0])

        optima = [int(i) + 1 for i in results[0][0][0]]


        dias = html.Ul([html.Li("Día "+ str(k+1) +":  " + str(round(x,2)), style={"color":"green","marginTop":"2%"}) if int(x)>int(simulacion[2]) else html.Li("Día "+ str(k+1) +":  " + str(round(x,2)), style={"color":"red","marginTop":"2%"}) for k,x in enumerate(simulacion[1])])


        cambios = html.Div(children=[
            html.H2("Se recomiendan los siguientes cambios en tu configuración:", style={"fontSize": "20px"}),
            html.H2(simulacion[0],
                    style={"fontSize": "20px", "textAlign": "center", "color": "green"})
        ])
        if simulacion[0] == None:
            cambios = html.Div(children=[
                html.H2("No se recomienda ningún cambio en tu configuración", style={"fontSize": "20px"})
            ])



        return '{}'.format(
            ''.join(map(str, optima))
        ),'{} ± {}'.format(round(simulacion[3],2),round(simulacion[4],2)),cambios,dias





app = dash.Dash(__name__,external_stylesheets=[dbc.themes.BOOTSTRAP])
app.scripts.config.serve_locally = True
app.config['suppress_callback_exceptions'] = True
content = html.Div(id="page-content", style={"marginLeft": "18rem",    "marginRight": "2rem",    "padding": "2rem 1rem",})
app.layout= html.Div([dcc.Location(id="url", refresh=False), sidebar, content])
app_name ='Abaco'
callbacks(app)



if __name__ == '__main__':
    app.run_server(debug=True, port=8050, host='0.0.0.0')

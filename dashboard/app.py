import os
import sys

import numpy as np
import pandas as pd

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

if sys.version_info < (3, 7):
    import pickle5 as pickle
else:
    import pickle


DATAPATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model')
with open(os.path.join(DATAPATH, 'knn.p'), 'rb') as pickle_file:
    model = pickle.load(pickle_file)

def plot_heatmap(df):
    fig = px.imshow(df,
                    labels=dict(x="Posición del tubo", y="Nivel de presión", color="Porcentaje de uso"),
                    x=['Pos. 1', 'Pos. 2', 'Pos. 3', 'Pos. 4', 'Pos. 5', 'Pos. 6', 'Pos. 7', 'Pos. 8',
                       'Pos. 9', 'Pos. 10', 'Pos. 11', 'Pos. 12'],
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
    cols = ['PresPos1', 'PresPos2', 'PresPos3', 'PresPos4', 'PresPos5', 'PresPos6',
            'PresPos7', 'PresPos8', 'PresPos9', 'PresPos10', 'PresPos11', 'PresPos12']
    rows_heat = ['NivPres0', 'NivPres1', 'NivPres2', 'NivPres3', 'NivPres4', 'NivPres5']
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



def layout():

    return html.Div(id='app', className='app', children=[
        #Empieza Header
        html.Div(className='header', children=[
            html.H1(
                "ABACO: Modelo de Día 0",
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
                        html.Button('Configuración inicial Día 0', id='predecir',className="predecir", n_clicks=0,style={"marginTop": "50px","width":"50%","height": "40px","-webkit-border-radius": "50px"})],
                        style={"textAlign":"center"})
                    ])
                )
            ])
        ]),
        html.Div(id='plot-content', className="plot-content", children=[
            html.H4("Configuración de los Vecinos más cercanos",style={"fontSize":"20px","textAlign":"center", "marginTop": "1px"}),
            dcc.Loading(className='dashbio-loading', children=html.Div(
                id='plot-div',
                children=dcc.Graph(
                    id='plot'
                ,style={"height":"420px"}),
            )),
            html.Div(id='results', className="results",
             children=[
                 html.H4(id="resultado-output3", style={"fontSize": "13px"}),
                 html.H4(id="resultado-output",style={"fontSize":"13px"}),
                 html.H4(id="resultado-output2",style={"fontSize":"13px"})
             ])
        ])
    ])
    ])


def callbacks(_app):
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
        return 'Configuración Óptima recomendada: {}'.format(
            results[0][0][0]
        ),'SQI esperado con la configuración recomendada: {}'.format(
            results[0][0][1]), 'SQI medio: {} ± {}'.format(mean,std),  plot



app = dash.Dash(__name__)
app.scripts.config.serve_locally = True
app.config['suppress_callback_exceptions'] = True
app.layout=layout()
app_name ='Abaco'
callbacks(app)



if __name__ == '__main__':
    app.run_server(debug=True, port=8050)

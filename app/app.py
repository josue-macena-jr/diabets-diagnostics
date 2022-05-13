import pandas as pd
import streamlit as st
from streamlit import caching
from datetime import datetime
from datetime import timedelta, date
import altair as alt
import base64
import io
from io import StringIO
import streamlit.components.v1 as components
import time
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier



st.set_page_config(
    page_title="Squad Scikit-Learn",
    layout="wide",
    page_icon=":shark",
    initial_sidebar_state="expanded",
)

st.sidebar.image('https://stacklabs02.s3.amazonaws.com/Scikit_learn_logo_small.svg.png', use_column_width=True)

#título
st.title("Projeto Diagnóstico Diabetes")

# Apresentação
st.subheader("Desenvolvido por:")
st.write(""" 
- [**Josué Macena**](https://www.linkedin.com/in/josue-junior-goncalves-macena/)""")

# Descriçaõ do projeto
st.subheader("Objetivo do Projeto:")
st.write("**Realizar uma previsão de pessoas que podem vir a desenvolver Diabetes com base nas suas características atuais, usando um algoritmo de Machine Learning para realizar uma predição.**")

# Descrição do dataset
with st.expander("Descrição do dataset", expanded=False):
    st.write("""
        
        Dataset selecionado: diabetes_binary_5050split_health_indicators_BRFSS2015.csv

        **Informação:** O conjunto de dados originalmente tem 330 recursos (colunas), mas com base na pesquisa de diabetes sobre fatores que influenciam a doença de diabetes e outras condições crônicas de saúde, apenas alguns recursos são incluídos nesta análise.

        **Fatores de risco importantes:** Pesquisas na área identificaram os seguintes fatores de risco importantes para diabetes e outras doenças crônicas, como doenças cardíacas (não em ordem estrita de importância):

        *   blood pressure high (pressão arterial (alta))
        *   cholesterol high (colesterol (alto))
        *   smoking (fumar)
        *   diabetes (diabetes)
        *   obesity (obesidade)
        *   age (idade)
        *   sex (sexo)
        *   race (raça)
        *   diet (dieta)
        *   exercise (exercício físico)
        *   alcohol consumption (consumo de alcool)
        *   BMI (IMC)
        *   Household Income (Renda Familiar)
        *   Marital Status (Estado Civil)
        *   Sleep (Dormir)
        *   Time since last checkup (Tempo desde o último check-up)
        *   Education (Educação)
        *   Health care coverage (Plano de Saúde)
        *   Mental Health (Saúde Mental)

        **As características selecionadas do dataset são:** 

        Variável de resposta/variável dependente:

        **Diabetes_binary**
        *   1 - Se o entrevistado têm diabetes
        *   0 - Se o entrevistado nâo têm diabetes\n
        Variáveis ​​independentes:\n

        **HighBP - Pressão alta**\n
        Adultos que foram informados de que têm pressão alta por um médico, enfermeiro ou outro profissional de saúde
        *  1 sim
        *  0 não


        **HighChol - Colesterol alto**\n
        ALGUMA VEZ você foi informado por um médico, enfermeiro ou outro profissional de saúde que seu colesterol no sangue está alto?
        *   1 sim
        *   0 não

        **CholCheck**

        **BMI/ IMC - Índice de Massa Corporal (IMC)**\n
        Valores: 12-98

        **Smoker - Fumante**\n
        Você fumou pelo menos 100 cigarros em toda a sua vida?
        *   1 sim
        *   0 não

        **Stroke - Derrame**\n
        (Já disse) você teve um derrame?
        *   1 sim
        *   0 não

        **Heart Disease or Attack - Doença cardíaca /coronária**\n
        Entrevistados que já relataram ter doença cardíaca coronária (DAC) ou infarto do miocárdio (IM)
        *   1 sim
        *   0 não

        **Phys Activity - Atividade Física**\n
        Adultos que relataram fazer atividade física ou exercício durante os últimos 30 dias além do trabalho regular
        *   1 sim
        *   0 não

        **Fruits - Dieta de Frutas**\n
        Consumir Frutas 1 ou mais vezes por dia
        *   1 sim
        *   0 não

        **Veggies - Dieta de Vegetais**\n
        Consuma Legumes 1 ou mais vezes por dia
        *   1 sim
        *   0 não

        **HvyAlcoholConsump - Consumo de álcool**\n
        Consumidores Excessivo(homens adultos com mais de 14 doses por semana e mulheres adultas com mais de 7 doses por semana)
        *   1 sim
        *   0 não

        **AnyHealthcare - Plano de Saúde**\n
        Você tem algum tipo de cobertura de saúde, incluindo seguro saúde, planos pré-pagos, como HMOs, ou planos governamentais, como Medicare ou Indian Health Service?
        *   1 sim
        *   0 não

        **NoDocbcCost - Despesa Médica**\n
        Houve algum momento nos últimos 12 meses em que você precisou consultar um médico, mas não pôde devido ao custo?
        *   1 sim
        *   2 não

        **GenHlth Saúde Geral - Você diria que em geral sua saúde é?**\n
        *   1: Excelente
        *   2: Muito bom
        *   3: Bom
        *   4: Razoável
        *   5: Ruim

        **MentHlth - Saúde Mental**\n
        Agora, pensando em sua saúde mental, que inclui estresse, depressão e problemas com emoções, por quantos dias nos últimos 30 dias sua saúde mental não foi boa?\n
        Valores: 1-30
        
        **PhysHlth -Saúde Física**\n
        Agora, pensando em sua saúde física, que inclui doenças e lesões físicas, por quantos dias nos últimos 30 dias sua saúde física não foi boa?\n
        Valores: 1-30
        
        **DiffWalk - Dificuldade de Caminhar**\n
        Você tem sérias dificuldades para caminhar ou subir escadas?
        *   1 sim
        *   2 não

        **Sex - Sexo**\n
        Indica o sexo do entrevistado.
        *   0: Masculino
        *   1: Feminino

        **Age - Idade**\n
        14 categorias de idade
        *   1: De 18 a 24
        *   2: De 25 a 29
        *   3: De 30 a 34
        *   4: De 35 a 39
        *   5: De 40 a 44
        *   6: De 45 a 49
        *   7: De 50 a 54
        *   8: De 55 a 59
        *   9: De 60 a 64
        *   10: De 65 a 69
        *   11: De 70 a 74
        *   12: De 75 a 79
        *   13: De 80 ou mais
        *   14: Não sabe/Recusou-se/Falta Dado/7 <= Idade <= 9


        **Education - Educação**\n
        Qual é a maior série ou ano de escola que você concluiu?
        *   1: Não concluiu o ensino médio
        *   2: Concluiu o ensino médio
        *   3: Frequentou a faculdade
        *   4: Concluiu a faculdade


        **Income - Renda Familiar**\n
        Sua renda familiar anual é de todas as fontes: (Se o respondente recusar em qualquer nível de renda, codifique "Recusado".)

        *   1: Menor que $10 mil
        *   2: 15 mil

        *   3: 20 mil
        *   4: 25 mil
        *   5: 35 mil
        *   6: 50 mil
        *   7: 75 mil
        *   8: Maior que $ 75 mil



    """)


#dataset
url = "https://raw.githubusercontent.com/josue-macena-jr/diabets-diagnostics/main/datasets/diabetes_binary_5050split_health_indicators_BRFSS2015.csv"
df = pd.read_csv(url)

df = df.astype(int)

# Menu Lateral
st.sidebar.subheader("Filtros para realizar a predição")

#nomedousuário
user_input = st.sidebar.text_input("Digite seu nome:", placeholder="Digite seu nome")


#dados dos usuários com a função
def get_user_date():

    CholAlto = st.sidebar.selectbox("Tem Colesterol Alto?", ("Sim", "Não"))
    if CholAlto == "Sim":
        CholAlto = 1
    else:
        CholAlto = 0

    PressAlta  = st.sidebar.selectbox("É hipertenso?", ("Sim", "Não"))
    if PressAlta == "Sim":
        PressAlta = 1
    else:
        PressAlta = 0

    IMC = st.sidebar.number_input("Índice de massa corporal", min_value=12, max_value=100, value=28)

    # Idade = st.sidebar.slider("Idade", 0, 100, 25)
    Idade = st.sidebar.number_input("Idade", min_value=0, max_value=100, value=30)
    if Idade >=18 and Idade <=24:
        Idade = 1
    elif Idade >=25 and Idade <=29:
        Idade = 2
    elif Idade >=30 and Idade <=34:
        Idade = 3
    elif Idade >=35 and Idade <=39:
        Idade = 4
    elif Idade >=40 and Idade <=44:
        Idade = 5
    elif Idade >=45 and Idade <=49:
        Idade = 6
    elif Idade >=50 and Idade <=54:
        Idade = 7
    elif Idade >=55 and Idade <=59:
        Idade = 8
    elif Idade >=60 and Idade <=64:
        Idade = 9
    elif Idade >=65 and Idade <=69:
        Idade = 10
    elif Idade >=70 and Idade <=74:
        Idade = 11
    elif Idade >=75 and Idade <=79:
        Idade = 12
    elif Idade > 80:
        Idade = 13
    else:
        Idade = 14    


    Sexo = st.sidebar.selectbox("Sexo", ("Masculino", "Feminino"))
    if Sexo == "Masculino":
        Sexo = 0
    else:
        Sexo = 1

    Fumante = st.sidebar.selectbox("É fumante?", ("Sim", "Não"))
    if Fumante == "Sim":
        Fumante = 1
    else:
        Fumante = 0

    Derrame = st.sidebar.selectbox("Já sofreu derrame?", ("Sim", "Não"))
    if Derrame == "Sim":
        Derrame = 1
    else:
        Derrame = 0

    ConsAlcool  = st.sidebar.selectbox("Consome álcool?", ("Sim", "Não"))
    if ConsAlcool == "Sim":
        ConsAlcool = 1
    else:
        ConsAlcool = 0

    Renda = st.sidebar.selectbox("Renda familiar anual", ("10 Mil", "15 Mil", "20 Mil", "25 Mil", "35 Mil", "50 Mil", 
                                    "75 Mil", "Maior que 75 Mil"))
    if Renda == "10 Mil":
        Renda = 1
    elif Renda == "15 Mil":
        Renda = 2
    elif Renda == "20 Mil":
        Renda = 3
    elif Renda == "25 Mil":
        Renda = 4
    elif Renda == "35 Mil":
        Renda = 5
    elif Renda == "50 Mil":
        Renda = 6
    elif Renda == "75 Mil":
        Renda = 7
    elif Renda == "Maior que 75 Mil":
        Renda = 8
    

    #dicionário para receber informações

    user_data = {
                'Colesterol': CholAlto,
                'Pressão Sanguínea': PressAlta,
                'Índice de massa corporal': IMC,
                'Idade': Idade,
                'Sexo': Sexo,
                'Fumante': Fumante,
                'Derrame': Derrame,
                'Consome Alcool': ConsAlcool,
                'Renda': Renda
                }

    features = pd.DataFrame(user_data, index=[0])

    return features

user_input_variables = get_user_date()




# Pagina Principal
# verificando o dataset
st.subheader("Exibindo uma amostra dos dados")


df = df.rename(columns = {'Diabetes_binary':'Diabetes', 
                            'HighBP':'PressAlta',  
                            'HighChol':'CholAlto',
                            'CholCheck':'ColCheck', 
                            'BMI':'IMC', 
                            'Smoker':'Fumante', 
                            'Stroke':'Derrame',
                            'HeartDiseaseorAttack':'CoracaoEnf', 
                            'PhysActivity':'AtivFisica', 
                            'Fruits':'Frutas',
                            'Veggies':"Vegetais", 
                            'HvyAlcoholConsump':'ConsAlcool', 
                            'AnyHealthcare':'PlSaude',
                            'NoDocbcCost':'DespMedica', 
                            'GenHlth':'SdGeral',
                            'MentHlth':'SdMental',
                            'PhysHlth':'SdFisica',
                            'DiffWalk':'DifCaminhar', 
                            'Sex':'Sexo',
                            'Age':'Idade',
                            'Education':'Educacao',
                            'Income':'Renda' })

# atributos para serem exibidos por padrão
# defaultcols = ["HighChol", "HighBP", "BMI", "Age", "Sex", "Smoker", "Stroke", "HvyAlcoholConsump", "Diabetes_binary"]
defaultcols = ["CholAlto", "PressAlta", "IMC", "Idade", "Sexo", "Fumante", "Derrame", "ConsAlcool", "Diabetes"]

# Exibir o dataframe dos chamados
with st.expander("Clique para recolher ou expandir", expanded=True):
    cols = st.multiselect("", df.columns.tolist(), default=defaultcols)
    # Dataframe
    st.dataframe(df[cols])


#dados de entrada
# x = df.drop(['Diabetes_binary'],1)
x = df[["CholAlto", "PressAlta", "IMC", "Idade", "Sexo", "Fumante", "Derrame", "ConsAlcool", "Renda"]]
y = df['Diabetes']


# Criando o objeto e treinando o modelo com Random Forest
arvore = RandomForestClassifier(criterion="entropy", max_depth=70, random_state=0, n_estimators=100)
arvore.fit(x, y)
score = arvore.score(x, y)


#previsão do resultado
prediction = arvore.predict(user_input_variables)

btn_predict = st.sidebar.button("Realizar Predição")

if btn_predict:
    #escrevendo o nome do usuário
    if user_input != "":
        st.write(user_input, ", Os dados selecionados foram:", user_input_variables)

    st.subheader('Previsão:')
    # st.write("Score da previsão:", round(score,2)*100, "%")
    if prediction == 1:
        st.write("**Positivo**")
        st.write("Você tem uma tendência a evoluir para um quadro de diabetes. Sugerimos que procure um médico,", 
        "pratique exercícios e se oriente com um nutricionista.")
    else:
        st.write("Negativo")
        st.write("Aparentemente você não tem as características de uma pessoa com diabetes. Mas não se descuide, faça exames periódicos,",  
        "pratique exercícios e tenha uma boa alimentação.")

# st.write("Score: ", arvore.score(y_test, prediction))









# fonte: https://medium.com/data-hackers/desenvolvimento-de-um-aplicativo-web-utilizando-python-e-streamlit-b929888456a5

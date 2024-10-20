import pandas as pd
import streamlit as st
from dotenv import load_dotenv, dotenv_values
import json
from langfuse.openai import OpenAI
from langfuse.decorators import observe
from pycaret.regression import load_model, predict_model
import instructor
from pydantic import BaseModel
import datetime
from typing import Optional

st.set_page_config(layout='wide')

env = dotenv_values(".env")

# ochrona klucza OpenAI API
if not st.session_state.get("openai_api_key"):
    if "OPENAI_API_KEY" in env:
        st.session_state["openai_api_key"] = env['OPENAI_API_KEY']

    else:
        st.info("Dodaj swój klucz API OpenAI aby móc korzystać z tej aplikacji")
        st.session_state["openai_api_key"] = st.text_input("Klucz API", type="password")
        if st.session_state["openai_api_key"]:
            st.rerun()

if not st.session_state.get("openai_api_key"):
    st.stop()

# wczytuję zmienne środowiskowe i ustawiam klientów OpenAI + łączę się z Langfuse
load_dotenv()
openai_client = OpenAI(api_key=st.session_state["openai_api_key"])
instructor_openai_client = instructor.patch(openai_client)

# dodanie listy z modelami
DEFAULT_MODEL_INDEX=0
models = ['gpt-4o', 'gpt-4o-mini']
if 'model' not in st.session_state:
    st.session_state['model'] = models[DEFAULT_MODEL_INDEX] #jeśli nie ma - przypisz 1 model z listy

# definiowanie LLM do wyłuskiwania danych w formacie JSON
class User(BaseModel):
    age: Optional[int] = None
    gender: Optional[str] = None
    hours_5km: int
    minutes_5km: int
    seconds_5km: int
    total_seconds_5km: int
    hours_10km: int
    minutes_10km: int
    seconds_10km: int
    total_seconds_10km: int
    pace_15km: Optional[float] = None

@observe()
def retrieve_structure_observed(text, response_model):

    messages = [
        {
        'role' : 'user',
        'content' : text,
        },
    ]

    res = instructor_openai_client.chat.completions.create(
        model= st.session_state['model'],
        temperature=0,
        response_model=response_model,
        messages= messages,
        #name='retrieve_structure_observed',
    )

    return res.model_dump()

# wczytuję wcześniej wytrenowany model AI
MODEL_NAME = 'linear_regression_pipeline'

@st.cache_resource
def get_model():
    return load_model(MODEL_NAME)

#
#PASEK BOCZNY
#

with st.sidebar:
    st.write(f'Aktualny model: {st.session_state["model"]}')

    # dodanie opcji wyboru modelu
    selected_model = st.selectbox("Wybrany model", models, index=DEFAULT_MODEL_INDEX)
    st.session_state['model'] = selected_model

#
# MAIN
#

st.title('\U0001f3c3 Półmaraton Estymator')

st.write('---')

if not 'name' in st.session_state:
    st.session_state['name'] = ""

if not 'greeted' in st.session_state:
    st.session_state['greeted'] = False

if not st.session_state['greeted']:
    name = st.text_input('Witaj! Jak masz na imię?', '')
    if name:
        st.session_state['name'] = name
        st.session_state['greeted'] = True
        st.rerun()

if st.session_state['greeted']:
    st.markdown(f'## Cześć {st.session_state["name"]}! :hand:')
    st.markdown('###### Znajdujesz się w aplikacji imitującej estymator półmaratonu biegowego.\n###### Uzupełnij poniższe informacje, a dzięki modelowi AI - dowiesz się ile wyniósłby Twój czas ukończenia półmaratonu.\n###### :arrow_left: Z lewej strony możesz wybrać model AI.')

    with st.expander('Opisz siebie', expanded=False):
        st.markdown('#### Podaj następujące informacje o sobie:\n - wiek\n - płeć\n - czas na 5km\n - czas na 10km\n - tempo w jakim jesteś w stanie przebiec 15km\n ###### <span style="color:red">Przykład:</span> Mam 28 lat, jestem mężczyzną, 5km przebiegam w 22 minuty, 10km w 50 minut, natomiast na 15km utrzymam tempo 5:30 min/km.', unsafe_allow_html=True)
        
        if 'user_data' not in st.session_state:
            st.session_state['user_data'] = ""

        st.session_state['user_data'] = st.text_area('Tutaj wpisz informacje:', value=st.session_state['user_data'])

        if st.button('Estymuj czas przebiegnięcia półmaratonu', use_container_width=True):
        
            if st.session_state['user_data']:
                
                data_for_predict = retrieve_structure_observed(st.session_state['user_data'], User)

                missing_columns = []

                if not data_for_predict['age']:
                    missing_columns.append('wiek')
                if not data_for_predict['gender']:
                    missing_columns.append('płeć')
                if data_for_predict['total_seconds_5km']==0:
                    missing_columns.append('czas na 5km')
                if data_for_predict['total_seconds_10km']==0:
                    missing_columns.append('czas na 10km')
                if not data_for_predict['pace_15km']:
                    missing_columns.append('tempo na 15km')

                if missing_columns:
                    st.error(f'Brakuje informacji: {", ".join(missing_columns)}. Proszę je uzupełnić!')    
                    #st.write(data_for_predict)
                else:
                        user_df = pd.DataFrame({
                            'Wiek' : data_for_predict['age'],
                            'Płeć' : 'M' if 'male' in data_for_predict['gender'].lower() or 'mężczyzna' in data_for_predict['gender'].lower() else 'K',
                            '5km Czas [sek]' : data_for_predict['total_seconds_5km'],
                            '10km Czas [sek]' : data_for_predict['total_seconds_10km'],
                            '15km Tempo [min/km]' : data_for_predict['pace_15km'],
                        }, index=[0])

                        # predykcja na podstawie danych użytkownika
                        model = get_model()
                        prediction = predict_model(model, data=user_df)
                        # wynik predykcji w sekundach
                        prediction_seconds = round(prediction["prediction_label"][0], 2)
                        # wynik predykcji w formacie czasowym: H:M:S
                        prediction_time = str(datetime.timedelta(seconds=int(prediction_seconds)))
                        estimate_result = st.success(f'Estymowany czas ukończenia półmaratonu (w formacie: H:M:S) wynosi {prediction_time}')
                        #st.write(data_for_predict)
            else:
                st.error('Najpierw dodaj informacje o sobie!')

        if st.button('Wyczyść'):
            st.session_state['user_data'] = ""
            st.rerun()
      
        


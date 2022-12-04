import streamlit as st

import tools

st.title("Исследование и прогнозирование потребительской активности в Международном аэропорту Шереметьево")

with st.sidebar:
    step_select = st.selectbox(
        label="Этап",
        options=[
            "Описание задачи",
            "Исследовательский анализ данных",
            "Обучение модели прогнозирования"
        ]
    )


if step_select == "Описание задачи":
    tools.show_description()

if step_select == "Исследовательский анализ данных":
    st.write(tools.read_data())

if step_select == "Обучение модели прогнозирования":
    with st.form("learning_config"):
        col1, col2 = st.columns(2)

        with col1:
            add_weather_data = st.checkbox("Добавить данные о погоде")
            add_mean_revenue = st.checkbox("Добавить среднюю выручку по точке")
            add_info_from_flight_radar = st.checkbox("Добавить информацию с ресурса flightradar24.com")
            add_busydays = st.checkbox("Добавить информацию о выходных днях")

        with col2:
            add_day = st.checkbox("Добавить День из даты в отдельную колонку")
            add_hour = st.checkbox("Добавить Час из даты в отдельную колонку")
            add_minute = st.checkbox("Добавить Минуту из даты в отдельную колонку")

        st.form_submit_button("Применить")

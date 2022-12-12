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
    revenue_05_2022, revenue_06_2022, pass_throw_05, pass_throw_06, sced, airport, airline = tools.read_data()
    st.write("## 1. Посмотрим на доступные данные")

    with st.expander("Показать блок с данными"):
        tools.show_data_samples()

    st.write("## 2. Визуализируем данные")
    with st.expander("Показать блок с визуализацией"):
        tools.visualize()


if step_select == "Обучение модели прогнозирования":
    revenue_05_2022, revenue_06_2022, pass_throw_05, pass_throw_06, sced, airport, airline = tools.read_data()
    user_options = {}

    with st.form("learning_config"):
        col1, col2 = st.columns(2)

        with col1:
            user_options["add_weather_data"] = st.checkbox("Добавить данные о погоде")
            user_options["add_mean_revenue"] = st.checkbox("Добавить среднюю выручку по точке")
            user_options["add_info_from_flight_radar"] = st.checkbox("Добавить информацию с ресурса flightradar24.com")
            user_options["add_busy_days"] = st.checkbox("Добавить информацию о выходных днях")
            user_options["add_aircraft_seats"] = st.checkbox("Добавить информацию о вместимости самолётов")

        with col2:
            user_options["add_day"] = st.checkbox("Добавить День из даты в отдельную колонку")
            user_options["add_hour"] = st.checkbox("Добавить Час из даты в отдельную колонку")
            user_options["add_minute"] = st.checkbox("Добавить Минуту из даты в отдельную колонку")
            user_options["add_period_of_day"] = st.checkbox("Добавить время суток (утро, день, вечер)")

        st.form_submit_button("Обучить и оценить модель")

    st.write(user_options)
    if st.checkbox("Start fit"):
        tools.fit_model(revenue_05_2022, user_options)

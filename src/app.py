import streamlit as st

import tools

st.title("Исследование и прогнозирование потребительской активности в Международном аэропорту Шереметьево")


tools.show_description()

with st.spinner("Загрузка данных"):
    revenue_05_2022, revenue_06_2022, pass_throw_05, pass_throw_06, sced, airport, airline = tools.read_data()
    all_data_list = [revenue_05_2022, revenue_06_2022, pass_throw_05, pass_throw_06, sced, airport, airline]

st.write("### Блок 1: Знакомство с данными")
with st.expander("Показать доступные данные"):
    tools.show_data_samples(3, all_data_list)

st.write("### Блок 2: Анализ данных")
with st.expander("Показать блок анализа данных"):
    tools.visualize(all_data_list)

st.write("### Блок 3: Теория по работе с такими данными")
tools.theory_block()

st.write("### Блок 4: Обучение модели")

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
        user_options["convert_data"] = st.checkbox("Преобразовать дату в отдельные колонки (день, месяц, год)")
        user_options["add_period_of_day"] = st.checkbox("Добавить время суток (утро, день, вечер)")

    user_options["task_type"] = st.selectbox(
        label="На чём обучать модель",
        options=["CPU", "GPU"]
    )

    st.form_submit_button("Применить параметры")


if st.checkbox("Обучить и оценить модель"):
    tools.fit_and_evaluate_model(all_data_list, user_options, task_type=user_options["task_type"])

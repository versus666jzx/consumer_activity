import streamlit as st

import tools

st.title("Исследование и прогнозирование потребительской активности в Международном аэропорту Шереметьево")


tools.show_description()

revenue_05_2022, revenue_06_2022, pass_throw_05, pass_throw_06, sced, airport, airline = tools.read_data()
all_data_list = [revenue_05_2022, revenue_06_2022, pass_throw_05, pass_throw_06, sced, airport, airline]

st.write("### Блок 1: Знакомство с данными")
with st.expander("Показать доступные данные"):
    tools.show_data_samples(3, all_data_list)

st.write("### Блок 2: Анализ данных")
with st.expander("Показать блок анализа данных"):
    tools.visualize(all_data_list)

st.write("### Блок 3: Работа с данными и построение гипотез")
with st.expander("Показать блок:"):
    data, data_stock = tools.hypothesis_and_segmentation_block(all_data_list)

st.write("### Блок 4: Обучение модели")
with st.expander("Показать блок:"):
    tools.fit_model_block()
    if st.checkbox("Обучить и оценить модель"):
        tools.fit_and_evaluate_model(data, data_stock)


st.write("### Блок 5: Выводы и рекомендации")
with st.expander("Показать блок:"):
    tools.conclusions_and_recommendations()

tools.practice_part()


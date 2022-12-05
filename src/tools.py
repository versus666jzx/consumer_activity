from typing import Union, List, Dict, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


def show_description():
    st.write(
        """
        ## Введение

        Международный аэропорт Шереметьево в ежегодном отчете
        Международного совета аэропортов (ACI) признан вторым по пассажиропотоку
        в Европе. Пассажиры, вылетающие из аэропорта, имеют существенную
        потребительскую активность, которая на сегодняшний день анализируется
        недостаточно глубоко. Прогнозирование активности пассажиров в зависимости
        от направления их вылета и времени нахождения в «чистой зоне», где
        сосредоточена значительная часть торговых точек и ресторанов, поможет
        аэропорту формировать эффективные маркетинговые предложения, а также
        корректировать планы по развитию маршрутной сети и инфраструктуры.
        Условие задачи (требования к решению)

        ## Что нужно получить

        Предлагается разработать математическую или ИИ-модель,
        прогнозирующую потребительскую активность в аэропорту в зависимости от
        различных факторов:

        1. На основании данных первого месяца из датасета (файлы
            05.2022_Выручка и 05.2022_Пассажиропоток), построить гипотезы по
            работе с данными и сегментации пассажиропотока в зависимости от
            различных факторов, таких как география рейсов, авиакомпания, время
            ожидания пассажира в «чистой зоне». Вы можете использовать другие
            факторы и (при необходимости) внешние источники данных для
            построения ваших гипотез.

        2. Определить сегменты, авиакомпании и направления, с которыми
            связана максимальная потребительская активность.

        3. Оценить влияние иных факторов, таких как задержки рейсов, погодные
            условия и другие факторы, которые вы можете предложить
            самостоятельно.

        4. Построить прогноз выручки на второй месяц из датасета на основании
            данных пассажиропотока (файл 06.2022_Пассажиропоток).

        5. Выработать рекомендации по увеличению выручки в зависимости от
            доступных факторов и сегментации пассажиропотока.

        ## Дано

        Входные данные:
        - Выручка арендаторов с привязкой к временным интервалам.
        - Информация о плановой и фактической дате вылета, отметка о случае
            отмены рейса.
        - Обезличенная информация о находящихся в «чистой зоне» пассажирах.

        Потенциальные дополнительные источники данных для задачи:
        - Информация с ресурса flightradar24
        - Информация с ресурса Яндекс-расписание
        - Информация о погодных условиях (ресурс GISMETEO)
        - Прочие источники
        """
    )


def show_data_samples():

    # т.к. read_data закеширована, то тут просто будет взят кеш, т.к. данные загружаются первый раз в app.py
    # и не будет повторной загрузки данных с диска
    revenue_05_2022, revenue_06_2022, pass_throw_05, pass_throw_06, sced, airport, airline = read_data()

    st.write("""
            ---
            Данные выручки торговых точек за май:
            """)
    st.table(revenue_05_2022.sample(7))
    st.button("Обновить", key="UPDATE")

    st.write("Статистические данные по выручке за май:")
    st.dataframe(revenue_05_2022["revenue"].describe())

    st.write("""
            ---
            Данные о пассажиропотоке за май:
            """)
    st.table(pass_throw_05.sample(7))
    st.button("Обновить", key="UPDATE 2")

    st.write("""
            ---
            Расписание рейсов за май:
            """)
    st.table(sced.sample(7))
    st.button("Обновить", key="UPDATE 3")

    st.write("""
            ---
            Справочник AIRLINES:
            """)
    st.table(airline.sample(7))
    st.button("Обновить", key="UPDATE 4")

    st.write("""
            ---
            Справочник AIRPORTS:
            """)
    st.table(airport.sample(7))
    st.button("Обновить", key="UPDATE 5")


def visualize():

    # т.к. read_data закеширована, то тут просто будет взят кеш, т.к. данные загружаются первый раз в app.py
    # и не будет повторной загрузки данных с диска
    revenue_05_2022, revenue_06_2022, pass_throw_05, pass_throw_06, sced, airport, airline = read_data()

    st.write("""
    ---
    Данные пассажиропотока по авиакомпаниям:
    """)

    flights = pd.pivot_table(pass_throw_05,
                             index=["Дата рейса", "Рейс", "Авиакомпания", "Направление куда летит", "Терминал"],
                             values=["Вход в чистую зону"],
                             aggfunc="count").reset_index()
    flights.rename(columns={'Вход в чистую зону': 'Кол_пассажиров'}, inplace=True)

    av = flights.groupby('Авиакомпания')['Кол_пассажиров'].sum().reset_index().sort_values(by='Кол_пассажиров',
                                                                                           ascending=False)
    fig = px.bar(av, x='Авиакомпания', y='Кол_пассажиров', text='Кол_пассажиров')
    fig.update_layout(
        xaxis_title_text='Авиакомпания',
        yaxis_title_text='Количество пассажиров',
        bargap=0.2,
        bargroupgap=0.1, font_size=15,
        width=1400
    )
    fig.update_traces(textfont_size=14)

    st.plotly_chart(fig, use_container_width=True)

    st.write("""
    ---
    Данные пассажиропотока по терминалу вылета:
    """)

    fig = px.histogram(
        x=pass_throw_05['Терминал'].value_counts().index,
        y=pass_throw_05['Терминал'].value_counts(),
        text_auto=True
    )

    fig.update_traces(textfont_size=14)

    fig.update_layout(
        xaxis_title_text='Терминал',
        yaxis_title_text='Количество пассажиров',
        bargap=0.2,
        bargroupgap=0.1, font_size=15,
        width=1400
    )

    st.plotly_chart(fig, use_container_width=True)


@st.cache(show_spinner=False)
def read_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame,
                         pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    with st.spinner("Загрузка данных..."):
        revenue_05_2022 = pd.read_excel("data/05.2022_Выручка.xlsx")
        revenue_05_2022 = prepare_revenue(revenue_05_2022)

        revenue_06_2022 = pd.read_excel("data/06.2022_Выручка.xlsx")
        revenue_06_2022 = prepare_revenue(revenue_06_2022)

        pass_throw_05 = pd.read_excel("data/05.2022_Пассажиропоток.xlsx")
        pass_throw_06 = pd.read_excel("data/06.2022_Пассажиропоток.xlsx")

        sced = pd.read_excel("data/Расписание рейсов 05-06.2022.xlsx")

        airport = pd.read_excel("data/Справочник_AIRPORTS.xlsx")
        airline = pd.read_excel("data/Справочник_AIRLINES.xlsx")

    return revenue_05_2022, revenue_06_2022, pass_throw_05, pass_throw_06, sced, airport, airline


def prepare_revenue(revenue: pd.DataFrame) -> pd.DataFrame:
    revenue = revenue.rename(columns={"Дата": "date",
                                      "Дата с разбивкой по 30 минут": "timeThirty",
                                      "Прибыль на момент времени": "revenue",
                                      "Точка продаж": "point",
                                      })

    revenue['revenue'] = revenue['revenue'].apply(lambda x: str(x).replace(u'\xa0', u''))
    revenue['revenue'] = revenue['revenue'].apply(lambda x: str(x).replace(u',', u'.'))
    revenue["revenue"] = revenue["revenue"].astype(float)

    revenue["timeThirty"] = pd.to_datetime(revenue["timeThirty"], utc=True)
    revenue["day_of_week"] = revenue["timeThirty"].dt.dayofweek
    revenue["day_of_year"] = revenue["timeThirty"].dt.dayofyear
    revenue["day"] = revenue["timeThirty"].dt.day
    revenue["month"] = revenue["timeThirty"].dt.month
    revenue["hour"] = revenue["timeThirty"].dt.hour
    revenue["minutes"] = revenue["timeThirty"].dt.minute

    return revenue


def add_weekend_revenue_06(revenue: pd.DataFrame) -> pd.DataFrame:

    revenue["is_weekend"] = np.logical_or(False, revenue["day_of_week"] == 6)
    revenue["is_weekend"] = np.logical_or(revenue["is_weekend"], revenue["day_of_week"] == 7)
    revenue["is_weekend"] = np.logical_or(revenue["is_weekend"], revenue["day"] == 13)
    revenue["is_weekend"] = revenue["is_weekend"].astype(int)

    return revenue


def data_prepare_by_user_choice(user_options: dict, data: pd.DataFrame):
    if user_options["Добавить среднюю выручку по точке"]:
        mean_revenue = (data[["point", "revenue"]]
                        .groupby(["point"])
                        .mean()
                        .sort_values(by="revenue")
                        .reset_index())\
            .rename(columns={"revenue": "mean_revenue"})
        data = data.merge(mean_revenue, how='left', left_on='point', right_on='point')
        del mean_revenue

    if user_options[""]:
        data["is_weekend"] = np.logical_or(False, data["day_of_week"] == 6)
        data["is_weekend"] = np.logical_or(data["is_weekend"], data["day_of_week"] == 7)
        data["is_weekend"] = np.logical_or(data["is_weekend"], data["day"] == 2)
        data["is_weekend"] = np.logical_or(data["is_weekend"], data["day"] == 3)
        data["is_weekend"] = np.logical_or(data["is_weekend"], data["day"] == 9)
        data["is_weekend"] = np.logical_or(data["is_weekend"], data["day"] == 10)
        data["is_weekend"] = data["is_weekend"].astype(int)

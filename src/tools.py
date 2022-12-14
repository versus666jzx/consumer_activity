import catboost
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.metrics import mean_squared_error, mean_absolute_error, max_error, r2_score
from sklearn.model_selection import train_test_split


def show_description():

    st.image("images/people-at-airport.jpg")

    st.write(
        """
        ## Актуальность тематики

        Международный аэропорт Шереметьево в ежегодном отчете
        Международного совета аэропортов (ACI) признан вторым по пассажиропотоку
        в Европе. Пассажиры, вылетающие из аэропорта, имеют существенную
        потребительскую активность, которая на сегодняшний день анализируется
        недостаточно глубоко. Прогнозирование активности пассажиров в зависимости
        от направления их вылета и времени нахождения в «чистой зоне», где
        сосредоточена значительная часть торговых точек и ресторанов, поможет
        аэропорту формировать эффективные маркетинговые предложения, а также
        корректировать планы по развитию маршрутной сети и инфраструктуры.
        """)

    with st.expander("Про потребительскую активность"):
        st.write("""
        
        Поведение потребителей, а также факторы, влияющие на принятие решения 
        о покупке — это то, на чем строится маркетинговая политика большинства 
        компаний на сегодняшний день. Воздействуя на людей с помощью различных 
        инструментов маркетинга, организации выстраивают целую систему.
        
        ##### Основные характеристики потребительского поведения
        
        - Рациональность. Клиент выбирает товар в соответствии со своими вкусами, 
          интересами, потребностями и финансовыми возможностями. Именно поэтому 
          производители стремятся как можно больше расширить ассортимент, предоставить 
          возможность выбора и сравнения продукции одной категории.
        
        - Независимость выбора. Тот случай, когда человек принимает решение о покупке самостоятельно.
        
        - Множественность. Количество предложений находится в прямой зависимости от 
          действий покупателя и наоборот. Учитывая, что сегодня рынок товаров и услуг
          переполнен различными продуктами, которые могут удовлетворить интересы 
          практически любого, поведение потребителей и факторы, определяющие его, 
          становятся с каждым днем все более разнообразными.
        
        """)

    st.write("""
        ## Задача

        Предлагается разработать модель, прогнозирующую потребительскую 
        активность в аэропорту в зависимости от
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

        3. Построить прогноз выручки на второй месяц из датасета на основании
            данных пассажиропотока (файл 06.2022_Пассажиропоток).

        4. Выработать рекомендации по увеличению выручки в зависимости от
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


def show_data_samples(n_samples: int, data_list: list):

    revenue_05_2022, revenue_06_2022, pass_throw_05, pass_throw_06, sced, airport, airline = data_list

    st.write("""
            ---
            Данные выручки торговых точек за май:
            """)
    st.table(revenue_05_2022.sample(n_samples))

    st.button("Обновить", key="UPDATE")
    st.write(f"Количество строк: {revenue_05_2022.shape[0]}")
    st.write(f"Количество столбцов: {revenue_05_2022.shape[1]}")

    col1, _ = st.columns(2)

    with col1:
        st.write("Статистические данные по выручке за май:")
        st.table(revenue_05_2022["revenue"].describe().rename("Выручка"))

    st.write("""
            ---
            Данные о пассажиропотоке за май:
            """)
    st.table(pass_throw_05.sample(n_samples))
    st.write(f"Количество строк: {pass_throw_05.shape[0]}")
    st.write(f"Количество столбцов: {pass_throw_05.shape[1]}")
    st.button("Обновить", key="UPDATE 2")

    st.write("""
            ---
            Расписание рейсов за май:
            """)
    st.table(sced.sample(7))
    st.write(f"Количество строк: {sced.shape[0]}")
    st.write(f"Количество столбцов: {sced.shape[1]}")
    st.button("Обновить", key="UPDATE 3")

    st.write("""
            ---
            Справочник AIRLINES:
            """)
    st.table(airline.sample(n_samples))
    st.write(f"Количество строк: {airline.shape[0]}")
    st.write(f"Количество столбцов: {airline.shape[1]}")
    st.button("Обновить", key="UPDATE 4")

    st.write("""
            ---
            Справочник AIRPORTS:
            """)
    st.table(airport.sample(n_samples))
    st.write(f"Количество строк: {airport.shape[0]}")
    st.write(f"Количество столбцов: {airport.shape[1]}")
    st.button("Обновить", key="UPDATE 5")


def visualize(data_list: list):

    revenue_05_2022, revenue_06_2022, pass_throw_05, pass_throw_06, sced, airport, airline = data_list

    st.subheader("Анализ пассажиропотока за май")

    flights = pd.pivot_table(pass_throw_05,
                             index=["Дата рейса", "Рейс", "Авиакомпания", "Направление куда летит", "Терминал"],
                             values=["Вход в чистую зону"],
                             aggfunc="count").reset_index()
    flights.rename(columns={"Вход в чистую зону": "Кол_пассажиров"}, inplace=True)

    av = flights.groupby("Авиакомпания")["Кол_пассажиров"].sum().reset_index().sort_values(by="Кол_пассажиров",
                                                                                           ascending=False)
    fig = px.bar(av, x="Авиакомпания", y="Кол_пассажиров", text="Кол_пассажиров")
    fig.update_layout(
        title_text="Данные пассажиропотока по авиакомпаниям",
        xaxis_title_text="Авиакомпания",
        yaxis_title_text="Количество пассажиров",
        bargap=0.2,
        bargroupgap=0.1, font_size=15,
        width=1400
    )
    fig.update_traces(textfont_size=14)

    st.plotly_chart(fig, use_container_width=True)
    st.write("""
    Очевидно кратное преимущество компании Аэрофлот - более чем в 9 раз от 
    ближайшего преследователя компании Smartavia.
    """)
    #################################################

    fig = px.histogram(
        x=pass_throw_05["Терминал"].value_counts().index,
        y=pass_throw_05["Терминал"].value_counts(),
        text_auto=True
    )

    fig.update_traces(textfont_size=14)

    fig.update_layout(
        title_text="Данные пассажиропотока по терминалу вылета",
        xaxis_title_text="Терминал",
        yaxis_title_text="Количество пассажиров",
        bargap=0.2,
        bargroupgap=0.1,
        font_size=15,
        width=1400
    )

    st.plotly_chart(fig, use_container_width=True)

    #################################################

    mean_flights = pd.pivot_table(flights,
                                  index=["Рейс", "Авиакомпания", "Направление куда летит", "Терминал"],
                                  values=["Кол_пассажиров", "Дата рейса"],
                                  aggfunc={"Кол_пассажиров": "mean", "Дата рейса": "count"}
                                  ).reset_index()
    mean_flights.rename(columns={"Дата рейса": "Количество рейсов", "Кол_пассажиров": "Ср_кол_пассажиров"},
                        inplace=True)

    top15_pass = mean_flights.sort_values(by=["Ср_кол_пассажиров"], ascending=False).head(15)
    top15_pass["Ср_кол_пассажиров"] = top15_pass["Ср_кол_пассажиров"].round(1)

    fig = px.bar(top15_pass, x="Ср_кол_пассажиров", y="Рейс", text="Ср_кол_пассажиров", orientation="h")
    fig.update_layout(
        title_text="Топ 15 рейсов по среднему числу пассажиров в мае",
        xaxis_title_text="Среднее количество пассажиров",
        yaxis_title_text="",
        bargap=0.2,
        bargroupgap=0.1,
        font_size=15,
        yaxis=dict(autorange="reversed")
    )

    st.plotly_chart(fig, use_container_width=True)

    #################################################

    st.markdown("---")
    st.subheader("Анализ выручки за май")

    day_rev = pd.pivot_table(revenue_05_2022,
                             index=["date", "point"],
                             values=["revenue"],
                             aggfunc="sum"
                             ).reset_index()

    res = (day_rev
           .rename({"revenue": "Выручка за месяц"}, axis=1)
           .groupby("point")["Выручка за месяц"]
           .sum()
           .round(1)
           .astype(int)
           .sort_values(ascending=False)
           )

    fig = px.bar(res[:20])

    fig.update_xaxes(
        tickangle=45
    )

    fig.update_layout(
        title_text="Топ 20 точек по выручке за месяц",
        xaxis_title_text="",
        yaxis_title_text="Выручка",
        bargroupgap=0.1,
        font_size=9,
        showlegend=False
    )

    st.plotly_chart(fig, use_container_width=True)

    #################################################

    data = revenue_05_2022.copy(deep=True)
    data["час_покупки"] = data["timeThirty"].dt.round("H")

    r = data.groupby("час_покупки")["revenue"].sum().reset_index()
    fig = px.bar(r, x="час_покупки", y="revenue", text="revenue")
    fig.update_layout(
        title_text="Почасовая выручка за май",
        xaxis_title_text="Дата",
        yaxis_title_text="Сумма выручки",
        bargap=0.2,
        bargroupgap=0.1,
        font_size=15
    )
    fig.update_traces(marker_color='#cd5a84', textfont_size=14)

    st.plotly_chart(fig)
    st.write("Из гистограммы видно, что выручка падает по ночам и в целом не однородна в течение дня.")

    #################################################

    data["час"] = data["час_покупки"].dt.hour
    mean_r = data.groupby("час")["revenue"].mean().round(0).reset_index()
    mean_hour = mean_r["revenue"].mean()

    fig = px.bar(mean_r, x="час", y="revenue",
                 labels={"revenue": "Средняя выручка в час"}
                 )
    fig.update_layout(
        title_text="Средняя почасовая выручка в мае",
        xaxis_title_text="Час",
        yaxis_title_text="Средняя выручка",
        font_size=15
    )

    fig.add_shape(
        type="line", line_color="salmon", line_width=3, opacity=1, line_dash="dot",
        x0=0, x1=1, xref="paper", y0=mean_hour, y1=mean_hour, yref="y"
    )

    st.plotly_chart(fig, use_container_width=True)
    st.write("""
    Видно, что выручка точек возрастает в утренние, обеденные и вечерние часы, а ночью сильно проседает.
    """)

    #################################################

    data["час"] = data["час_покупки"].dt.hour
    sum_day = data.groupby(["date"])["revenue"].sum().reset_index()
    fig = px.line(sum_day, x="date", y="revenue", markers=True,
                  labels={"date": "Дата", "revenue": "Сумма выручки"}
                  )
    fig.update_traces(textposition="bottom right",
                      marker_color="#cd5a84", line_color="#cd5a84")

    fig.update_layout(
        title_text="Динамика выручки по дням",
        bargap=0.2,
        bargroupgap=0.1,
        font_size=15
    )

    st.plotly_chart(fig, use_container_width=True)
    st.write("""
    Видим слабый восходящий тренд после 11 мая и пиковые значения 6, 20, 27 мая - это пятницы. 
    Отметим это, может пригодиться.
    """)


def hypothesis_and_segmentation_block(all_data_list: list[pd.DataFrame, ...]):
    revenue_05_2022, revenue_06_2022, pass_throw_05, pass_throw_06, sced, airport, airline = all_data_list
    # revenue_05_2022["timeHour"] = revenue_05_2022["timeThirty"].dt.floor("60min")
    # pass_throw_05["timeHour"] = pass_throw_05["Дата рейса"].dt.floor("60min")
    # pass_throw_05["timeHour"] = pd.to_datetime(pass_throw_05["timeHour"], utc=True)
    # res = revenue_05_2022.merge(pass_throw_05, on="timeHour")
    # st.write(revenue_05_2022.shape, res.shape)

    st.write("""
    Нам доступно достаточно много различных данных для построение модели прогнозирования
    выручки торговых точек. Для работы с таким большим количеством данных стоит придерживаться
    определенной концепции:
     - соберем данные в одну таблицу
     - т.к. у нас открытые данные, соберем дополнительные данные из открытых источников
     - создадим новые признаки и имеющихся данных
     - построим модель
     - оценим влияние признаков на целевой признак
     - спрогнозируем целевой признак и оценим модель
    
    """)


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
    revenue["timeHour"] = revenue['timeThirty'].dt.floor('60min')
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


def data_prepare_by_user_choice(revenue_data: pd.DataFrame, user_options: dict):

    if user_options["add_weather_data"]:
        weather = pd.read_csv("data/svo_weather.csv")
        flight_radar_data = pd.read_csv("data/data_from_flightradar24.csv")
        weather["datetime"] = pd.to_datetime(weather["datetime"], utc=True)
        weather = weather.merge(flight_radar_data[["iata", "icao"]], how='left')
        weather_on_svo = weather[weather["iata"] == 'SVO'][["datetime", "temp", "wind_speed", "visibility", "sky_coverage"]]
        weather_on_svo = weather_on_svo.rename(columns={"temp": "SVO_temp",
                                                        "wind_speed": "SVO_wind",
                                                        "visibility": "SVO_visibility",
                                                        "sky_coverage": "SVO_sky_coverage"
                                                        })
        revenue_data = revenue_data.merge(weather_on_svo, left_on='timeHour', right_on='datetime', how='left')

        del weather, flight_radar_data, weather_on_svo

    if user_options["add_mean_revenue"]:
        mean_revenue = (revenue_data[["point", "revenue"]]
                        .groupby(["point"])
                        .mean()
                        .sort_values(by="revenue")
                        .reset_index())\
            .rename(columns={"revenue": "mean_revenue"})
        revenue_data = revenue_data.merge(mean_revenue, how='left', left_on='point', right_on='point')
        del mean_revenue

    if user_options["add_info_from_flight_radar"]:
        flight_radar_data = pd.read_csv("data/data_from_flightradar24.csv")

    if user_options["convert_data"]:
        revenue_data["day"] = revenue_data["date"].dt.day
        revenue_data["month"] = revenue_data["date"].dt.month
        revenue_data["year"] = revenue_data["date"].dt.year

    if user_options["add_busy_days"]:
        revenue_data["is_weekend"] = np.logical_or(False, revenue_data["day_of_week"] == 6)
        revenue_data["is_weekend"] = np.logical_or(revenue_data["is_weekend"], revenue_data["day_of_week"] == 7)
        revenue_data["is_weekend"] = np.logical_or(revenue_data["is_weekend"], revenue_data["day"] == 2)
        revenue_data["is_weekend"] = np.logical_or(revenue_data["is_weekend"], revenue_data["day"] == 3)
        revenue_data["is_weekend"] = np.logical_or(revenue_data["is_weekend"], revenue_data["day"] == 9)
        revenue_data["is_weekend"] = np.logical_or(revenue_data["is_weekend"], revenue_data["day"] == 10)
        revenue_data["is_weekend"] = revenue_data["is_weekend"].astype(int)

    for col_to_drop in ["date", "datetime", "timeThirty", "timeHour", "point"]:
        if col_to_drop in revenue_data.columns:
            del revenue_data[col_to_drop]

    features = revenue_data.drop("revenue", axis=1)
    target = revenue_data["revenue"]

    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.3, random_state=42)

    return X_train, X_test, y_train, y_test


def fit_and_evaluate_model(all_data: list, user_options: dict, task_type: str = "GPU"):
    revenue_05_2022, revenue_06_2022, pass_throw_05, pass_throw_06, sced, airport, airline = all_data
    eval_res = {}

    cbr = catboost.CatBoostRegressor(iterations=100, task_type=task_type, random_state=25)
    X_train, X_test, y_train, y_test = data_prepare_by_user_choice(revenue_05_2022, user_options)
    cbr.fit(X_train, y_train, silent=True)
    preds = cbr.predict(X_test)

    for metrick in [mean_squared_error, mean_absolute_error, max_error, r2_score]:
        if metrick.__name__ == "mean_squared_error":
            eval_res["mean_root_squared_error"] = metrick(y_test, preds, squared=False)
        else:
            eval_res[metrick.__name__] = metrick(y_test, preds)

    st.table(pd.Series(eval_res, name="Значение метрики"))

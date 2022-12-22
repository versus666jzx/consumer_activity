import catboost
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.metrics import mean_squared_error, mean_absolute_error, max_error, r2_score
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from geopy.distance import geodesic


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
        ## Кому будет полезен этот кейс?
        
        - Студентам направлений аналитики и консалтинга
        - Будущим проджект и продакт менеджерам
        - Аналитикам данных
        - Аналитикам рыночного риска
        - Директорам по данным (CDO - Chief Data Officer)
        
        ##### А если у меня другой профиль?
        
        Во-первых, всегда полезно знать о современных технологиях, которые применяются во множестве профессиональных 
        областей. Во-вторых, в ходе выполнения работы вы ознакомитесь с примером решения широко распространенной 
        бизнес-задачи. А для того, чтобы разработать и внедрить в деятельность организации такое решение, 
        требуется вовлечение в команду специалистов разного уровня и разных ролей: как IT-специальностей, 
        так и других профилей.
        
        ## Задача

        Предлагается разработать модель, прогнозирующую потребительскую 
        активность в аэропорту в зависимости от
        различных факторов:

        1. На основании данных мая месяца из датасета (файлы
            05.2022_Выручка и 05.2022_Пассажиропоток), построить гипотезы по
            работе с данными и сегментации пассажиропотока в зависимости от
            различных факторов, таких как география рейсов, авиакомпания, время
            ожидания пассажира в «чистой зоне». Вы можете использовать другие
            факторы и (при необходимости) внешние источники данных для
            построения ваших гипотез.

        2. Определить сегменты, авиакомпании и направления, с которыми
            связана максимальная потребительская активность.

        3. Построить прогноз выручки на конец месяца из датасета.

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
    
    #### Соберем данные в одну таблицу
    
    Т.к. у нас нет единого столбца, по которому мы могли бы объединить все данные, придется делать это
    в несколько шагов:
    1. Все данные, в которых есть время, будем разбивать на получасовые интервалы.
    2. Для предсказания выручки каждой точки продаж преобразуем данные по выручке таким образом,
       чтобы данные по каждой точки продаж находились в отдельной колонке. В дальнейшем мы сможем
       использовать их как целевые колони при обучении модели.
       
       В результате преобразований получим такую таблицу:
    """)

    a = pd.pivot_table(revenue_05_2022, values='revenue', index=['timeThirty'], columns=['point'], aggfunc=np.sum)
    a = a.fillna(0)
    st.table(a.head())

    st.write("""
    3. Данные из справочника AIRPORTS можем объединить с данными пассажиропотока по столбцам IATACODE и Direction.
       Коды ИАТА это трехсимвольный буквенно-цифровой геокод, обозначающий многие аэропорты и 
       мегаполисы по всему миру, определенный Международной ассоциацией воздушного транспорта (IATA).
       
       В результате преобразований получим такую таблицу:
    """)
    airport_new = airport.copy()
    airport_new['NAME'] = airport_new['NAME'].apply(lambda x: x.split('/')[-1])
    airport_new['NAME'] = airport_new['NAME'].apply(lambda x: x.replace(' ', ''))
    df_pass1 = pass_throw_05.merge(airport_new.set_index('IATACODE'), left_on='Направление куда летит', right_on='IATACODE', validate='m:m')
    st.table(df_pass1.sample(5))

    st.write("""    
    4. Преобразуем данные по пассажиропотоку преобразуем аналогично данным по выручке, но в колонках у нас будут
       города, в которые был осуществлен вылет, а значения - количество пассажиров.
       
       В результате преобразований получим такую таблицу:
    """)

    # группируем по направлению вылета (город)
    potok = df_pass1.groupby(['Вход в чистую зону', 'TOWN']).agg({'Рейс': 'count'})
    table = pd.pivot_table(potok, values='Рейс', index=['Вход в чистую зону'], columns=['TOWN'], aggfunc=np.sum)
    table = table.fillna(0)
    # делаем разбивку по полчаса
    potok_inter = table.resample('30T').sum()
    potok_inter = potok_inter.reset_index()
    potok_inter = potok_inter.rename(columns={'CZ_enter': '30_minutes_interval'})

    st.table(potok_inter.sample(5))

    st.write("""
    5. Объединяем данные по пассажиропотоку с данными по выручке и получаем такую таблицу:
    """)

    inter = revenue_05_2022.groupby(['timeThirty'], as_index=False).agg({'revenue': 'sum'})
    potok_inter["Вход в чистую зону"] = pd.to_datetime(potok_inter["Вход в чистую зону"], utc=True)
    df_may = inter.merge(potok_inter, left_on='timeThirty', right_on='Вход в чистую зону').drop("Вход в чистую зону", axis=1)
    may = df_may.merge(a, on='timeThirty', validate='m:m')
    may_stock = df_may.merge(a, on='timeThirty', validate='m:m').drop("timeThirty", axis=1)
    st.table(may.sample(5))

    st.write("""
    Получили данные за каждые полчаса по выручке и количеству пассажиров, вылетающих по разным направлениям.
    
    ---
    
    #### Добавим данные из внешних источников
    
    Т.к. у нас открытые данные, давайте добавим данные из внешних источников, такие как:
     - данные о погоде и данные с сайта flightradar24
    """)
    st.write("Пример данных о погоде:")
    weather = pd.read_csv("data/svo_weather.csv")
    st.write(weather.sample(3))
    st.write("Пример данных с сайта flightradar24:")
    flightradar24 = pd.read_csv("data/data_from_flightradar24.csv")
    st.write(flightradar24.sample(3))
    st.write(" - данные о количестве мест в самолетах")
    seats = pd.read_csv("data/aircraft_seats.csv")
    st.write(seats.sample(3))

    st.write("""
    Данные о погоде и с сайта flightradar24 имеют два общих столбца "iata" и "icao", объединим их по данным столбцам и
    возьмем из данных о погоде только столбцы "datetime", "temp", "wind_speed", "visibility", "sky_coverage".
    
    После добавления данных получим такую таблицу:
    """)

    weather["datetime"] = pd.to_datetime(weather["datetime"], utc=True)
    weather = weather.merge(flightradar24, on="icao", how='left')
    weather_on_svo = weather[weather["iata"] == 'SVO'][["datetime", "temp", "wind_speed", "visibility", "sky_coverage", "lat", "lon", "alt", "country"]]
    weather_on_svo = weather_on_svo.rename(columns={"temp": "SVO_temp",
                                                    "wind_speed": "SVO_wind",
                                                    "visibility": "SVO_visibility",
                                                    "sky_coverage": "SVO_sky_coverage"
                                                    })
    weather["datetime"] = weather["datetime"].dt.floor("30min")
    may = may.merge(weather_on_svo, left_on='timeThirty', right_on='datetime', how='left').drop("datetime", axis=1)
    st.write(may.sample(4))

    st.write("#### Сформулируем гипотезы и создадим новые признаки для их проверки")

    st.write("""
    1. Мы можем из координат полета вычислить дистанцию с помощью библиотеки geopy, таким образом мы сможем проверить
       гипотезу о том, что данный параметр влияет на выручку.
       Для этого нам нужны координаты точки отправления, это аэропорт Шереметьево (его координаты нам известны,
       это lat=55.972641, long=37.414581) и координаты аэропорта назначения (они есть у нас в данных).
    """)

    def calc_distance(coords):
        lat, lon = coords
        SVO_airport = (55.972641, 37.414581)
        try:
            dest_airport = (lat, lon)
            return geodesic(SVO_airport, dest_airport).km
        except:
            return None

    may["distance"] = may[["lon", "lat"]].apply(calc_distance, axis=1)
    st.write(may.sample(4))

    st.write("""
    2. Т.к. мы не можем использовать колонку с датой и временем в обучении, нам следует ее удалить из обучающей выборки,
       но информацию о дате и времени можно сохранить преобразовав ее в несколько отдельных колонок (день, месяц, год,
       час, минута). Таким образом мы проверим гипотезу о том, что выручка зависит от времени. После данного шага
       столбец с датой можно удалить.
       
       После добавления этих данных получим такую таблицу:
    """)
    may["day_of_week"] = may["timeThirty"].dt.dayofweek
    may["day_of_year"] = may["timeThirty"].dt.dayofyear
    may["day"] = may["timeThirty"].dt.day
    may["month"] = may["timeThirty"].dt.month
    may["hour"] = may["timeThirty"].dt.hour
    may["minutes"] = may["timeThirty"].dt.minute
    may = may.drop("timeThirty", axis=1)

    st.write(may.sample(4))
    
    st.write("""
    3. Добавим данные о выходных днях. Таким образом мы проверим гипотезу о том, что выручка в выходные больше, чем
       в другие дни.
       
       После добавления этих данных получим такую таблицу:
    """)
    
    may["is_weekend"] = np.logical_or(False, may["day_of_week"] == 6)
    may["is_weekend"] = np.logical_or(may["is_weekend"], may["day_of_week"] == 7)
    may["is_weekend"] = np.logical_or(may["is_weekend"], may["day"] == 13)
    may["is_weekend"] = may["is_weekend"].astype(int)
    st.write(may.sample(4))

    may["country"] = may["country"].fillna("Unknown")
    may["country"] = may["country"].astype('category')

    return may, may_stock


def fit_model_block():
    st.write("""
        Т.к. в нашем распоряжении данные распределенные во времени, то при обучении модели
        нам случайно перемешивать в фолдах значения всего временного ряда без сохранения 
        его структуры нельзя, иначе в процессе потеряются все взаимосвязи наблюдений друг с другом.
        
        Поэтому при оценке моделей на временных будем использовать [TimeSeriesSplit](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html).
        Суть достаточно проста — начинаем обучать модель на небольшом отрезке временного ряда, от 
        начала до некоторого $t$, делаем прогноз на $t+n$ шагов вперед и считаем ошибку. 
        Далее расширяем обучающую выборку до $t+n$ значения и прогнозируем с $t+n$ до $t+2*n$, так продолжаем 
        двигать тестовый отрезок ряда до тех пор, пока не упрёмся в последнее доступное наблюдение. 
        В итоге получим столько фолдов, сколько $n$ уместится в промежуток между изначальным обучающим 
        отрезком и всей длиной ряда.
        
        
    """)
    st.image("images/time_serise_valid.png", caption="Пример применения TimeSeriesSplit")

    st.write("""
    Теперь, для сравнения, обучим модель CatBoostRegressor на данных до обогащения их из 
    внешних источников, а также новыми признаками и после и посмотрим разницу.
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

    return revenue


def add_weekend_revenue_06(revenue: pd.DataFrame) -> pd.DataFrame:

    revenue["is_weekend"] = np.logical_or(False, revenue["day_of_week"] == 6)
    revenue["is_weekend"] = np.logical_or(revenue["is_weekend"], revenue["day_of_week"] == 7)
    revenue["is_weekend"] = np.logical_or(revenue["is_weekend"], revenue["day"] == 13)
    revenue["is_weekend"] = revenue["is_weekend"].astype(int)

    return revenue


def fit_and_evaluate_model(data: pd.DataFrame, data_stock: pd.DataFrame, task_type: str = "CPU"):

    eval_res = {}
    stock_eval_res = {}

    cbr = catboost.CatBoostRegressor(iterations=500, task_type=task_type, random_state=25)
    cbr_for_stock = catboost.CatBoostRegressor(iterations=500, task_type=task_type, random_state=25)
    ts_split = TimeSeriesSplit(gap=0, max_train_size=None, n_splits=2, test_size=200)
    for train_index, test_index in ts_split.split(data):
        train = data.iloc[train_index]
        test = data.iloc[test_index]
        features_train = train.drop("revenue", axis=1)
        target_train = train["revenue"]
        features_test = test.drop("revenue", axis=1)
        target_test = test["revenue"]

        cbr.fit(features_train, target_train, silent=True, cat_features=["country"])
        preds = cbr.predict(features_test)

    for metrick in [mean_squared_error, mean_absolute_error, max_error, r2_score]:
        if metrick.__name__ == "mean_squared_error":
            eval_res["mean_root_squared_error"] = metrick(target_test, preds, squared=False)
        else:
            eval_res[metrick.__name__] = metrick(target_test, preds)

    for train_index, test_index in ts_split.split(data_stock):
        train = data_stock.iloc[train_index]
        test = data_stock.iloc[test_index]
        features_train = train.drop("revenue", axis=1)
        target_train = train["revenue"]
        features_test = test.drop("revenue", axis=1)
        target_test_stock = test["revenue"]

        cbr_for_stock.fit(features_train, target_train, silent=True)
        preds_stock = cbr_for_stock.predict(features_test)

    for metrick in [mean_squared_error, mean_absolute_error, max_error, r2_score]:
        if metrick.__name__ == "mean_squared_error":
            stock_eval_res["mean_root_squared_error"] = metrick(target_test_stock, preds_stock, squared=False)
        else:
            stock_eval_res[metrick.__name__] = metrick(target_test_stock, preds_stock)

    res = pd.DataFrame(
        data=[
            pd.Series(eval_res),
            pd.Series(stock_eval_res)
        ]
    ).T
    res.columns = ["До обогащения", "После обогащения"]

    st.table(res)

    st.write("""
    График значений выручки на истинных значениях выручки, предсказанных моделью на не обогащенных данных
    и на обогащенных данных.
    """)

    st.line_chart({
        "Предикт по не обогащенным данным": preds_stock,
        "Предикт по обогащенным данным": preds,
        "Истинная выручка": target_test
    })

    st.write("""
    График значений важности признаков модели, обученной на обогащённых данных.
    """)

    df = pd.Series(cbr.feature_importances_, index=cbr.feature_names_, name="Важность признака").sort_values(ascending=False)

    st.bar_chart(
        df
    )
    st.write("Отсортированные значение важностей признаков:")
    st.write(df)


def conclusions_and_recommendations():
    st.write("""
    В результате решения данной задачи, была построена модель прогнозирование потребительской активности в 
    Международном аэропорту Шереметьево на основании данных выручки торговых точек и пассажиропотока,
    а также других факторов, полученных как из открытых источников, так и с помощью feature engineering.
    
    Получив рабочую модель и проанализировав результаты её работы, мы можем сделать выводы
    о том, сработали ли наши гипотезы или нет.
    
    ---
    Гипотезы, которые сработали:
    
    1. Есть зависимость выручки от погоды.
    2. Есть зависимость выручки от времени суток.
    3. Есть слабая зависимость выручки от выходных дней.
    
    ---
    Гипотезы, которые не сработали:
    1. Отсутствует зависимость выручки от длины полета.
    
    ---
    Также мы видим, что на выручку одни точки продаж влияют значительно сильнее других, это может
    быть следствием плохой мобильности пассажиров между терминалами и внутри терминалов аэропорта.
    
    """)
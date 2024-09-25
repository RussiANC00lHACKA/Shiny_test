from shiny import App, ui, reactive, render
import numpy as np
import plotly.express as px
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
import tempfile
import pandas as pd

# Загрузка данных
iris = load_iris()
X = iris.data[:, :2]  # Используем только два признака для простоты
y = iris.target

# Интерфейс пользователя
app_ui = ui.page_fluid(
    ui.panel_title("Панель ввода для k-NN"),
    
    # Панель ввода
    ui.input_numeric("x1", "Признак 1 (например, длина чашелистика):", value=5.0),
    ui.input_numeric("x2", "Признак 2 (например, ширина чашелистика):", value=3.0),
    ui.input_slider("k", "Количество соседей (k):", min=1, max=10, value=3),
    ui.input_checkbox("normalize", "Нормализовать данные?", value=True),
    ui.input_action_button("run", "Рассчитать"),
    
    # Добавляем вывод результата сразу под кнопкой "Рассчитать"
    ui.output_ui("result"),  # Используем output_ui для возможности форматирования
    
    # Делаем разметку с двумя колонками: одна для графика, другая для таблицы
    ui.row(
        ui.column(6, ui.output_image("plot")),  # Левая колонка для графика
        ui.column(6, 
            ui.h4("Таблица ближайших соседей"),  # Добавляем заголовок для таблицы
            ui.output_table("neighbors_table")   # Правая колонка для таблицы
        )
    )
)

# Логика приложения
def server(input, output, session):
    # Реактивная функция, которая срабатывает при нажатии кнопки
    @reactive.event(input.run)
    def perform_calculation():
        # Получаем введенные данные
        x1 = input.x1()
        x2 = input.x2()
        k = input.k()
        normalize = input.normalize()
        
        # Новый экземпляр для классификации
        new_point = np.array([[x1, x2]])
        
        # Нормализация данных, если выбрано
        if normalize:
            X_min = X.min(axis=0)
            X_max = X.max(axis=0)
            X_norm = (X - X_min) / (X_max - X_min)
            new_point_norm = (new_point - X_min) / (X_max - X_min)
        else:
            X_norm = X
            new_point_norm = new_point

        # Создаем и обучаем модель k-NN
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_norm, y)
        
        # Выполняем классификацию
        prediction = knn.predict(new_point_norm)
        predicted_class = iris.target_names[prediction][0]
        
        # Найдем ближайших соседей
        distances, indices = knn.kneighbors(new_point_norm)
        nearest_neighbors = X_norm[indices[0]]

        # Создаем таблицу с информацией о ближайших соседях
        neighbors_info = {
            "Признак 1": nearest_neighbors[:, 0],
            "Признак 2": nearest_neighbors[:, 1],
            "Класс": iris.target_names[y[indices[0]]]
        }
        neighbors_df = pd.DataFrame(neighbors_info)

        # Возвращаем результат
        return predicted_class, new_point_norm, X_norm, nearest_neighbors, knn, neighbors_df

    # Связываем результат с интерфейсом
    @output
    @render.ui
    def result():
        predicted_class, _, _, _, _, _ = perform_calculation()
        # HTML форматирование текста
        return ui.markdown(f"<h3 style='color: green;'>Новый экземпляр классифицирован как: <b>{predicted_class}</b></h3>")

    # Визуализация графика через сохранение в файл
    @output
    @render.image
    def plot():
        # Здесь распакуем все 6 значений
        predicted_class, new_point_norm, X_norm, nearest_neighbors, knn, _ = perform_calculation()

        # Построение границ принятия решения
        h = 0.02  # Шаг сетки для визуализации
        x_min, x_max = X_norm[:, 0].min() - 1, X_norm[:, 0].max() + 1
        y_min, y_max = X_norm[:, 1].min() - 1, X_norm[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
        Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        # Построение графика с Plotly
        fig = px.scatter(x=X_norm[:, 0], y=X_norm[:, 1], color=y.astype(str),
                         labels={'x': 'Признак 1', 'y': 'Признак 2', 'color': 'Класс'},
                         title="Распределение данных и новый экземпляр")
        
        # Добавляем новый экземпляр на график (фиолетовая большая точка)
        fig.add_scatter(x=[new_point_norm[0][0]], y=[new_point_norm[0][1]], mode='markers', 
                        marker=dict(size=15, color='purple'), name='Новый экземпляр')
        
        # Добавляем ближайших соседей на график с выделением желтым цветом
        fig.add_scatter(x=nearest_neighbors[:, 0], y=nearest_neighbors[:, 1], mode='markers', 
                        marker=dict(size=10, color='yellow'), name='Ближайшие соседи')

        # Сохраняем график в файл с использованием временной директории
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmpfile:
            fig.write_image(tmpfile.name)
            return {"src": tmpfile.name, "alt": "График k-NN"}

    # Таблица с информацией о ближайших соседях
    @output
    @render.table
    def neighbors_table():
        _, _, _, _, _, neighbors_df = perform_calculation()
        return neighbors_df

# Создаем приложение
app = App(app_ui, server)

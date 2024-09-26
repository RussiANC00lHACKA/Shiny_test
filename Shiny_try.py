from shiny import App, ui, reactive, render
import numpy as np
import plotly.express as px
import pandas as pd

# Загрузка данных
iris_data = {
    'data': np.array([
        [5.1, 3.5], [4.9, 3.0], [4.7, 3.2], [4.6, 3.1], [5.0, 3.6],
        [5.4, 3.9], [4.6, 3.4], [5.0, 3.4], [4.4, 2.9], [4.9, 3.1],
        [5.4, 3.7], [4.8, 3.4], [5.8, 2.7], [5.6, 3.0], [5.1, 3.3],
        [5.7, 2.8], [5.1, 3.8], [5.4, 3.4], [5.1, 3.5], [5.5, 2.4],
        [6.5, 2.8], [6.2, 2.9], [5.6, 2.9], [5.8, 4.0], [6.0, 2.2],
        [5.6, 3.0], [5.7, 2.8], [5.4, 3.4], [5.5, 2.5], [6.1, 3.0],
        [6.0, 2.7], [5.8, 4.1], [6.4, 2.8], [6.6, 3.0], [6.8, 2.8],
        [6.7, 3.0], [6.0, 2.9], [5.7, 3.8], [5.5, 2.4], [5.6, 2.7],
        [5.9, 3.2], [6.0, 2.9], [6.1, 2.9], [5.6, 3.0], [6.3, 2.5],
        [6.5, 3.0], [7.6, 3.0], [6.8, 2.8], [6.7, 3.1], [6.1, 3.0],
    ]),
    'target': np.array([
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        2, 2, 2, 2, 2, 2, 2, 2, 2, 2
    ]),
    'target_names': np.array(['Setosa', 'Versicolor', 'Virginica'])
}

# Функция для расчета расстояний
def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2, axis=1))

# Реализация k-NN
def knn_classifier(X, y, new_point, k):
    distances = euclidean_distance(X, new_point)
    indices = np.argsort(distances)[:k]  # Найдем индексы k ближайших соседей
    nearest_labels = y[indices]
    
    # Возвращаем класс с наибольшим числом соседей
    unique, counts = np.unique(nearest_labels, return_counts=True)
    majority_class = unique[np.argmax(counts)]
    
    return majority_class, indices  # Возвращаем также индексы

# Интерфейс пользователя
app_ui = ui.page_fluid(
    ui.panel_title("Панель ввода для k-NN"),
    
    # Панель ввода
    ui.input_numeric("x1", "Признак 1 (например, длина чашелистика):", value=5.0),
    ui.input_numeric("x2", "Признак 2 (например, ширина чашелистика):", value=3.0),
    ui.input_slider("k", "Количество соседей (k):", min=1, max=10, value=3),
    ui.input_action_button("run", "Рассчитать"),
    
    ui.output_ui("result"),
    ui.row(
        ui.column(6, ui.output_ui("plot")),
        ui.column(6, ui.h4("Таблица ближайших соседей"), ui.output_table("neighbors_table"))
    )
)

# Логика приложения
def server(input, output, session):
    @reactive.event(input.run)
    def perform_calculation():
        x1 = input.x1()
        x2 = input.x2()
        k = input.k()
        
        new_point = np.array([[x1, x2]])
        predicted_class, indices = knn_classifier(iris_data['data'], iris_data['target'], new_point, k)
        
        # Находим ближайших соседей
        nearest_neighbors = iris_data['data'][indices]
        neighbors_labels = iris_data['target'][indices]
        neighbors_names = iris_data['target_names'][neighbors_labels]
        
        # Создаем таблицу соседей
        neighbors_df = pd.DataFrame({
            'Признак 1': nearest_neighbors[:, 0],
            'Признак 2': nearest_neighbors[:, 1],
            'Класс': neighbors_names  # Используем имена классов
        })

        return predicted_class, new_point, nearest_neighbors, neighbors_df

    @output
    @render.ui
    def result():
        predicted_class, _, _, _ = perform_calculation()
        return ui.markdown(f"<h3 style='color: green;'>Новый экземпляр классифицирован как: <b>{iris_data['target_names'][predicted_class]}</b></h3>")

    @output
    @render.ui
    def plot():
        predicted_class, new_point, nearest_neighbors, _ = perform_calculation()

        fig = px.scatter(
            x=iris_data['data'][:, 0], 
            y=iris_data['data'][:, 1], 
            color=iris_data['target_names'][iris_data['target']],  # Используем имена классов
            labels={'x': 'Признак 1', 'y': 'Признак 2', 'color': 'Класс'},
            title="Распределение данных и новый экземпляр",
            width=700,  # Ширина графика
            height=500  # Высота графика
        )

        # Добавляем новый экземпляр на график
        fig.add_scatter(
            x=[new_point[0][0]], 
            y=[new_point[0][1]], 
            mode='markers', 
            marker=dict(size=15, color='purple', symbol='star'), 
            name='Новый экземпляр'
        )

        # Добавляем ближайшие соседи на график
        if nearest_neighbors.size > 0:
            fig.add_scatter(
                x=nearest_neighbors[:, 0], 
                y=nearest_neighbors[:, 1], 
                mode='markers', 
                marker=dict(size=10, color='yellow', symbol='x'), 
                name='Ближайшие соседи'
            )

        # Убираем цветовую шкалу
        fig.update_layout(coloraxis_showscale=False)

        return ui.HTML(fig.to_html(full_html=False))

    @output
    @render.table
    def neighbors_table():
        _, _, _, neighbors_df = perform_calculation()
        return neighbors_df

# Создаем приложение
app = App(app_ui, server)

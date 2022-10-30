import numpy as np
from sklearn.model_selection import train_test_split
import copy
from typing import NoReturn


# Task 1

class Perceptron:
    def __init__(self, iterations: int = 100):
        """
        Parameters
        ----------
        iterations : int
        Количество итераций обучения перцептрона.

        Attributes
        ----------
        w : np.ndarray
        Веса перцептрона размерности X.shape[1] + 1 (X --- данные для обучения), 
        w[0] должен соответстовать константе, 
        w[1:] - коэффициентам компонент элемента X.

        Notes
        -----
        Вы можете добавлять свои поля в класс.
        
        """

        self.iterations = iterations
        self.w = None
        self.ymin = None
        self.ymax = None
    
    
    def _fit_step(self, X: np.ndarray, y: np.ndarray):
        pred = np.sign(X @ self.w.T)
        neq = y != pred
        self.w += y[neq] @ X[neq, :]
    
    
    def _fit(self, X: np.ndarray, y: np.ndarray):
        self.w = np.zeros(X.shape[1])
        for _ in range(self.iterations):
            self._fit_step(X, y)
    
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Обучает простой перцептрон. 
        Для этого сначала инициализирует веса перцептрона,
        а затем обновляет их в течении iterations итераций.
        
        Parameters
        ----------
        X : np.ndarray
            Набор данных, на котором обучается перцептрон.
        y: np.ndarray
            Набор меток классов для данных.
        
        """
        self.ymin = y.min()
        self.ymax = y.max()
        yh = (y != self.ymin) * 2 - 1
        Xh = np.hstack((np.tile(1, (X.shape[0], 1)), X))
        self._fit(Xh, yh)
    
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Предсказывает метки классов.
        
        Parameters
        ----------
        X : np.ndarray
            Набор данных, для которого необходимо вернуть метки классов.
        
        Return
        ------
        labels : np.ndarray
            Вектор индексов классов 
            (по одной метке для каждого элемента из X).
        
        """
        real = self.w[0] + X @ self.w[1:].T
        return (np.sign(real) + 1) // 2 * (self.ymax - self.ymin) + self.ymin

# Task 2

class PerceptronBest(Perceptron):
    def __init__(self, iterations: int = 100):
        super().__init__(iterations)
        self.w_best = None
        self.neq_best = None
    
    
    def _fit_step(self, X: np.ndarray, y: np.ndarray):
        pred = np.sign(X @ self.w.T)
        neq = y != pred
        
        neq_sum = neq.sum()
        if neq_sum < self.neq_best:
            self.w_best[:] = self.w[:]
            self.neq_best = neq_sum
        
        self.w += y[neq] @ X[neq, :]
    
    
    def _fit(self, X: np.ndarray, y: np.ndarray):
        self.w_best = np.empty(X.shape[1])
        self.neq_best = y.shape[0] + 1
        super()._fit(X, y)
        self.w = self.w_best
    
# Task 3

def transform_images(images: np.ndarray) -> np.ndarray:
    """
    Переводит каждое изображение в вектор из двух элементов.
        
    Parameters
    ----------
    images : np.ndarray
        Трехмерная матрица с черное-белыми изображениями.
        Её размерность: (n_images, image_height, image_width).

    Return
    ------
    np.ndarray
        Двумерная матрица с преобразованными изображениями.
        Её размерность: (n_images, 2).
    """
    
    # Будем перебирать фичи вида "сумма по строке", "сумма по столбцу", "среднее в прямоугольнике", пока не получится что-то хорошее.
    # Что-то хорошее получилось на сумме пикселей по верхней 1/8 картинки и 4/8 .. 6/8 от высоты.
    # Поскольку сумма и среднее прямо пропорциональны, то они взаимозаменяемы.
    _, ih, iw = images.shape
    sv = images[:, : ih // 8, :].mean(axis=(1, 2))
    sh = images[:, 4 * ih // 8 : 6 * ih // 8, :].mean(axis=(1, 2))
    return np.stack((sv, sh)).T
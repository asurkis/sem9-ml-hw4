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
    
    
    def _fit_step(self, X: np.ndarray, y: np.ndarray):
        pred = np.sign(X @ self.w.T)
        neq = y != pred
        self.w[:] += np.sum(y[neq].reshape((-1, 1)) * X[neq, :], axis=0)
    
    
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
        yh = 2 * y - 1
        Xh = np.hstack((np.tile(1, (X.shape[0], 1)), X))
        self._fit(Xh, yh)
    
    
    def _predict(self, X: np.ndarray) -> np.ndarray:
        real = self.w[0] + X @ self.w[1:].T
        return np.sign(real, casting='unsafe', dtype=np.int64)
    
    
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
        return (self._predict(X) + 1) // 2

# Task 2

class PerceptronBest:

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
    
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Обучает перцептрон.

        Для этого сначала инициализирует веса перцептрона, 
        а затем обновляет их в течении iterations итераций.

        При этом в конце обучения оставляет веса, 
        при которых значение accuracy было наибольшим.
        
        Parameters
        ----------
        X : np.ndarray
            Набор данных, на котором обучается перцептрон.
        y: np.ndarray
            Набор меток классов для данных.
        
        """
        yh = 2 * y - 1
        Xh = np.hstack((np.tile(1, (X.shape[0], 1)), X))
        self.w = np.zeros(Xh.shape[1])
        
        w_best = None
        neq_best = yh.shape[0] + 1
        
        for _ in range(self.iterations):
            pred = np.sign(Xh @ self.w.T)
            neq = yh != pred
            
            neq_sum = neq.sum()
            if neq_sum < neq_best:
                w_best = np.copy(self.w)
                neq_best = neq_sum
            
            self.w[:] += np.sum(yh[neq].reshape((-1, 1)) * Xh[neq, :], axis=0)
        
        self.w = w_best
    
    
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
        return (np.sign(real) + 1) // 2
    
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
    return np.zeros((images.shape[0], 2))
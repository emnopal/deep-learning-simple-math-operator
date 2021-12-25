import numpy as np
import pandas as pd


class Generator:

    def __init__(self, num=10, operation='add'):
        if num * num > 10_000_000:
            raise ValueError(f"I'm not sure your PC can handle bigger than {num * num}, so i threw this error!")
        self.X = num
        self.Y = num
        self.operation = operation

    def generate_X(self, include_zeros=False):
        if include_zeros:
            return np.array([i for i in range(self.X+1) for j in range(self.Y+1)])
        return np.array([i for i in range(1, self.X+1) for j in range(1, self.Y+1)])

    def generate_Y(self, include_zeros=False):
        if include_zeros:
            return np.array([j for i in range(self.X+1) for j in range(self.Y+1)])
        return np.array([j for i in range(1, self.X+1) for j in range(1, self.Y+1)])

    def generate_number(self, include_zeros=False):
        return self.generate_X(include_zeros=include_zeros), self.generate_Y(include_zeros=include_zeros)

    def generate_add(self, include_zeros=False):
        a, b = self.generate_number(include_zeros=include_zeros)
        return np.array(a+b)

    def generate_subtract(self, include_zeros=False):
        a, b = self.generate_number(include_zeros=include_zeros)
        return np.array(a-b)

    def generate_multiply(self, include_zeros=False):
        a, b = self.generate_number(include_zeros=include_zeros)
        return np.array(a*b)

    def generate_divide(self):
        a, b = self.generate_number(include_zeros=False)
        return np.array(a/b)

    def generate_dataframe(self, X, y, results):
        return pd.DataFrame({'X': X, 'Y': y, 'Result': results})

    def generate(self, include_zeros=False):
        X, y = self.generate_number(include_zeros=include_zeros)
        if self.operation == 'divide':
            X, y = self.generate_number(include_zeros=False)
            results = self.generate_divide()
        elif self.operation == 'add':
            results = self.generate_add(include_zeros=include_zeros)
        elif self.operation == 'multiply':
            results = self.generate_multiply(include_zeros=include_zeros)
        elif self.operation == 'subtract':
            results = self.generate_subtract(include_zeros=include_zeros)
        else:
            raise ValueError('Operation not supported')
        return self.generate_dataframe(X, y, results)

    def split_to_Xy(self, dataframe):
        return dataframe[['X', 'Y']], dataframe['Result']


def generate_dataset(operation, num=100, include_zeros=True):
    gene = Generator(num=num, operation=operation)
    a = gene.generate(include_zeros=include_zeros)
    X, y = gene.split_to_Xy(a)
    return X, y

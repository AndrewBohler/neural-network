from collections import namedtuple
import colorama
from colorama import Fore, Style
import itertools
import json
import numpy as np
import math
import random
import sys
import time
from typing import List, Tuple, Callable, Union, Type, Dict, Optional

Activation = Union[Callable, str] # str is a key that maps to func
Derivative = Callable

# round inf to this number
INF = np.finfo(float).max
NEGINF = np.finfo(float).min

def sigmoid(x: np.array, derivative=False):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x: np.array):
    sig = sigmoid(x)
    return sig * (1 - sig)


def mean_squared_error(y_hat, y):
    return np.sqrt(((y_hat - y)**2).mean())

def mean_log_error(y_hat: np.ndarray, y: np.ndarray):
    cost = y * np.log(y_hat) - (1 - y) * np.log(1 - y_hat)
    return np.nan_to_num(cost, copy=False, neginf=NEGINF).mean(axis=0)


class NeuralNetwork:
    def __init__(self,
    layers: List[Tuple[int, Activation, Derivative]],
    learning_rate: float = 0.001,
    loss_func: Callable = mean_squared_error,
    output_func: Callable = lambda output: output,
    ):
        if len(layers) < 3:
            msg = f'{layers} needs to be 3 or more layers'
            raise ValueError(msg)
        
        self.network = layers
        self.weights = [
            np.random.randn(a[0], b[0]) * np.sqrt(2 / a[0])
            for a, b in zip(self.network[:-1], self.network[1:])
        ]
        self.layers = [None for layer in layers]
        self.activations = [layer[1] for layer in layers] # first activation should be None
        self.derivatives = [layer[2] for layer in layers]
        self.learning_rate = learning_rate
        self.loss_function = loss_func
        self.output_function = output_func

    @property
    def output(self) -> np.ndarray:
        return self.output_function(self.layers[-1])

    @property
    def raw_output(self) -> np.ndarray:
        return self.layers[-1]

    @property
    def loss(self) -> float:
        return self.loss_function(self.raw_output)

    def __repr__(self) -> str:
        name = self.__class__.__name__
        n = str([
            (l[0], l[1].__name__)
            for l in self.network
        ]).replace("'", "")
        lr = self.learning_rate

        return f'{name}({n}, learning_rate={lr})'
    
    def __str__(self) -> str:
        layers = "\n".join([
            ', '.join([
                f'layer {i}: {layer[0]}',
                f'{layer[1].__name__ if callable(layer[1]) else layer[1]}',
                f'{layer[2].__name__ if callable(layer[2]) else layer[2]}'
            ])
            for i, layer in enumerate(self.network)
        ])
        lr = self.learning_rate
        return f'{layers}\nlearning_rate: {lr}'

    # def __radd__(self, other) -> object:
    #     return __add__(other)

    def __add__(self, other) -> object:
        if type(other) == tuple:
            assert len(other) > 0
            assert other[0] is int
            assert callable(other[1]) or other[1] in self.act_funcs
            network = []
            network.extend(self.network)
            network.append(other)
            len_weights = len(self.weights)
            new_nn = self.__class__(network, learning_rate=self.learning_rate)
            for i, w in enumerate(self.weights):
                new_nn.weights[i] = np.array(w)

            return new_nn

        elif other.__class__ == self.__class__:
            network = []
            network.extend(self.network)
            network.extend(other.network)
            learning_rate = other.learning_rate
            new_nn = self.__class__(network, learning_rate=learning_rate)
            for i, w in enumerate(self.weights):
                new_nn.weights[i] = np.array(w)

            for i, w in enumerate(other.weights, start=len(self.weights)):
                new_nn.weights[i] = np.array(w)

            return new_nn

        else:
            raise TypeError(f'cannot add types {type(self)} and {type(other)}')

    def __len__(self) -> int:
        return len(self.layers)
        

    def feedforward(self, X) -> np.ndarray:
        """returns the output layer"""
        self.layers[0] = X
        b = 0 # unimplimented bias
        for l, W in enumerate(self.weights, start=1):
            A = self.layers[l-1]
            Z = self.activations[l](np.dot(A, W)) + b # bias correct?
            self.layers[l] = Z
        return self.layers[-1]

    def backprop(self, cost: Union[float, np.ndarray], function='gradient descent') -> None:
        
        def gradient_descent() -> None:
            dA = cost
            dZ = None
            dW = None
            for l in range(len(self.weights)-1, 0, -1): # skip layer0 (input)
                dZ = dA * self.derivatives[l](self.layers[l+1])
                dW = self.layers[l].T.dot(dZ)
                dA = dZ.dot(self.weights[l].T)
                self.weights[l] -= dW * self.learning_rate

        def binary_classification():
            raise NotImplementedError

        def _sigmoid(y, h) -> float:
            return (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()

        loss_func = {
            'gradient descent': gradient_descent,
            # 'binary': binary_classification,
        }

        loss_func[function]()

    def train(self,
        X: np.ndarray,
        Y: np.ndarray,
        batches: int,
        verbose=1,
        test_interval: int = None,
        print_interval: float = 1.0, # seconds
        iteration_number: int = None, # optional value to output during run
    ) -> None:
        start_time = time.perf_counter()
        b_width = int(np.log10(batches-1)) + 1
        batch = 0
        cost = self.loss_function(self.feedforward(X), Y)
        cost_hist = cost
        prediction = self.feedforward(X)

        def print_batch() -> None:
            nonlocal cost
            nonlocal cost_hist
            avg = cost.mean()
            avg_h = cost_hist.mean()
            change = abs(avg) - abs(avg_h)
            direction = change < 0
            color = Fore.GREEN if direction else Fore.RED
            arrow = '↑' if not direction else '↓'
            b_num = f'{batch}'

            correct_elements = Y == self.output
            correct_count = (correct_elements.sum(axis=1) == Y.shape[1]).sum()
            if iteration_number is not None:
                print(f'{[iteration_number]}', end=' ')
            print(f'{batch:>{b_width}}/{batches}', end=' ')
            print(f'average cost: {Fore.YELLOW}{avg:.4e}{Fore.RESET}', end=' ')
            print(f'({color}{arrow} {abs(change):.4e}{Fore.RESET})', end=' ')
            print(f'Correct predictions: {correct_count}/{prediction.shape[0]}', end=' ')
            print()
            
        print_batch()
        print_time = time.perf_counter()
        # train
        while batch < batches:
            prediction = self.feedforward(X)
            cost = self.loss_function(prediction, Y)
            self.backprop(cost)
            neg, pos = [], []
            for weight in self.weights:
                for w in weight.flat:
                    if w > 0: pos.append(w)
                    elif w < 0: neg.append(w)

            # sys.stdout.write(f'\ravg pos w: {np.mean(pos):>10.3f}, avg neg w: {np.mean(neg):>10.3f}')
            # sys.stdout.flush()

            if time.perf_counter() - print_time > print_interval:
                print_batch()
                cost_hist = cost
                print_time = time.perf_counter()

            batch += 1
        if not iteration_number is None:
            print(f'[{iteration_number}]', end=' ')
        print(f'completed {batches:,} batches in {time.perf_counter() - start_time:.2f} seconds')

    
    def compute(self, X) -> np.array:
        """returns self.output"""
        self.feedforward(X)
        return self.output

if __name__ == '__main__':
    print('import NeuralNetwork to use in your script')
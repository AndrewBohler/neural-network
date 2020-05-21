from itertools import chain
import numpy as np
import colorama
from colorama import Fore, Style

from neural_network import NeuralNetwork, sigmoid, sigmoid_derivative, mean_log_error

# binary operation to apply,
# if this is None then it will ask in the terminal
OP = None

INPUT_BITS = 4 # per operand
HIDDEN_LAYERS =[
    # (neurons, activation, derivative)
    (INPUT_BITS*2, sigmoid, sigmoid_derivative),
]
OUTPUT_BITS = 4 # dependant on operation
OUTPUT_ACTIVATION = sigmoid
OUTPUT_DERIVATIVE = sigmoid_derivative

BATCH_SIZE = 2**INPUT_BITS
BATCHES = 100000
MINI_BATCH_SIZE = None # not implimented
MINI_BATCHES = None # not implimented
LEARNING_RATE = 1e-6
LOSS_FUNCTION = mean_log_error



def int_to_bin(*args: int, bitwidth: int) -> np.ndarray:
    """returns an array with shape (len(args), bitwidth)"""
    b = np.zeros((len(args), bitwidth), dtype=int)
    
    for i, arg in enumerate(args):
        n = format(arg, 'b')
        
        if len(n) > bitwidth:
            msg = f'{n} is more than {bitwidth} bits wide'
            raise OverflowError(msg)

        n = '0' * (bitwidth - len(n)) + n

        for j, char in enumerate(n):
            b[i, j] = int(char)

    return b


BIN_OPS = {
    'ADD' : lambda x, y: x + y,
    'SUB' : lambda x, y: x - y,
    'AND' : lambda x, y: x & y,
    'OR'  : lambda x, y: x | y,
    'XOR' : lambda x, y: x ^ y,
    'NAND': lambda x, y: int(
        ''.join([
            '1' if not (a=='0' and b=='0') else '0'
            for a, b in zip(
                int_to_bin(x, INPUT_BITS),
                int_to_bin(y, INPUT_BITS)
            )
        ]),
        2 # bytes
    ),
}

def user_set_operation():
    global OP
    choice = ''
    print('Operation not set, please choose an operation.')
    while choice not in BIN_OPS:
        print('Operations:', ' | '.join(BIN_OPS))
        choice = input('type "exit" to quit> ').upper()
        if choice == 'EXIT':
            quit()

        elif choice in BIN_OPS:
            OP = choice
            break

        print('operation not found...')


def two_num_permutations(start=0, end=2**INPUT_BITS):
    """start=0 (default) implies unsigned ints"""
    while True:
        for a in range(start, end):
            for b in range(start, end):
                yield (a, b)


def test():
    nn = NeuralNetwork(
        [
            (INPUT_BITS*2, None, None),
            *HIDDEN_LAYERS,
            (OUTPUT_BITS, OUTPUT_ACTIVATION, OUTPUT_DERIVATIVE),
        ],
        learning_rate=1e-6,
        loss_func=mean_log_error,
        output_func=lambda output: np.around(output).astype(int),
    )

    print(Fore.CYAN, 'using:', Fore.YELLOW)
    print(nn, Fore.CYAN)
    print(f'to calculate {INPUT_BITS}-bit {OP}', Fore.RESET)
    
    nums = two_num_permutations()
    training_set = [next(nums) for _ in range(BATCH_SIZE)]

    training_batch = np.stack([
        int_to_bin(*t, bitwidth=INPUT_BITS).reshape(INPUT_BITS*2)
        for t in training_set
    ])
    training_ans = int_to_bin(*[
        BIN_OPS[OP](a, b)
        for a, b in training_set
    ], bitwidth=INPUT_BITS)

    iteration = 0

    while True:
        iteration += 1
        nn.train(
            training_batch,
            training_ans,
            BATCHES,
            # test_interval=1000,
            # testX=confirmation_batch,
            # testY=confirmation_ans,
            iteration_number=iteration,
        )

        prediction = nn.raw_output
        print(f'[{iteration}]', end=' ')
        print(f'prediction min:', end=' ')
        print(f'{Fore.YELLOW}{prediction.min():.2e}{Fore.RESET}', end=' ')
        print(f'max:', end=' ')
        print(f'{Fore.YELLOW}{prediction.max():.2e}{Fore.RESET}', end=' ')
        print(f'mean:', end=' ')
        print(f'{Fore.YELLOW}{prediction.mean():.2e}{Fore.RESET}', end=' ')
        print()
        
        all_weights = np.array([w for weight in nn.weights for w in weight.flat])
        print(f'[{iteration}] weights', end=' ')
        print(f'min:', end=' ')
        print(f'{Fore.YELLOW}{all_weights.min():.2e}{Fore.RESET}', end=' ')
        print(f'max:', end=' ')
        print(f'{Fore.YELLOW}{all_weights.max():.2e}{Fore.RESET}', end=' ')
        print(f'mean_abs:', end=' ')
        print(f'{Fore.YELLOW}{abs(all_weights).mean():.2e}{Fore.RESET}', end=' ')
        print()

        print(f'{Fore.CYAN}Truth{Fore.RESET}: same {Fore.YELLOW}different', end=' ')
        print(f'{Fore.RESET}| {Fore.CYAN}Prediction{Fore.RESET}: {Fore.GREEN}correct {Fore.RED}incorrect{Fore.RESET}')
        
        for p_row, a_row, equal in zip(
            nn.output,
            training_ans,
            nn.output == training_ans,
        ):
            msg = ''.join([
            Fore.RESET + str(a)
                if e
                else Fore.YELLOW + str(a)
                for p, a, e in zip(p_row, a_row, equal) 
            ]) + Fore.RESET + ' | '

            msg += ''.join([
                Fore.GREEN + str(p)
                if e
                else Fore.RED + str(p)
                for p, a, e in zip(p_row, a_row, equal)
            ]) + Fore.RESET
            if equal.all(): msg += ' Correct Answer!'

            print(msg)


if __name__ == '__main__':
    colorama.init()
    if not OP:
        user_set_operation()

    test()

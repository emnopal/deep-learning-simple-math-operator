import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import re

from model.valid_load import (
    load_model,
    score,
    print_accuracy,
    predicting_model
)

LOWER_EXIT_COMMAND = ["exit", "quit", "q", "stop", "e", "x", ":q"]
UPPER_EXIT_COMMAND = list(map(lambda x: x.upper(), LOWER_EXIT_COMMAND))
EXIT_COMMAND = LOWER_EXIT_COMMAND + UPPER_EXIT_COMMAND + list(map(lambda x: x.title(), LOWER_EXIT_COMMAND))

if __name__ == "__main__":

    while True:

        input_num = input(
            "Please input X //math operator ( + - * : )// Y to predict? (exit, quit or q to exit!): ").replace(" ", "")

        if input_num in EXIT_COMMAND:
            print("Bye!")
            break

        X, y = map(float, (re.findall(r'[^(+|-|*|:);]+', input_num)))

        operator = "".join(re.findall(r'[(+|-|*|:)]', input_num))

        if operator == '+':
            filepath = 'saved_model/addition.h5'
            model = load_model(filepath)
            prediction = predicting_model(model, X, y)[0][0]
            real = X+y
            print(f"Results of {X} + {y}")
            print(f"=> Predicted: {prediction}, Real: {real}")
            print(print_accuracy(real, prediction))

        elif operator == '-':
            filepath = 'saved_model/substraction.h5'
            model = load_model(filepath)
            prediction = predicting_model(model, X, y)[0][0]
            real = X-y
            print(f"Results of {X} - {y}")
            print(f"=> Predicted: {prediction}, Real: {real}")
            print(print_accuracy(real, prediction))

        elif operator == '*':
            filepath = 'saved_model/multiplication.h5'
            model = load_model(filepath)
            prediction = predicting_model(model, X, y)[0][0]
            real = X*y
            print(f"Results of {X} * {y}")
            print(f"=> Predicted: {prediction}, Real: {real}")
            print(print_accuracy(real, prediction))

        elif operator == ':':
            filepath = 'saved_model/divide.h5'
            model = load_model(filepath)
            prediction = predicting_model(model, X, y)[0][0]
            try:
                real = X/y
            except ZeroDivisionError:
                raise ZeroDivisionError(
                    "Sorry, 0 is not good for denumerator! please try another number")
            print(f"Results of {X} / {y}")
            print(f"=> Predicted: {prediction}, Real: {real}")
            print(print_accuracy(real, prediction))

        else:
            raise ValueError(
                "Sorry, I can't understand your math operator. Please try again!")

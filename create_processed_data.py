"""
This script is used to create a dataset suitable for training an embedding
model with sentence transformers using the triplet loss function. 

The values in the "negative" column are randomly selected from the "positive" column.

'For example, if your loss function accepts (anchor, positive, negative) triplets, then your first, second, and third dataset columns correspond with anchor, positive, and negative, respectively. This means that your first and second column must contain texts that should embed closely, and that your first and third column must contain texts that should embed far apart. That is why depending on your loss function, your dataset column order matters.'
https://huggingface.co/blog/train-sentence-transformers

"""
import random
from typing import List
from copy import copy 

import pandas as pd
from datasets import load_dataset, concatenate_datasets, Dataset

def select_random(m: int, options: List):
    """
    Select a random number between 0 and n (inclusive), ensuring the selected number is not equal to m.

    Parameters:
    ----------
    m : int
        The number to exclude from the selection.

    options : list
        List of numbers to randomly select from. m is removed from this list.

    Returns:
    -------
    int
        A random number between 0 and n, not equal to m.
    """
    # Create a list of numbers from 0 to n, excluding m
    numbers = copy(options)

    numbers.remove(m)

    # Select a random number from the filtered list
    return random.choice(numbers)

dataset_path = "DDSC/da-wikipedia-queries-gemma"

dataset = load_dataset(dataset_path)

all_options = [i for i in range(len(dataset["train"]))]

# for storing the index position of the selected value in "positive" that will act as the negative for the given query
selected_index_positions = []

random.seed(42)

for i in all_options:

    selected_number = select_random(m=i, options=all_options)

    selected_index_positions.append(selected_number)

df = {
    "anchor": dataset["train"]["query"],
    "positive": dataset["train"]["positive"]
}

df = pd.DataFrame(df)

# create a "negative" column where each cell value is a randomly selected row in the "positive" column
df["negative"] = df["positive"].iloc[selected_index_positions].reset_index(drop=True)

df["negative_index_pos"] = selected_index_positions

# Make sure we did not put any values from "positive" in the "negative" column
for index, row in df.iterrows():
    assert index != row["negative_index_pos"]

print(df.iloc[0,:])

dataset_final = Dataset.from_pandas(df)

dataset_final.push_to_hub("DDSC/da-wikipedia-queries-gemma-processed")

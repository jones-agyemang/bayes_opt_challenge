import numpy as np
from pathlib import Path

# Resolve paths relative to this file so the script works from any CWD
BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DATA_DIR = BASE_DIR / "processed_data"

NUMBER_OF_FUNCTIONS = 8

load_input = lambda x, type: np.load(f'./initial_data/function_{x}/initial_{type}.npy')

"""
return 
Example:
{
    1: [
        [0.982314, 0.017686],
        [0.650584, 0.578516]
    ],
    2: [
        [0.048921, 0.951079],
        [0.702428, 0.268982]
    ],
    3: [
        [0.873512, 0.126488, 0.521903],
        [1.00000e-06, 9.99999e-01, 9.99999e-01]
    ],
    ...
}
"""
def load_prior_data(io):
    """
    All prior results are expected to be in the processed data directory
    """
    data = load_processed_input_data(f'{PROCESSED_DATA_DIR}/{io}.txt')
    return to_results_dict(data)

def load_input_data(func_id):
    io = 'inputs'.lower()
    prior_data = load_prior_data(io)
    initial_data = load_input(func_id, io)
    data = np.vstack([initial_data, prior_data[func_id]])
    return data

# Output
def load_output_data(func_id):
    initial_data = load_input(func_id, 'outputs')
    path = f'{PROCESSED_DATA_DIR}/outputs.txt'
    prior_outputs = load_processed_output_data(path)
    prior_outputs = to_results_dict(prior_outputs)
    prior_data = prior_outputs[func_id]
    return np.concatenate([initial_data, prior_data])

def load_processed_input_data(path):
    fluff = ['[', 'array(', ' ', ']', ')']

    file = open(path).read()
    
    # Strictly returns the last array only
    split_file_data = file.split(']\n[')

    results = []
    for data in split_file_data:
        raw_contents = data.split('),')

        for idx, content in enumerate(raw_contents):
            for f in fluff:
                content = content.replace(f, '')
            results.append(list(map(np.float64, content.split(','))))
    return results

def load_processed_output_data(path):
    fluff = ['[', 'np.float64(', ')', ',', ']']

    file = open(path).read()
    raw_contents = file.split()

    results = []
    for content in raw_contents:
        for f in fluff:
            content = content.replace(f, '')
        results.append(np.float64(content))
    return np.asarray(results)

def to_results_dict(data):
    """
    Groups data by taking elements at indices:
    """
    return {
        i + 1: [data[j] for j in range(i, len(data), NUMBER_OF_FUNCTIONS)]
        for i in range(NUMBER_OF_FUNCTIONS)
    }
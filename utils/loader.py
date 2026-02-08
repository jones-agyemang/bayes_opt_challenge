import numpy as np

# ----
def load_evaluation_data(path):
    fluff = ['[', 'np.float64(', ')', ',', ']']

    file = open(path).read()
    raw_contents = file.split()

    results = []
    for content in raw_contents:
        for f in fluff:
            content = content.replace(f, '')
        results.append(np.float64(content))

    return results
# path = '/Users/jones.agyemang/Downloads/Capstone Project – Week 1 Submission Processed/outputs.txt'
# path = './processed_data/week_1/outputs.txt'
# results = load_evaluation_data(path)
# print(f'Evaluation report from Week 1: {results}')


# ----
def load_sample_data(path):
    fluff = ['[', 'array(', ' ', ']', ')']

    file = open(path).read()
    raw_contents = file.split('),')

    results = []
    for idx, content in enumerate(raw_contents):
        for f in fluff:
            content = content.replace(f, '')
        results.append(list(map(np.float64, content.split(','))))
    return results
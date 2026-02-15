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

# ----
def load_sample_data(path):
    fluff = ['[', 'array(', ' ', ']', ')']

    file = open(path).read()
    
    # Strictly returns the last array only
    split_file_data = file.split(']\n[')
    # breakpoint()

    results = []
    for data in split_file_data:
        raw_contents = data.split('),')

        for idx, content in enumerate(raw_contents):
            for f in fluff:
                content = content.replace(f, '')
            results.append(list(map(np.float64, content.split(','))))
    return results
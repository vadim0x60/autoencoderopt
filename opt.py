import os
import requests
from pathlib import Path
import json as j
import pickle
import itertools
import nevergrad as ng
from random import sample
from heapq import heappush, heappushpop
from uuid import uuid4

COMPILE_SERVER_API = os.environ.get('COMPILE_SERVER_API') or 'https://tree2tree.app/api'
LATENT_DIM = int(os.environ.get('LATENT_DIM') or '150')
MAX_TESTS = int(os.environ.get('MAX_TESTS') or '32')
TOP_K = int(os.environ.get('TOP_K') or '10')
TASK = os.environ['TASK']
MIN_FITNESS = float('-inf')
OPTIMIZER = ng.optimizers.registry[os.environ.get('OPTIMIZER') or 'NGOpt']
BUDGET = int(os.environ.get('BUDGET') or '10000')
RANGE = int(os.environ.get('RANGE') or '6')

datasets_path = Path(__file__).parent / 'datasets'
solutions_path = Path(__file__).parent / 'solutions'
os.makedirs(solutions_path, exist_ok=True)

class SourceFile():
    def __init__(self, code):
        self.filename = f'{uuid4()}.cpp'
        self.persist = False
        with open(solutions_path / self.filename, 'w') as f:
            f.write(code)

    def __lt__(self, other):
        return self.filename < other.filename

    def __del__(self):
        if not self.persist:
            os.remove(solutions_path / self.filename)

    def read(self):
        with open(solutions_path / self.filename, 'r') as f:
            return f.read()

class BackendError(Exception):
    pass

def backend(method, path, json=None):
    try:
        response = requests.request(method, COMPILE_SERVER_API + path, json=json)
        response.raise_for_status()
        return response.json()
    except (requests.exceptions.HTTPError, j.JSONDecodeError) as e:
        raise BackendError from e

class Executable():
    def __init__(self, source_file):
        self.source_file = source_file
        
    def __enter__(self):
        self.binary_name = backend('POST', '/programs', json={'program': self.source_file.read()})['binary']
        return self

    def __exit__(self, exc_type, exc_value, trace):
        backend('DELETE', f'/programs/{self.binary_name}')

    def run(self, input_lines):
        stdin = '\n'.join(input_lines)
        stdout = backend('POST', f'/programs/{self.binary_name}/run', json={'input': stdin})['stdout']
        return stdout.split('\n')

def correctness(expected_outputs, outputs):
    score = 0
    for expected_line, actual_line in zip(expected_outputs, outputs):
        score += (expected_line == actual_line)
    score /= len(expected_outputs)
    return score

def test_program(program, test_cases):
    with Executable(program) as e:
        test_results = [correctness(output_lines, e.run(input_lines)) for input_lines, output_lines in test_cases]
        return sum(test_results) / len(test_results)

def disnumerate_prefix(data, prefix):
    lines = []
    for line_number in itertools.count(start=1):
        try:
            lines.append(data[f'{prefix}{line_number}'])
        except KeyError:
            break
    return lines

def as_line(line):
    if isinstance(line, list):
        return ' '.join(map(str, line))
    else:
        return str(line)

def load_test_cases(path):
    with open(path, 'r') as f:
        for line in f.readlines():
            test_case = j.loads(line)
            input_lines = list(map(as_line, disnumerate_prefix(test_case, 'input')))
            output_lines = list(map(as_line, disnumerate_prefix(test_case, 'output')))
            yield input_lines, output_lines

def sample_tests(task_name):
    random_cases = list(load_test_cases(datasets_path / task_name / f'{task_name}-random.json'))
    edge_cases = list(load_test_cases(datasets_path / task_name / f'{task_name}-edge.json'))

    return (edge_cases + sample(random_cases, MAX_TESTS))[:MAX_TESTS]
    
def decode_vector(vector):
    decoding_request = {
        'latent_vector': vector.tolist(),
        'temperature': 0,
        'top_k': 0,
        'top_p': 0
    }

    return SourceFile(backend('POST', '/decode', json=decoding_request)['program'])

best_programs = []

for i in range(TOP_K + 1):
    heappush(best_programs, (MIN_FITNESS, SourceFile(str(i))))

def unfitness_function(task):
    def test_vector(vector):
        global best_fitness
        test_cases = sample_tests(task)
        try:
            program = decode_vector(vector)
            fitness = test_program(program, test_cases)
            heappushpop(best_programs, (fitness, program))
        except BackendError:
            fitness = MIN_FITNESS
        return - fitness

    return test_vector

experiment = f'{TASK}-{OPTIMIZER.__name__}'

optimizer = OPTIMIZER(parametrization=ng.p.Array(shape=(LATENT_DIM,), lower=-RANGE, upper=RANGE), budget=BUDGET)

try:
    recommendation = optimizer.minimize(unfitness_function(TASK), verbosity=2)
finally:
    with open(solutions_path / f'{experiment}.json', 'w') as f:
        j.dump({program.filename: fitness for fitness, program in best_programs}, f)

    for fitness, program in best_programs:
        if fitness > MIN_FITNESS:
            program.persist = True
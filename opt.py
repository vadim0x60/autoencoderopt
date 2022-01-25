import os
import requests
from pathlib import Path
import json as j
import itertools
import wandb
import nevergrad as ng
from random import sample

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
    def __init__(self, source):
        self.source = source
        
    def __enter__(self):
        self.binary_name = backend('POST', '/programs', json={'program': self.source})['binary']
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

def objects2lines(objects):
    lines = []
    for object in objects:
        if isinstance(object, list):
            # In competitive programming tasks arrays are represented as:
            # $ARRAY_LENGTH
            # $ARRAY_ELEMENT_1, $ARRAY_ELEMENT_2, ..., $ARRAY_ELEMENT_ARRAY_LENGTH
            lines.append(str(len(object)))
            lines.append(' '.join(map(str, object)))
        else:
            lines.append(str(object))
    return lines

def parse_test_case(test_case_str):
    test_case = j.loads(test_case_str)
    input_lines = []
    output_lines = []

    input_lines, output_lines = (objects2lines(disnumerate_prefix(test_case, prefix)) 
                                 for prefix in ('input', 'output'))
    return input_lines, output_lines

def sample_tests(task_name):
    random_cases, edge_cases = ((datasets_path / task_name / f'{task_name}-{test_set}.json').read_text().split('\n')
                                for test_set in ['random', 'edge'])

    selected_cases = (edge_cases + sample(random_cases, MAX_TESTS))[:MAX_TESTS]
    return map(parse_test_case, selected_cases)
    
def decode_vector(vector):
    decoding_request = {
        'latent_vector': vector.tolist(),
        'temperature': 0,
        'top_k': 0,
        'top_p': 0
    }

    return backend('POST', '/decode', json=decoding_request)['program']

def evaluate_vector(vector):
    try:
        program = decode_vector(vector)
        fitness = test_program(program, test_cases)
    except BackendError:
        fitness = MIN_FITNESS
    return program, fitness

best_fitness = MIN_FITNESS

def make_report(optimizer, candidate, program, fitness):
    global best_fitness
    best_fitness = max(fitness, best_fitness)

    report = {
        "#num-ask": optimizer.num_ask,
        "#num-tell": optimizer.num_tell,
        "#num-tell-not-asked": optimizer.num_tell_not_asked,
        "#lineage": candidate.heritage["lineage"],
        "#generation": candidate.generation,
        "#parents_uids": [],
        "#program": program,
        "#fitness": fitness,
        "#best_fitness": best_fitness
    }  

    if isinstance(candidate._meta.get("sigma"), float):
        report["#meta-sigma"] = candidate._meta["sigma"] 
    if candidate.generation > 1:
        report["#parents_uids"] = candidate.parents_uids
    return report

wandb.init(project='autoencoderopt')
wandb.config = {name: globals()[name] for name in ['LATENT_DIM', 'MAX_TESTS', 'TOP_K', 'TASK', 'OPTIMIZER', 'BUDGET', 'RANGE']}

experiment = f'{TASK}-{OPTIMIZER.__name__}'
optimizer = OPTIMIZER(parametrization=ng.p.Array(shape=(LATENT_DIM,), lower=-RANGE, upper=RANGE), budget=BUDGET)

try:
    for _ in range(optimizer.budget):
        candidate = optimizer.ask()
        test_cases = sample_tests(TASK)
        program, fitness = evaluate_vector(candidate.value)

        wandb.log(make_report(optimizer, candidate, program, fitness))

        optimizer.tell(candidate, - fitness)
finally:
    recommendation = optimizer.provide_recommendation()
    program, fitness = evaluate_vector(recommendation.value)
    wandb.log(make_report(optimizer, recommendation, program, fitness))
    program = '\n'.join([backend('GET', '/imports'), program])
    with open(solutions_path / f'{experiment}-{fitness}.json') as f:
        j.dump(program, f)
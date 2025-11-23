import functools
import multiprocessing
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from itertools import product
from calvin_agent.evaluation.multistep_sequences import get_sequences_for_state2
from calvin_agent.evaluation.utils import temp_seed
from calvin_agent.evaluation.multistep_sequences import flatten

@functools.lru_cache
def get_sequences(num_sequences=1000, num_workers=None, use_random_seed=False):
    possible_conditions = {
        "led": [0, 1],
        "lightbulb": [0, 1],
        "slider": ["right", "left"],
        "drawer": ["closed", "open"],
        "red_block": ["table", "slider_right", "slider_left"],
        "blue_block": ["table", "slider_right", "slider_left"],
        "pink_block": ["table", "slider_right", "slider_left"],
        "grasped": [0],
    }
    f = lambda l: l.count("table") in [1, 2] and l.count("slider_right") < 2 and l.count("slider_left") < 2
    value_combinations = filter(f, product(*possible_conditions.values()))
    initial_states = [dict(zip(possible_conditions.keys(), vals)) for vals in value_combinations]

    num_sequences_per_state = list(map(len, np.array_split(range(num_sequences), len(initial_states))))
    if use_random_seed:
        import time
        seed_value = int(time.time() * 1000000) % (2**31)
    else:
        seed_value = 0  
    
    with temp_seed(seed_value):
        num_workers = multiprocessing.cpu_count() if num_workers is None else num_workers
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            results = flatten(
                executor.map(
                    get_sequences_for_state2, zip(initial_states, num_sequences_per_state, range(len(initial_states)))
                )
            )
        results = list(zip(np.repeat(initial_states, num_sequences_per_state), results))
        np.random.shuffle(results)
    return results
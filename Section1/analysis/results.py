import json
from pathlib import Path

import numpy as np


ROOT_DIR = Path(__file__).resolve().parents[2]


def _json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, set):
        return sorted(value)
    raise TypeError(f'Cannot serialize {type(value).__name__} to JSON')


def write_json(filename, data):
    output_directory = ROOT_DIR / 'results'
    output_directory.mkdir(parents=True, exist_ok=True)

    output_path = output_directory / filename
    with output_path.open('w', encoding='utf-8') as output_file:
        json.dump(data, output_file, indent=2, default=_json_default)
        output_file.write('\n')

    print(f'Saved results: {output_path.relative_to(ROOT_DIR)}')
    return output_path
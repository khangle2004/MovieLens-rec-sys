from pathlib import Path

import matplotlib.pyplot as plt


ROOT_DIR = Path(__file__).resolve().parents[2]


def save_figure(filename, section='Section1'):
    output_directory = ROOT_DIR / 'figures' / section
    output_directory.mkdir(parents=True, exist_ok=True)

    output_path = output_directory / filename
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f'Saved figure: {output_path.relative_to(ROOT_DIR)}')
    return output_path
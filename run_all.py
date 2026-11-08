import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SCRIPTS = [
	ROOT / 'Section1' / 'section1.py',
]


def main():
	for script in SCRIPTS:
		print(f'\nRunning {script.relative_to(ROOT)}', flush=True)
		subprocess.run(
			[sys.executable, str(script)],
			cwd=ROOT,
			check=True,
		)


if __name__ == '__main__':
	main()

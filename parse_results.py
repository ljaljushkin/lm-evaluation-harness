import os
from pathlib import Path
import json
s = Path('/home/devuser/nlyalyus/projects/lm-evaluation-harness/runs')

paths = s.glob('**/results.json')
paths = sorted(paths, key=os.path.getmtime)
for path in paths:
    print(path)
    with path.open() as f:
        j = json.load(f)
        r = j['results']
        print(json.dumps(r, indent=4))


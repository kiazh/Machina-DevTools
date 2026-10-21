import json
import os
import re
import subprocess
from pathlib import Path
log = Path('/tmp/machina-sweep-backend.log').read_text()
nodeids = re.findall(r'^FAILED (\S+)', log, re.MULTILINE)
Path('/tmp/machina-sweep-failed-nodeids.json').write_text(json.dumps(nodeids, indent=2))
env = dict(os.environ, PYTHON='/Users/shadow/machina/.venv/bin/python', HYPOTHESIS_STORAGE_DIRECTORY='/tmp/machina-sweep-hypothesis')
with Path('/tmp/machina-sweep-backend-rerun.log').open('w') as out:
    result = subprocess.run(['bash', '/Users/shadow/machina-devtools/run-tests.sh', *nodeids, '-q', '-p', 'no:cacheprovider', '--tb=short'], env=env, stdout=out, stderr=subprocess.STDOUT)
raise SystemExit(result.returncode)

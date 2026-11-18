import json
import os
from pathlib import Path
import signal
import shutil
import subprocess
import sys
import tempfile
import time

def evaluate(code, task, timeout=5, backend=None, memory_mb=512, image='hcp-mbpp-evaluator:py313-numpy226'):
    setup = task.get('setup', [])
    if isinstance(setup, list):
        setup = '\n'.join(setup)
    payload = json.dumps({'code': code, 'tests': task['tests'], 'setup': setup})
    start = time.perf_counter()
    backend = backend or os.environ.get('HCP_EVALUATOR_BACKEND', 'auto')
    if backend == 'auto':
        backend = 'native' if sys.platform == 'linux' else 'docker'
    with tempfile.TemporaryDirectory(prefix='hcp-eval-', dir=os.environ.get('TMPDIR', '/tmp')) as directory:
        os.chmod(directory, 0o755)
        with open(Path(directory) / 'output.txt', 'w+') as log:
            environment = {'PATH': '/usr/bin:/bin', 'PYTHONDONTWRITEBYTECODE': '1', 'OPENBLAS_NUM_THREADS': '1', 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1', 'BLIS_NUM_THREADS': '1', 'NUMEXPR_NUM_THREADS': '1'}
            payload_text = payload
            if backend == 'docker':
                evaluator_path = Path(__file__).with_name('child.py').resolve()
                docker = shutil.which('docker')
                if not docker:
                    raise RuntimeError('Docker CLI is required for the local Linux-isolated evaluator')
                command = [docker, 'run', '--rm', '-i', '--network', 'none', '--read-only', '--memory', f'{memory_mb}m', '--cpus', '1', '--pids-limit', '32', '--cap-drop', 'ALL', '--security-opt', 'no-new-privileges', '--user', '65534:65534', '--tmpfs', '/tmp:rw,noexec,nosuid,size=16m', '-e', 'OPENBLAS_NUM_THREADS=1', '-e', 'OMP_NUM_THREADS=1', '-e', 'MKL_NUM_THREADS=1', '-e', 'BLIS_NUM_THREADS=1', '-e', 'NUMEXPR_NUM_THREADS=1', '-v', f'{evaluator_path}:/evaluator/child.py:ro', image, 'python', '-I', '-B', '/evaluator/child.py']
            elif backend == 'native':
                if sys.platform != 'linux':
                    raise RuntimeError('native seccomp evaluator requires Linux; set HCP_EVALUATOR_BACKEND=docker')
                command = ['/usr/bin/python3', '-I', '-B', str(Path(__file__).with_name('child.py'))]
            else:
                raise ValueError(f'unsupported evaluator backend: {backend}')
            process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=log, stderr=subprocess.STDOUT, cwd=directory, env=environment, start_new_session=True, text=True)
            timed_out = False
            try:
                process.communicate(payload_text, timeout=timeout)
            except subprocess.TimeoutExpired:
                timed_out = True
                os.killpg(process.pid, signal.SIGKILL)
                process.communicate()
            log.seek(0)
            feedback = log.read(65536)
    return {'passed': process.returncode == 0 and 'HCP_ALL_TESTS_PASS' in feedback, 'returncode': process.returncode, 'feedback': ('TIMEOUT\n' if timed_out else '') + feedback[-4000:], 'latency_seconds': time.perf_counter() - start, 'timed_out': timed_out, 'backend': backend}

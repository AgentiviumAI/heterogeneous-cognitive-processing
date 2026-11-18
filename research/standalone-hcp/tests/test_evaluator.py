from evaluators.process_runner import evaluate

def task(test):
    return {'tests': [test], 'setup': []}

def test_correct_and_incorrect():
    assert evaluate('def add(a,b): return a+b', task('assert add(2,3)==5'))['passed']
    assert not evaluate('def add(a,b): return a-b', task('assert add(2,3)==5'))['passed']

def test_network_and_write_denied():
    assert evaluate('import socket\ntry:\n socket.socket()\nexcept PermissionError:\n blocked=True', task('assert blocked'))['passed']
    assert evaluate('try:\n open("created.txt", "w")\nexcept PermissionError:\n blocked=True', task('assert blocked'))['passed']

def test_cpu_timeout():
    result = evaluate('while True: pass', task('assert True'))
    assert not result['passed']

def test_numpy_single_thread_import():
    assert evaluate('import numpy as np\nx=np.array([1,2])', task('assert x.sum()==3'))['passed']

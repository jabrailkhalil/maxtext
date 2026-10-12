"""Author-only exact-source CPU head/base verification; not part of the PR."""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET

BASE = 'a230f3e3def30651d55512307285a4298c8ff015'
HEAD = '2bda00aeb150572c1fcd81c918fc6f0778132fc8'
SOURCE = 'src/maxtext/checkpoint_conversion/utils/hf_shape.py'
TEST = 'tests/unit/hf_shape_test.py'
parser = argparse.ArgumentParser()
parser.add_argument('mode', choices=('head', 'base'))
args = parser.parse_args()

def blob(revision, path):
    return subprocess.check_output(['git', 'show', f'{revision}:{path}'])

base_source = blob(BASE, SOURCE)
head_source = blob(HEAD, SOURCE)
test_source = blob(HEAD, TEST)
assert Path(SOURCE).read_bytes().replace(b'\r\n', b'\n') == head_source
assert Path(TEST).read_bytes().replace(b'\r\n', b'\n') == test_source

def function_asts(raw):
    return {node.name: ast.dump(node, include_attributes=False)
            for node in ast.parse(raw).body if isinstance(node, ast.FunctionDef)}

base_functions = function_asts(base_source)
head_functions = function_asts(head_source)
assert base_functions.keys() < head_functions.keys()
assert set(head_functions) - set(base_functions) == {'OLMO3_HF_WEIGHTS_TO_SHAPE'}
assert all(head_functions[name] == body for name, body in base_functions.items())

def registry(raw):
    for node in ast.parse(raw).body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'HF_SHAPE'
                                                for t in node.targets):
            return {key.value: value.id for key, value in zip(node.value.keys, node.value.values)}
    raise AssertionError('Missing registry')

base_registry = registry(base_source)
head_registry = registry(head_source)
assert all(head_registry[key] == value for key, value in base_registry.items())
assert set(head_registry) - set(base_registry) == {'olmo3-7b', 'olmo3-7b-pt', 'olmo3-32b'}

# This replaces one source file only in this disposable author-CI checkout.
source = head_source if args.mode == 'head' else base_source
Path(SOURCE).write_bytes(source)
environment = dict(os.environ, PYTHONPATH=str(Path('src').resolve()),
                   JAX_PLATFORMS='cpu', DECOUPLE_GCLOUD='TRUE')
report_path = Path(f'olmo3-{args.mode}-junit.xml')
result = subprocess.run([sys.executable, '-m', 'pytest', TEST,
                         'tests/unit/newly_added_detection_test.py', '-q', '--tb=short',
                         f'--junitxml={report_path}'], env=environment, check=False)
cases = ET.parse(report_path).findall('.//testcase')
assert len(cases) == 34, len(cases)
failed = [case.attrib['name'] for case in cases if case.find('failure') is not None]
assert not any(case.find('error') is not None or case.find('skipped') is not None for case in cases)
if args.mode == 'head':
    assert result.returncode == 0 and not failed, failed
else:
    assert result.returncode == 1 and len(failed) == 16, (result.returncode, failed)
    assert all(name.startswith('test_olmo3_') for name in failed)
    for case in cases:
        failure = case.find('failure')
        if failure is not None:
            assert 'KeyError' in failure.attrib['message'] and 'olmo3-' in failure.attrib['message']

proof = {'mode': args.mode, 'source_revision': HEAD if args.mode == 'head' else BASE,
         'test_revision': HEAD, 'source_sha256': hashlib.sha256(source).hexdigest(),
         'test_sha256': hashlib.sha256(test_source).hexdigest(),
         'test_count': len(cases), 'failures': failed,
         'old_function_asts_unchanged': len(base_functions),
         'old_registry_entries_unchanged': len(base_registry),
         'python': sys.version, 'platform': sys.platform}
Path(f'olmo3-{args.mode}-proof.json').write_text(json.dumps(proof, indent=2) + '\n', encoding='utf-8')
print('OLMO3_SOURCE_PROOF=' + json.dumps(proof), flush=True)

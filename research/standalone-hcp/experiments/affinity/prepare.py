import argparse
import json
from pathlib import Path
import yaml
from .clients import Clients
from .common import digest, write_new, file_hash
from benchmarks.bfcl.adapter import prepare as bfcl_prepare
from benchmarks.scifact.candidate_pool import prepare as scifact_prepare

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/affinity-v1.yaml')
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    destination = Path(config['manifest_root'])
    if (destination / 'freeze.json').exists():
        freeze = json.loads((destination / 'freeze.json').read_text())
        assert freeze['config_hash'] == digest(config)
        assert all(file_hash(p) == sha for p, sha in freeze['manifest_sha256'].items())
        print('ALREADY_FROZEN', flush=True)
        return
    clients = Clients(config)
    clients.health()
    for dataset, prepare in [('bfcl', bfcl_prepare), ('scifact', scifact_prepare)]:
        path = destination / f'{dataset}.json'
        if not path.exists():
            value = prepare(Path(config['raw_root']) / dataset, config, clients.fits)
            write_new(path, value)
        print('PREPARED', dataset, flush=True)
    write_new(destination / 'freeze.json', {'config': config, 'config_hash': digest(config), 'manifest_sha256': {str(destination / f'{name}.json'): file_hash(destination / f'{name}.json') for name in ['bfcl', 'scifact']}, 'prompt_policy': 'Fixed before comparative inference; no outcome-dependent selection or tuning.'})
    print('FROZEN', flush=True)

if __name__ == '__main__':
    main()

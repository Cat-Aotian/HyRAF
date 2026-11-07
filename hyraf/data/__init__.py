import json, os
FOLD_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'folds')


def load_folds(name, fold):
    d = json.load(open(os.path.join(FOLD_DIR, f'{name}.json')))
    return d.get('labels'), d['folds'][str(fold)]

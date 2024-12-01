from env import *

import json

def eval_results(drv_clean=False) -> float:
    """
        Evaluate the results from the experiment, we use Area-Delay Product (ADP) as final metric
    """
    best_adp = float('inf')
    best_dir = None

    for data_dir in os.listdir(RESULT_DIR):
        result_json_path = os.path.join(RESULT_DIR, data_dir, 'result.json')
        if not os.path.exists(result_json_path):
            continue
        with open(result_json_path, 'r') as f:
            result_json = json.load(f)

        if drv_clean and result_json['drv'] != 0:
            continue

        adp = result_json['area'] * result_json['delay']
        best_adp = min(best_adp, adp)
        if adp == best_adp:
            best_dir = data_dir

    return best_adp, best_dir

if __name__ == '__main__':
    best_adp, best_dir = eval_results()
    print(f'Best ADP: {best_adp}')
    print(f'Best Directory: {best_dir}')
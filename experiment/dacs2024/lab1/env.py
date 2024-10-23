# setup environment variables

import os
import sys

CLDSE_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(CLDSE_ROOT)

LAB1_ROOT = os.path.join(CLDSE_ROOT, 'experiment/dacs2024/lab1')

RESULT_DIR = os.path.join(LAB1_ROOT, 'results')

ASAP7_ROOT = '/root/asap7'

GENUS_BIN = '/opt/cadence/GENUS20.12.001/bin/genus'

INNOVUS_BIN = '/opt/cadence/INNOVUS/bin/innovus'


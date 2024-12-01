# setup environment variables

import os
import sys

CLDSE_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(CLDSE_ROOT)

LAB2_ROOT = os.path.join(CLDSE_ROOT, 'experiment/dacs2024/lab2')

RESULT_DIR = os.path.join(LAB2_ROOT, 'results')

SKY130_ROOT = os.path.join(CLDSE_ROOT, 'experiment/dacs2024/lab2/pdk')

GENUS_BIN = '/opt/cadence/GENUS20.12.001/bin/genus'

INNOVUS_BIN = '/opt/cadence/INNOVUS/bin/innovus'


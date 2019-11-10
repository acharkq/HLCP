import os
from pathlib import Path


project_dir = Path(__file__).parent.parent
bin_dir = project_dir / 'bin'
critical_bin_dir =  bin_dir / 'critical'
civil_bin_dir = bin_dir / 'civil'

config_bin = bin_dir / 'config_bin'  # the bin for cause2id .etc projection


projection = os.path.join(config_bin, 'projection.pkl')
# the projection relation (cause2index, law2index, cause_tree); 1. critical; 2. civil;

critical_model_path = critical_bin_dir / 'models'
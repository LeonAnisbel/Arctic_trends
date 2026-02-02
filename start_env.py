import os

os.system('conda env_trends create -f trends.yml')
os.system('conda activate env_trends')

try:
    os.makedirs('plots')
except OSError:
    pass

try:
    os.makedirs('outputs')
except OSError:
    pass



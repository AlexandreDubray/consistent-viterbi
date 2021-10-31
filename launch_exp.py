import os
import sys
import subprocess

from multiprocessing import Pool

config_path = sys.argv[1]
nthreads = int(sys.argv[2])

config_files = [os.path.join(config_path, f) for f in os.listdir(config_path) if os.path.isfile(os.path.join(config_path, f))]

def launch_rust(config):
    subprocess.run(['./target/release/consistent-viterbi', '-c', config, '-e'])

with Pool(nthreads) as p:
    p.map(launch_rust, config_files)

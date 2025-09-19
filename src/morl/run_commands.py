import subprocess
import multiprocessing
import numpy as np
import time
import pathlib
import os

def execute_command(command):
    t = np.random.uniform() * 10
    time.sleep(t)
    print(f"Running command: {command}")
    env = dict(os.environ)
    process = subprocess.Popen(command, stdout=subprocess.PIPE, shell=True, stderr=subprocess.PIPE, env=env)
    output, error = process.communicate()
    returncode = process.poll()
    if returncode is not None:
        print(command, " is over")
        print(f"Output: {output.decode()}")
    else:
        print("**********wrong")
        print(f"Error: {error.decode()}")

    return output.decode()

# multiprocessing
if __name__ == '__main__': 
    pathlib.Path('logs/').mkdir(parents=True, exist_ok=True)    
    commands = ["nohup python ./src/morl/pretrain_eval_multi.py --start 950000 --end 970000 > ./logs/95-97.log &",
               "nohup python ./src/morl/pretrain_eval_multi.py --start 980000 --end 1000000 > ./logs/98-100.log &",]
    
    process_num = 2  
    pool = multiprocessing.Pool(processes=process_num)
    results = [
        pool.apply_async(execute_command, args=(command,))
        for command in commands
    ]
    pool.close()
    pool.join()
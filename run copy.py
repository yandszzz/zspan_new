import subprocess
import time


def run_script(script, name, **kwargs):
    cmd = ["python", script, "--name", str(name)]
    for key, value in kwargs.items():
        cmd.extend([f"--{key}", str(value)])
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"Error running {script}")
        return False
    return True


def run_main_scripts(satellite, name):
    # Define the scripts to be run sequentially
    scripts = ["main_SDE.py", "main_RSP.py", "main_FUG.py", "test.py"]

    # Parameters for main_SDE.py
    sde_params = {
        "lr": 0.0005,
        "epochs": 250,
        "batch_size": 1,
        "device": 'cuda',
        "satellite": satellite,
        "name": name
    }

    # Parameters for main_RSP.py
    rsp_params = {
        "lr": 0.001,
        "epochs": 150,
        "batch_size": 8,
        "device": 'cuda',
        "satellite": satellite,
        "name": name
    }

    # Parameters for main_FUG.py
    fug_params = {
        "lr": 0.0005,
        "epochs": 50,
        "batch_size": 1,
        "device": 'cuda',
        "satellite": satellite,
        "name": name
    }

    # Parameters for test.py
    test_params = {
        "satellite": satellite,
        "name": name
    }

    # Run scripts sequentially
    for script in scripts:
        if script == "main_SDE.py":
            if not run_script(script, **sde_params):
                return False
        elif script == "main_RSP.py":
            if not run_script(script, **rsp_params):
                return False
        elif script == "main_FUG.py":
            if not run_script(script, **fug_params):
                return False
        elif script == "test.py":
            if not run_script(script, **test_params):
                return False
        else:
            if not run_script(script, name):
                return False

    return True


if __name__ == "__main__":
    current_satellite = 'wv3/'
    current_name = 0
    print(f'training data is {current_satellite}{current_name}')
    t1 = time.time()
    if run_main_scripts(current_satellite, current_name):
        print(f"Completed run with NAME={current_name}")
    else:
        print("Script execution failed. Exiting...")
    t2 = time.time()
    print(f'total time: {t2-t1}s')

import subprocess

# Define the range of seeds
seeds = range(5)  # 10 different seeds (0 to 9)

# Define the script to run and the default arguments
# script_name = "main.py"  # Replace with the name of your script
script_name = "src/pure_model/ppo_train.py"  # Replace with the name of your script
env = "Humanoid-v5"
epochs = 50
total_timesteps = 10000000
exp_name = "ppo"
# exp_name = "dyna_ppo"

# Iterate over the seeds and run the script
for seed in seeds:
    command = [
        "python3", script_name,
        "--env", env,
        "--seed", str(seed),
        "--epochs", str(epochs),
        "--total_timesteps", str(total_timesteps),
        "--exp_name", exp_name
    ]
    print(f"Running: {' '.join(command)}")
    subprocess.run(command)

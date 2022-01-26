1) git clone https://github.com/raphaelspiekermann/TransformerForIMUbasedHAR.git
2) pip install -r requirements.txt
3) run main.py -> generates config.json & meta_config.json  
4) Setup config.json: 
    - Add a Path to a directory where the data and experiment results will be saved:\n
        > Example: /home/user/transformer_experiments (folder "transformer_experiments" will be generated if not already existing)
        > 3 Subfolders "data" for storing dataset, "runs" for single experiments and "experiments" for experiments containing multiple subexperiments will be created
    - model: check models/model_configs.json for different models
    - device_id: cpu / cuda / cuda:idx
    - split_type: person / person_random
5) run main.py 
    - Args:
        > -v for verbose mode (logs printed in terminal)
        > -m for running multiple experiments from meta_config.json
        > --experiment name_of_experiment: experiments will be stored under path_from_config/experiments/name_of_experiment
    - Examples for basic usage:
        > python3 main.py -v                        | single experiment specified in config.json
        > python3 main.py -m --experiment exp1234   | multiple experiment, settings from meta_config overwrite config for each run, usefull if for example you want to run the same experiment with different seeds          

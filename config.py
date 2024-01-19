config = {
    "system": {
        "train": {
            "epochs": 5000,
            "steps_per_epoch": 40,
            "learning_rate": 0.01,
            "noise_range": [-0.01, 0.01],
            "enable_jit": True,
            "enable_param_logging": False
        },
        "debug": False,
        "test_steps": 100,
    },
    "controller": {
        "name": "neural",
        "default": {
            "params_range": [0,1]
        },
        "neural": {
            "inputs": 3,
            "outputs": 1,
            "hidden_layers": [3],
            "activations": ["linear","linear"],
            "init_weight_range": [0,.1],
            "init_bias_range": [0,.1]
        }
    },
    "plant": {
        "name": "cournot",
        "bathtub": {
            "target": 1,
            "area": 100,
            "cross_section": 1,
            "gravity": 9.81
        },
        "cournot": {
            "target": 0.5,
            "max_price": 2.1,
            "margin_cost": 0.1
        },
        "robot": {
            "target": 0.25*3.14,
            "angle0": 0,
            "delta_time": 0.1,
            "mass": .5,
            "length": 2.0,
            "gravity": 9.81,
            "multiplier": 1,
            "coulomb_fric": 0.4,
            "viscous_fric": 0.6,
            "interval": [-3.14, 3.14],
        },
        "robot_multi": {
            "target": [0.25*3.14, 0.25*3.14],
            "angle0": [0, 0],
            "delta_time": 0.1,
            "mass": .5,
            "length": 2.0,
            "gravity": [0, 9.81],
            "multiplier": 1,
            "coulomb_fric": 0.4,
            "viscous_fric": 0.6,
            "interval": [-3.14, 3.14],
        },
        "optimizer": {
            "param1": 0,
            "param2": 0
        }
    }
}

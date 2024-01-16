h_params = {
    "system": {
        "epochs": 2000,
        "steps_per_epoch": 30,
        "learning_rate": 0.01,
        "noise_range": [-0.1, 0.1],
        "enable_jit": True,
        "enable_param_logging": False
    },
    "controller": {
        "name": "default",
        "default": {},
        "neural": {
            "inputs": 3,
            "outputs": 1,
            "hidden_layers": [3],
            "activations": ["relu","linear"],
            "init_weight_range": [-.1,.1],
            "init_bias_range": [-.1,.1]
        }
    },
    "plant": {
        "name": "robot",
        "bathtub": {
            "target": 5,
            "area": 100,
            "cross_section": 1,
            "gravity": 9.81
        },
        "cournot": {
            "max_price": 100,
            "margin_cost": 0.1
        },
        "robot": {
            "target": 0.1*3.14,
            "angle0": 0,
            "dt": 0.1,
            "mass": 10,
            "length": 2,
            "gravity": 1,
        },
        "optimizer": {
            "param1": 0,
            "param2": 0
        }
    }
}

h_params = {
    "system": {
        "epochs": 2000,
        "steps_per_epoch": 20,
        "learning_rate": 0.01,
        "noise_range": [-0.1, 0.1],
        "enable_jit": True,
        "enable_param_logging": True
    },
    "controller": {
        "name": "neural",
        "default": {},
        "neural": {
            "inputs": 3,
            "outputs": 1,
            "hidden_layers": [3],
            "activations": ["linear","linear"],
            "init_weight_range": [-1,1],
            "init_bias_range": [-1,1]
        }
    },
    "plant": {
        "name": "bathtub",
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
        "optimizer": {
            "param1": 0,
            "param2": 0
        }
    }
}

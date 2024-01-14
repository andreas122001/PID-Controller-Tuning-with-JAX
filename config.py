h_params = {
    "system": {
        "epochs": 10,
        "steps_per_epoch": 10,
        "learning_rate": 0.01,
        "noise_range": [-0.05, 0.05],
        "enable_jit": False
    },
    "controller": {
        "name": "default",
        "nn": {
            "hidden_layers": [2,2],
            "activations": ["Tanh"],
            "init_weight_range": [-10,10],
            "init_bias_range": [-10,10]
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

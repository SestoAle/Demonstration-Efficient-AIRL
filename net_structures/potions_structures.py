potions_net = [
    [
        {
            'type' : 'retrieve',
            'tensors' : ['global_in']
        },
        {
            "type": "conv2d",
            "size": 32,
            "window": (1,1),
            "stride": 1,
            "bias": False,
            "activation": 'tanh'
        },
        {
            "type": "conv2d",
            "size": 32,
            "window": (3, 3),
            "stride": 1,
            "activation": 'relu'
        },
        {
            "type": "conv2d",
            "size": 64,
            "window": (3, 3),
            "stride": 1,
            "activation": 'relu'
        },
        {
            'type': 'flatten'
        },
        {
            'type' : 'register',
            'tensor' : 'global_out'
        }
    ],
    [
        {
            'type' : 'retrieve',
            'tensors' : ['local_in']
        },
        {
            "type": "conv2d",
            "size": 32,
            "window": (1,1),
            "stride": 1,
            "bias": False,
            "activation": 'tanh'
        },
        {
            "type": "conv2d",
            "size": 32,
            "window": (3, 3),
            "stride": 1,
            "activation": 'relu'
        },
        {
            "type": "conv2d",
            "size": 64,
            "window": (3, 3),
            "stride": 1,
            "activation": 'relu'
        },
        {
            'type': 'flatten'
        },
        {
            'type' : 'register',
            'tensor' : 'local_out'
        }
    ],
    [
        {
            'type' : 'retrieve',
            'tensors' : ['local_in_two']
        },
        {
            "type": "conv2d",
            "size": 32,
            "window": (1,1),
            "stride": 1,
            "bias": False,
            "activation": 'tanh'
        },
        {
            "type": "conv2d",
            "size": 32,
            "window": (3, 3),
            "stride": 1,
            "activation": 'relu'
        },
        {
            "type": "conv2d",
            "size": 64,
            "window": (3, 3),
            "stride": 1,
            "activation": 'relu'
        },
        {
            'type': 'flatten'
        },
        {
            'type' : 'register',
            'tensor' : 'local_out_two'
        }
    ],
    [
        {
            'type' : 'retrieve',
            'tensors': ['global_out', 'local_out', 'local_out_two'],
            'aggregation': 'concat',
        },
        {
            "type": "dense",
            "size": 256,
            "activation": 'relu'
        },
        {
            'type' : 'register',
            'tensor' : 'first_FC'
        }
    ],
    [
        {
            'type' : 'retrieve',
            'tensors': ['prev_action']
        },
        {
            'type': 'flatten'
        },
        {
            'type': 'register',
            'tensor': 'action_out'
        }
    ],
    [
        {
            'type' : 'retrieve',
            'tensors': ['first_FC', 'action_out'],
            'aggregation': 'concat',
        },
        {
            "type": "dense",
            "size": 256,
        }
    ]
]
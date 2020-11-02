minigrid = [
    [
        {
            'type' : 'retrieve',
            'tensors' : ['global_in']
        },
        {
            "type": "embedding",
            "size": 32,
        },
        {
            "type": "conv2d",
            "size": 32,
            "window": (3, 3),
            "stride": 1,
            "activation": 'relu'
        },
        # {
        #     "type": "conv2d",
        #     "size": 64,
        #     "window": (3, 3),
        #     "stride": 1,
        #     "activation": 'relu'
        # },
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
            "type": "embedding",
            "size": 32,
        },
        {
            "type": "conv2d",
            "size": 32,
            "window": (3, 3),
            "stride": 1,
            "activation": 'relu'
        },
        # {
        #     "type": "conv2d",
        #     "size": 64,
        #     "window": (3, 3),
        #     "stride": 1,
        #     "activation": 'relu'
        # },
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
            'tensors': ['global_out', 'local_out'],
            'aggregation': 'concat',
        },
        {
            "type": "dense",
            "size": 256,
            "activation": 'relu'
        },
        # {
        #     "type": "dense",
        #     "size": 256,
        #     "activation": 'relu'
        # }
    ]
]

minigrid_global = [
    [
        {
            'type' : 'retrieve',
            'tensors' : ['global_in']
        },
        {
            "type": "conv2d",
            "size": 16,
            "window": (1, 1),
            "stride": 1,
            "activation": 'tanh'
        },
        {
            "type": "conv2d",
            "size": 32,
            "window": (2, 2),
            "stride": 1,
            "activation": 'relu'
        },
        {
            "type": "pool2d",
            "reduction": 'max',
            "window": (2, 2),
            "stride": 2,
        },
        {
            "type": "conv2d",
            "size": 32,
            "window": (2, 2),
            "stride": 1,
            "activation": 'relu'
        },
        {
            "type": "pool2d",
            "reduction": 'max',
            "window": (2, 2),
            "stride": 2,
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
            'tensors': ['global_out'],
            'aggregation': 'concat',
        },
        {
            "type": "dense",
            "size": 64,
            "activation": 'relu'
        }
    ]
]
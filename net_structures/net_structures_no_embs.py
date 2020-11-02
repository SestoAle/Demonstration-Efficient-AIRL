

dc2_net_conv_with_different_stats = [
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
            'tensors' : ['agent_stats']
        },
        {
            "type": "embedding",
            "size": 256,
        },
        {
            'type': 'flatten'
        },
        {
            "type": "dense",
            "size": 256,
            "activation": 'relu'
        },
        {
            'type' : 'register',
            'tensor' : 'agent_stats_out'
        }
    ],
    [
        {
            'type': 'retrieve',
            'tensors': ['target_stats']
        },
        {
            "type": "embedding",
            "size": 256
        },
        {
            'type': 'flatten'
        },
        {
            "type": "dense",
            "size": 256,
            "activation": 'relu'
        },
        {
            'type': 'register',
            'tensor': 'target_stats_out'
        }
    ],
    [
        {
            'type' : 'retrieve',
            'tensors': ['global_out', 'local_out', 'local_out_two', 'agent_stats_out', 'target_stats_out'],
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
dc2_baseline_conv_with_different_stats = [
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
            'tensor' : 'base_global_out'
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
            'tensor' : 'base_local_out'
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
            'tensor' : 'base_local_out_two'
        }
    ],
    [
        {
            'type': 'retrieve',
            'tensors': ['agent_stats']
        },
        {
            "type": "embedding",
            "size": 256
        },
        {
            'type': 'flatten'
        },
        {
            "type": "dense",
            "size": 128,
            "activation": 'relu'
        },
        {
            'type': 'register',
            'tensor': 'base_agent_stats_out'
        }
    ],
    [
        {
            'type': 'retrieve',
            'tensors': ['target_stats']
        },
        {
            "type": "embedding",
            "size": 256
        },
        {
            'type': 'flatten'
        },
        {
            "type": "dense",
            "size": 128,
            "activation": 'relu'
        },
        {
            'type': 'register',
            'tensor': 'base_target_stats_out'
        }
    ],
    [
        {
            'type' : 'retrieve',
            'tensors': ['base_global_out', 'base_local_out', 'base_local_out_two', 'base_agent_stats_out', 'base_target_stats_out'],
            'aggregation': 'concat',
        },
        {
            "type": "dense",
            "size": 256,
            "activation": 'relu'
        },
        {
            "type": "dense",
            "size": 256,
            "activation": 'relu'
        }
    ]
]

dc2_net_conv_with_different_stats_no_embs = [
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
            'tensors' : ['agent_stats']
        },
        {
            "type": "conv2d",
            "size": 256,
            "window": (1,1),
            "stride": 1,
            "activation": 'tanh'
        },
        {
            'type': 'flatten'
        },
        {
            "type": "dense",
            "size": 256,
            "activation": 'relu'
        },
        {
            'type' : 'register',
            'tensor' : 'agent_stats_out'
        }
    ],
    [
        {
            'type': 'retrieve',
            'tensors': ['target_stats']
        },
        {
            "type": "conv2d",
            "size": 256,
            "window": (1, 1),
            "stride": 1,
            "activation": 'tanh'
        },
        {
            'type': 'flatten'
        },
        {
            "type": "dense",
            "size": 256,
            "activation": 'relu'
        },
        {
            'type': 'register',
            'tensor': 'target_stats_out'
        }
    ],
    [
        {
            'type' : 'retrieve',
            'tensors': ['global_out', 'local_out', 'local_out_two', 'agent_stats_out', 'target_stats_out'],
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
dc2_baseline_conv_with_different_stats_no_embs = [
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
            'tensor' : 'base_global_out'
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
            'tensor' : 'base_local_out'
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
            'tensor' : 'base_local_out_two'
        }
    ],
    [
        {
            'type': 'retrieve',
            'tensors': ['agent_stats']
        },
        {
            "type": "conv2d",
            "size": 256,
            "window": (1, 1),
            "stride": 1,
            "activation": 'tanh'
        },
        {
            'type': 'flatten'
        },
        {
            "type": "dense",
            "size": 128,
            "activation": 'relu'
        },
        {
            'type': 'register',
            'tensor': 'base_agent_stats_out'
        }
    ],
    [
        {
            'type': 'retrieve',
            'tensors': ['target_stats']
        },
        {
            "type": "conv2d",
            "size": 256,
            "window": (1, 1),
            "stride": 1,
            "activation": 'tanh'
        },
        {
            'type': 'flatten'
        },
        {
            "type": "dense",
            "size": 128,
            "activation": 'relu'
        },
        {
            'type': 'register',
            'tensor': 'base_target_stats_out'
        }
    ],
    [
        {
            'type' : 'retrieve',
            'tensors': ['base_global_out', 'base_local_out', 'base_local_out_two', 'base_agent_stats_out', 'base_target_stats_out'],
            'aggregation': 'concat',
        },
        {
            "type": "dense",
            "size": 256,
            "activation": 'relu'
        },
        {
            "type": "dense",
            "size": 256,
            "activation": 'relu'
        }
    ]
]


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

from new_layers.TransformerMap import Transformer, Mask, OutputPositionItem, ScatterEmbedding

# Neural network structure
net_transformer = [
    [
        {
            'type' : 'retrieve',
            'tensors' : ['global_in']
        },
        {
            "type": "embedding",
            "size": 32
        },
        {
            "type": "conv2d",
            "size": 32,
            "window": (3, 3),
            "stride": 1,
            "activation": 'relu'
        },
        {
            'type': 'pool2d',
            'reduction': 'max'
        },
        {
            "type": "conv2d",
            "size": 64,
            "window": (3, 3),
            "stride": 1,
            "activation": 'relu'
        },
        {
            'type': 'pool2d',
            'reduction': 'max'
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
            "type": "embedding",
            "size": 32
        },
        {
            "type": "conv2d",
            "size": 32,
            "window": (3, 3),
            "stride": 1,
            "activation": 'relu'
        },
        {
            'type': 'pool2d',
            'reduction': 'max'
        },
        {
            "type": "conv2d",
            "size": 64,
            "window": (3, 3),
            "stride": 1,
            "activation": 'relu'
        },
        {
            'type': 'pool2d',
            'reduction': 'max'
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
            "type": "embedding",
            "size": 32
        },
        {
            "type": "conv2d",
            "size": 32,
            "window": (3, 3),
            "stride": 1,
            "activation": 'relu'
        },
        {
            'type': 'pool2d',
            'reduction': 'max'
        },
        {
            "type": "conv2d",
            "size": 64,
            "window": (3, 3),
            "stride": 1,
            "activation": 'relu'
        },
        {
            'type': 'pool2d',
            'reduction': 'max'
        },
        {
            'type': 'flatten'
        },
        {
            'type' : 'register',
            'tensor' : 'local_out_two'
        }
    ],

    # Transformer Block
    # First embeddings
    [
        {
            'type': 'retrieve',
            'tensors': ['items']
        },
        {
            'type': 'conv1d',
            'window': 1,
            'stride': 1,
            'size': 256,
            'activation': 'relu'
        },
        {
            'type': 'register',
            'tensor': 'items_emb'
        }
    ],
    [
        {
            'type': 'retrieve',
            'tensors': ['agent']
        },
        {
            'type': 'conv1d',
            'window': 1,
            'stride': 1,
            'size': 256,
            'activation': 'relu'
        },
        {
            'type': 'register',
            'tensor': 'agent_emb'
        }
    ],
    [
        {
            'type': 'retrieve',
            'tensors': ['target']
        },
        {
            'type': 'conv1d',
            'window': 1,
            'stride': 1,
            'size': 256,
            'activation': 'relu'
        },
        {
            'type': 'register',
            'tensor': 'target_emb'
        }
    ],
    # Creating Mask
    [
        {
            'type' : Mask,
            'tensors' : ['items', 'agent', 'target'],
            'value': 99.0
        },
        {
            'type': 'register',
            'tensor': 'mask'
        }

    ],

    # Self attention
    [
        {
            'type' : 'retrieve',
            'tensors' : ['items_emb', 'agent_emb', 'target_emb'],
            'aggregation': 'concat',
        },
        {
            "type": Transformer,
            "n_head": 2,
            "hidden_size": 256,
            "pooling": 'max',
            'num_entities': 22,
            'mask_name': 'mask'
        },
        {
            'type': 'flatten'
        },
        {
            'type' : 'register',
            'tensor' : 'trans_out'
        }
    ],

    [
        {
            'type' : 'retrieve',
            'tensors': ['global_out', 'local_out', 'local_out_two', 'trans_out'],
            'aggregation': 'concat',
        },
        {
            "type": "dense",
            "size": 256,
            "activation": 'relu'
        },
        {
            'type': 'register',
            'tensor': 'first_FC'
        }
    ],
    [
        {
            'type': 'retrieve',
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
            'type': 'retrieve',
            'tensors': ['first_FC', 'action_out'],
            'aggregation': 'concat',
        },
        {
            "type": "dense",
            "size": 256,
            "activation": 'relu'
        }
    ]
]


net_transformer_map = [
    # Transformer Block
    # First embeddings
    [
        {
            'type': 'retrieve',
            'tensors': ['items']
        },
        {
            'type': 'conv1d',
            'window': 1,
            'stride': 1,
            'size': 64,
            'activation': 'relu'
        },
        {
            'type': 'register',
            'tensor': 'items_emb'
        }
    ],
    # Creating Mask
    [
        {
            'type': Mask,
            'tensors': ['items'],
            'value': 99.0,
            'num_entities': 100
        },
        {
            'type': 'register',
            'tensor': 'mask'
        }

    ],
    # Self attention
    [
        {
            'type': 'retrieve',
            'tensors': ['items_emb'],
            'aggregation': 'concat',
        },
        {
            "type": Transformer,
            "n_head": 2,
            "hidden_size": 64,
            "pooling": 'none',
            'num_entities': 100,
            'mask_name': 'mask'
        },
        {
            'type': 'register',
            'tensor': 'trans_global'
        }
    ],
    [
        {
            'type' : 'retrieve',
            'tensors' : ['global_in']
        },
        {
            "type": "embedding",
            "size": 32
        },
        {
            'type': 'register',
            'tensor': 'global_emb'
        }
    ],
    [
        {
            'type': 'retrieve',
            'tensors': ['global_emb', 'trans_global'],
            'aggregation': 'concat',
            'axis': 2
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
    # Transformer Block
    # Second embeddings
    [
        {
            'type': 'retrieve',
            'tensors': ['items_local']
        },
        {
            'type': 'conv1d',
            'window': 1,
            'stride': 1,
            'size': 64,
            'activation': 'relu'
        },
        {
            'type': 'register',
            'tensor': 'items_emb_local'
        }
    ],
    # Creating Mask
    [
        {
            'type': Mask,
            'tensors': ['items_local'],
            'value': 99.0,
            'num_entities': 25
        },
        {
            'type': 'register',
            'tensor': 'mask_local'
        }

    ],
    # Self attention
    [
        {
            'type': 'retrieve',
            'tensors': ['items_emb_local'],
            'aggregation': 'concat',
        },
        {
            "type": Transformer,
            "n_head": 2,
            "hidden_size": 64,
            "pooling": 'none',
            'num_entities': 25,
            'mask_name': 'mask_local',
        },
        {
            'type': 'register',
            'tensor': 'trans_local'
        }
    ],
    [
        {
            'type' : 'retrieve',
            'tensors' : ['local_in']
        },
        {
            "type": "embedding",
            "size": 32
        },
        {
            'type': 'register',
            'tensor': 'local_emb'
        }
    ],
    [
        {
            'type': 'retrieve',
            'tensors': ['local_emb', 'trans_local'],
            'aggregation': 'concat',
            'axis': 2
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
    # Transformer Block
    # third embeddings
    [
        {
            'type': 'retrieve',
            'tensors': ['items_local_two']
        },
        {
            'type': 'conv1d',
            'window': 1,
            'stride': 1,
            'size': 64,
            'activation': 'relu'
        },
        {
            'type': 'register',
            'tensor': 'items_emb_local_two'
        }
    ],
    # Creating Mask
    [
        {
            'type': Mask,
            'tensors': ['items_local_two'],
            'value': 99.0,
            'num_entities': 9
        },
        {
            'type': 'register',
            'tensor': 'mask_local_two'
        }

    ],
    # Self attention
    [
        {
            'type': 'retrieve',
            'tensors': ['items_emb_local_two'],
            'aggregation': 'concat',
        },
        {
            "type": Transformer,
            "n_head": 2,
            "hidden_size": 64,
            "pooling": 'none',
            'num_entities': 9,
            'mask_name': 'mask_local_two',
        },
        {
            'type': 'register',
            'tensor': 'trans_local_two'
        }
    ],
    [
        {
            'type' : 'retrieve',
            'tensors' : ['local_in_two']
        },
        {
            "type": "embedding",
            "size": 32
        },
        {
            'type': 'register',
            'tensor': 'local_two_emb'
        }
    ],
    [
        {
            'type': 'retrieve',
            'tensors': ['local_two_emb', 'trans_local_two'],
            'aggregation': 'concat',
            'axis': 2
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
            'type': 'register',
            'tensor': 'first_FC'
        }
    ],
    [
        {
            'type': 'retrieve',
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
            'type': 'retrieve',
            'tensors': ['first_FC', 'action_out'],
            'aggregation': 'concat',
        },
        {
            "type": "dense",
            "size": 256,
            "activation": 'relu'
        }
    ]
]
# Baseline net structure
baseline_transformer = [
    [
        {
            'type' : 'retrieve',
            'tensors' : ['global_in']
        },
        {
            "type": "embedding",
            "size": 32
        },
        {
            "type": "conv2d",
            "size": 32,
            "window": (3, 3),
            "stride": 1,
            "activation": 'relu'
        },
        {
            'type': 'pool2d',
            'reduction': 'max'
        },
        {
            "type": "conv2d",
            "size": 64,
            "window": (3, 3),
            "stride": 1,
            "activation": 'relu'
        },
        {
            'type': 'pool2d',
            'reduction': 'max'
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
            "type": "embedding",
            "size": 32
        },
        {
            "type": "conv2d",
            "size": 32,
            "window": (3, 3),
            "stride": 1,
            "activation": 'relu'
        },
        {
            'type': 'pool2d',
            'reduction': 'max'
        },
        {
            "type": "conv2d",
            "size": 64,
            "window": (3, 3),
            "stride": 1,
            "activation": 'relu'
        },
        {
            'type': 'pool2d',
            'reduction': 'max'
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
            "type": "embedding",
            "size": 32
        },
        {
            "type": "conv2d",
            "size": 32,
            "window": (3, 3),
            "stride": 1,
            "activation": 'relu'
        },
        {
            'type': 'pool2d',
            'reduction': 'max'
        },
        {
            "type": "conv2d",
            "size": 64,
            "window": (3, 3),
            "stride": 1,
            "activation": 'relu'
        },
        {
            'type': 'pool2d',
            'reduction': 'max'
        },
        {
            'type': 'flatten'
        },
        {
            'type' : 'register',
            'tensor' : 'base_local_out_two'
        }
    ],

    # Transformer Block
    # First embeddings
    [
        {
            'type': 'retrieve',
            'tensors': ['items']
        },
        {
            'type': 'conv1d',
            'window': 1,
            'stride': 1,
            'size': 256,
            'activation': 'relu'
        },
        {
            'type': 'register',
            'tensor': 'base_items_emb'
        }
    ],
    [
        {
            'type': 'retrieve',
            'tensors': ['agent']
        },
        {
            'type': 'conv1d',
            'window': 1,
            'stride': 1,
            'size': 256,
            'activation': 'relu'
        },
        {
            'type': 'register',
            'tensor': 'base_agent_emb'
        }
    ],
    [
        {
            'type': 'retrieve',
            'tensors': ['target']
        },
        {
            'type': 'conv1d',
            'window': 1,
            'stride': 1,
            'size': 256,
            'activation': 'relu'
        },
        {
            'type': 'register',
            'tensor': 'base_target_emb'
        }
    ],
    # Creating Mask
    [
        {
            'type' : Mask,
            'tensors' : ['items', 'agent', 'target'],
            'value': 99.0
        },
        {
            'type': 'register',
            'tensor': 'base_mask'
        }

    ],

    # Self attention
    [
        {
            'type' : 'retrieve',
            'tensors' : ['base_items_emb', 'base_agent_emb', 'base_target_emb'],
            'aggregation': 'concat',
        },
        {
            "type": Transformer,
            "n_head": 2,
            "hidden_size": 256,
            "pooling": 'max',
            'num_entities': 22,
            'mask_name': 'base_mask'
        },
        {
            'type': 'flatten'
        },
        {
            'type' : 'register',
            'tensor' : 'base_trans_out'
        }
    ],

    [
        {
            'type' : 'retrieve',
            'tensors': ['base_global_out', 'base_local_out', 'base_local_out_two', 'base_trans_out'],
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

baseline_transformer_map = [
    # Transformer Block
    # First embeddings
    [
        {
            'type': 'retrieve',
            'tensors': ['items']
        },
        {
            'type': 'conv1d',
            'window': 1,
            'stride': 1,
            'size': 64,
            'activation': 'relu'
        },
        {
            'type': 'register',
            'tensor': 'base_items_emb'
        }
    ],
    # Creating Mask
    [
        {
            'type': Mask,
            'tensors': ['items'],
            'value': 99.0,
            'num_entities': 100
        },
        {
            'type': 'register',
            'tensor': 'base_mask'
        }

    ],
    # Self attention
    [
        {
            'type': 'retrieve',
            'tensors': ['base_items_emb'],
            'aggregation': 'concat',
        },
        {
            "type": Transformer,
            "n_head": 2,
            "hidden_size": 64,
            "pooling": 'none',
            'num_entities': 100,
            'mask_name': 'base_mask',
        },
        {
            'type': 'register',
            'tensor': 'base_trans_global'
        }
    ],
    [
        {
            'type' : 'retrieve',
            'tensors' : ['global_in']
        },
        {
            "type": "embedding",
            "size": 32
        },
        {
            'type': 'register',
            'tensor': 'base_global_emb'
        }
    ],
    [
        {
            'type': 'retrieve',
            'tensors': ['base_global_emb', 'base_trans_global'],
            'aggregation': 'concat',
            'axis': 2
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
    # Transformer Block
    # Second embeddings
    [
        {
            'type': 'retrieve',
            'tensors': ['items_local']
        },
        {
            'type': 'conv1d',
            'window': 1,
            'stride': 1,
            'size': 64,
            'activation': 'relu'
        },
        {
            'type': 'register',
            'tensor': 'base_items_emb_local'
        }
    ],
    # Creating Mask
    [
        {
            'type': Mask,
            'tensors': ['items_local'],
            'value': 99.0,
            'num_entities': 25
        },
        {
            'type': 'register',
            'tensor': 'base_mask_local'
        }

    ],
    # Self attention
    [
        {
            'type': 'retrieve',
            'tensors': ['base_items_emb_local'],
            'aggregation': 'concat',
        },
        {
            "type": Transformer,
            "n_head": 2,
            "hidden_size": 64,
            "pooling": 'none',
            'num_entities': 25,
            'mask_name': 'base_mask_local',
        },
        {
            'type': 'register',
            'tensor': 'base_trans_local'
        }
    ],
    [
        {
            'type' : 'retrieve',
            'tensors' : ['local_in']
        },
        {
            "type": "embedding",
            "size": 32
        },
        {
            'type': 'register',
            'tensor': 'base_local_emb'
        }
    ],
    [
        {
            'type': 'retrieve',
            'tensors': ['base_local_emb', 'base_trans_local'],
            'aggregation': 'concat',
            'axis': 2
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
    # Transformer Block
    # third embeddings
    [
        {
            'type': 'retrieve',
            'tensors': ['items_local_two']
        },
        {
            'type': 'conv1d',
            'window': 1,
            'stride': 1,
            'size': 64,
            'activation': 'relu'
        },
        {
            'type': 'register',
            'tensor': 'base_items_emb_local_two'
        }
    ],
    # Creating Mask
    [
        {
            'type': Mask,
            'tensors': ['items_local_two'],
            'value': 99.0,
            'num_entities': 9
        },
        {
            'type': 'register',
            'tensor': 'base_mask_local_two'
        }

    ],
    # Self attention
    [
        {
            'type': 'retrieve',
            'tensors': ['base_items_emb_local_two'],
            'aggregation': 'concat',
        },
        {
            "type": Transformer,
            "n_head": 2,
            "hidden_size": 64,
            "pooling": 'none',
            'num_entities': 9,
            'mask_name': 'base_mask_local_two',
        },
        {
            'type': 'register',
            'tensor': 'base_trans_local_two'
        }
    ],
    [
        {
            'type' : 'retrieve',
            'tensors' : ['local_in_two']
        },
        {
            "type": "embedding",
            "size": 32
        },
        {
            'type': 'register',
            'tensor': 'base_local_two_emb'
        }
    ],
    [
        {
            'type': 'retrieve',
            'tensors': ['base_local_two_emb', 'base_trans_local_two'],
            'aggregation': 'concat',
            'axis': 2
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
            'tensor' : 'base_agent_stats_out'
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

from new_layers.Transformer import Transformer

# Neural network structures
dc2_net_embs = [
    [
        [
            {
                'type' : 'input',
                'names' : ['global_in']
            },
            {
                "type": "embedding",
                "indices": 7,
                "size": 16
            },
            {
                'type' : 'output',
                'name' : 'global_embs'
            }
        ],
        [
            {
                'type' : 'input',
                'names' : ['attr_global_1']
            },
            {
                "type": "embedding",
                "indices": 9,
                "size": 32
            },
            {
                'type' : 'output',
                'name' : 'attr_global_1_embs'
            }
        ],
        [
            {
                'type' : 'input',
                'names' : ['attr_global_2']
            },
            {
                "type": "embedding",
                "indices": 9,
                "size": 32
            },
            {
                'type' : 'output',
                'name' : 'attr_global_2_embs'
            }
        ],
        [
            {
                'type' : 'input',
                'names' : ['attr_global_3']
            },
            {
                "type": "embedding",
                "indices": 9,
                "size": 32
            },
            {
                'type' : 'output',
                'name' : 'attr_global_3_embs'
            }
        ],
        [
            {
                'type': 'input',
                'names': ['attr_global_4']
            },
            {
                "type": "embedding",
                "indices": 9,
                "size": 32
            },
            {
                'type': 'output',
                'name': 'attr_global_4_embs'
            }
        ],
        [
            {
                'type': 'input',
                'names': ['attr_global_5']
            },
            {
                "type": "embedding",
                "indices": 9,
                "size": 32
            },
            {
                'type': 'output',
                'name': 'attr_global_5_embs'
            }
        ],
        [
            {
                'type': 'input',
                'names': ['attr_global_1_embs', 'attr_global_2_embs', 'attr_global_3_embs', 'attr_global_4_embs', 'attr_global_5_embs'],
                'aggregation_type': 'concat',
                'axis': 3
            },
            {
                'type': 'conv2d',
                'activation': 'relu',
                'window': (1,1),
                'stride': 1,
                'size': 16
            },
            {
                'type': 'output',
                'name': 'weapon_emb'
            }
        ],
        {
            'type': 'input',
            'names': ['global_embs', 'weapon_emb'],
            'aggregation_type': 'concat',
            'axis': 3
        },
        {
            "type": "conv2d",
            "size": 32,
            "window": (3,3),
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
            'type' : 'output',
            'name' : 'global_out'
        }
    ],
    [
        [
            {
                'type' : 'input',
                'names' : ['local_in']
            },
            {
                "type": "embedding",
                "indices": 7,
                "size": 16
            },
            {
                'type' : 'output',
                'name' : 'local_embs'
            }
        ],
        [
            {
                'type' : 'input',
                'names' : ['attr_local_1']
            },
            {
                "type": "embedding",
                "indices": 9,
                "size": 32
            },
            {
                'type' : 'output',
                'name' : 'attr_local_1_embs'
            }
        ],
        [
            {
                'type' : 'input',
                'names' : ['attr_local_2']
            },
            {
                "type": "embedding",
                "indices": 9,
                "size": 32
            },
            {
                'type' : 'output',
                'name' : 'attr_local_2_embs'
            }
        ],
        [
            {
                'type' : 'input',
                'names' : ['attr_local_3']
            },
            {
                "type": "embedding",
                "indices": 9,
                "size": 32
            },
            {
                'type' : 'output',
                'name' : 'attr_local_3_embs'
            }
        ],
        [
            {
                'type': 'input',
                'names': ['attr_local_4']
            },
            {
                "type": "embedding",
                "indices": 9,
                "size": 32
            },
            {
                'type': 'output',
                'name': 'attr_local_4_embs'
            }
        ],
        [
            {
                'type': 'input',
                'names': ['attr_local_5']
            },
            {
                "type": "embedding",
                "indices": 9,
                "size": 32
            },
            {
                'type': 'output',
                'name': 'attr_local_5_embs'
            }
        ],
        [
            {
                'type': 'input',
                'names': ['attr_local_1_embs', 'attr_local_2_embs', 'attr_local_3_embs', 'attr_local_4_embs', 'attr_local_5_embs'],
                'aggregation_type': 'concat',
                'axis': 3
            },
            {
                'type': 'conv2d',
                'activation': 'relu',
                'window': (1,1),
                'stride': 1,
                'size': 16
            },
            {
                'type': 'output',
                'name': 'weapon_local_emb'
            }
        ],
        {
            'type': 'input',
            'names': ['local_embs', 'weapon_local_emb'],
            'aggregation_type': 'concat',
            'axis': 3
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
            'type' : 'output',
            'name' : 'local_out'
        }
    ],
    [
        [
            {
                'type': 'input',
                'names': ['local_in_two']
            },
            {
                "type": "embedding",
                "indices": 7,
                "size": 16
            },
            {
                'type': 'output',
                'name': 'local_two_embs'
            }
        ],
        [
            {
                'type': 'input',
                'names': ['attr_local_two_1']
            },
            {
                "type": "embedding",
                "indices": 9,
                "size": 32
            },
            {
                'type': 'output',
                'name': 'attr_local_two_1_embs'
            }
        ],
        [
            {
                'type': 'input',
                'names': ['attr_local_two_2']
            },
            {
                "type": "embedding",
                "indices": 9,
                "size": 32
            },
            {
                'type': 'output',
                'name': 'attr_local_two_2_embs'
            }
        ],
        [
            {
                'type': 'input',
                'names': ['attr_local_two_3']
            },
            {
                "type": "embedding",
                "indices": 9,
                "size": 32
            },
            {
                'type': 'output',
                'name': 'attr_local_two_3_embs'
            }
        ],
        [
            {
                'type': 'input',
                'names': ['attr_local_two_4']
            },
            {
                "type": "embedding",
                "indices": 9,
                "size": 32
            },
            {
                'type': 'output',
                'name': 'attr_local_two_4_embs'
            }
        ],
        [
            {
                'type': 'input',
                'names': ['attr_local_two_5']
            },
            {
                "type": "embedding",
                "indices": 9,
                "size": 32
            },
            {
                'type': 'output',
                'name': 'attr_local_two_5_embs'
            }
        ],
        [
            {
                'type': 'input',
                'names': ['attr_local_two_1_embs', 'attr_local_two_2_embs', 'attr_local_two_3_embs', 'attr_local_two_4_embs', 'attr_local_two_5_embs'],
                'aggregation_type': 'concat',
                'axis': 3
            },
            {
                'type': 'conv2d',
                'activation': 'relu',
                'window': (1,1),
                'stride': 1,
                'size': 16
            },
            {
                'type': 'output',
                'name': 'weapon_local_two_emb'
            }
        ],
        {
            'type': 'input',
            'names': ['local_two_embs', 'weapon_local_two_emb'],
            'aggregation_type': 'concat',
            'axis': 3
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
            'type' : 'output',
            'name' : 'local_out_two'
        }
    ],
    [
        {
            'type' : 'input',
            'names' : ['agent_stats']
        },
        {
            "type": "embedding",
            "indices": 107,
            "size": 256
        },
        {
            "type": "nonlinearity",
            "name": 'tanh'
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
            'type' : 'output',
            'name' : 'agent_stats_out'
        }
    ],
    [
        {
            'type': 'input',
            'names': ['target_stats']
        },
        {
            "type": "embedding",
            "indices": 32,
            "size": 256
        },
        {
            "type": "nonlinearity",
            "name": 'tanh'
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
            'type': 'output',
            'name': 'target_stats_out'
        }
    ],
    [
        {
            'type' : 'input',
            'names': ['global_out', 'local_out', 'local_out_two', 'agent_stats_out', 'target_stats_out'],
            'aggregation_type': 'concat',
        },
        {
            "type": "dense",
            "size": 256,
            "activation": 'relu'
        },
        {
            'type' : 'output',
            'name' : 'first_FC'
        }
    ],
    [
        {
            'type' : 'input',
            'names': ['action']
        },
        {
            'type': 'flatten'
        },
        {
            'type': 'output',
            'name': 'action_out'
        }
    ],
    [
        {
            'type' : 'input',
            'names': ['first_FC', 'action_out'],
            'aggregation_type': 'concat',
        },
        {
            "type": "internal_lstm",
            "size": 256,
        }
    ]
]
dc2_net_conv = [
    [
        {
            'type' : 'input',
            'names' : ['global_in']
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
            'type' : 'output',
            'name' : 'global_out'
        }
    ],
    [
        {
            'type' : 'input',
            'names' : ['local_in']
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
            'type' : 'output',
            'name' : 'local_out'
        }
    ],
    [
        {
            'type' : 'input',
            'names' : ['local_in_two']
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
            'type' : 'output',
            'name' : 'local_out_two'
        }
    ],
    [
        {
            'type' : 'input',
            'names' : ['stats']
        },
        {
            "type": "embedding",
            "indices": 136,
            "size": 128
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
            'type' : 'output',
            'name' : 'stats_out'
        }
    ],
    [
        {
            'type' : 'input',
            'names': ['global_out', 'local_out', 'local_out_two', 'stats_out'],
            'aggregation_type': 'concat',
        },
        {
            "type": "dense",
            "size": 256,
            "activation": 'relu'
        },
        {
            'type' : 'output',
            'name' : 'first_FC'
        }
    ],
    [
        {
            'type' : 'input',
            'names': ['action']
        },
        {
            'type': 'flatten'
        },
        {
            'type': 'output',
            'name': 'action_out'
        }
    ],
    [
        {
            'type' : 'input',
            'names': ['first_FC', 'action_out'],
            'aggregation_type': 'concat',
        },
        {
            "type": "internal_lstm",
            "size": 256,
        }
    ]
]
dc2_net_conv_with_eq = [
    [
        {
            'type' : 'input',
            'names' : ['global_in']
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
            'type' : 'output',
            'name' : 'global_out'
        }
    ],
    [
        {
            'type' : 'input',
            'names' : ['local_in']
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
            'type' : 'output',
            'name' : 'local_out'
        }
    ],
    [
        {
            'type' : 'input',
            'names' : ['local_in_two']
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
            'type' : 'output',
            'name' : 'local_out_two'
        }
    ],
    [
        {
            'type' : 'input',
            'names' : ['stats']
        },
        {
            "type": "embedding",
            "indices": 63,
            "size": 64
        },
        {
            "type": "nonlinearity",
            "name": 'tanh'
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
            'type' : 'output',
            'name' : 'stats_out'
        }
    ],
    [
        {
            'type' : 'input',
            'names' : ['equip']
        },
        {
            "type": "embedding",
            "indices": 74,
            "size": 64
        },
        {
            "type": "nonlinearity",
            "name": 'tanh'
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
            'type' : 'output',
            'name' : 'equip_out'
        }
    ],
    [
        {
            'type' : 'input',
            'names': ['global_out', 'local_out', 'local_out_two', 'stats_out', 'equip_out'],
            'aggregation_type': 'concat',
        },
        {
            "type": "dense",
            "size": 256,
            "activation": 'relu'
        },
        {
            'type' : 'output',
            'name' : 'first_FC'
        }
    ],
    [
        {
            'type' : 'input',
            'names': ['action']
        },
        {
            'type': 'flatten'
        },
        {
            'type': 'output',
            'name': 'action_out'
        }
    ],
    [
        {
            'type' : 'input',
            'names': ['first_FC', 'action_out'],
            'aggregation_type': 'concat',
        },
        {
            "type": "internal_lstm",
            "size": 256,
        }
    ]
]

dc2_net_conv_same_stat = [
    [
        {
            'type' : 'input',
            'names' : ['global_in']
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
            "size": 64,
            "window": (3, 3),
            "stride": 1,
            "activation": 'relu'
        },
        {
            "type": "conv2d",
            "size": 128,
            "window": (3, 3),
            "stride": 1,
            "activation": 'relu'
        },
        {
            'type': 'flatten'
        },
        {
            'type' : 'output',
            'name' : 'global_out'
        }
    ],
    [
        {
            'type' : 'input',
            'names' : ['local_in']
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
            "size": 64,
            "window": (3, 3),
            "stride": 1,
            "activation": 'relu'
        },
        {
            "type": "conv2d",
            "size": 128,
            "window": (3, 3),
            "stride": 1,
            "activation": 'relu'
        },
        {
            'type': 'flatten'
        },
        {
            'type' : 'output',
            'name' : 'local_out'
        }
    ],
    [
        {
            'type' : 'input',
            'names' : ['local_in_two']
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
            "size": 64,
            "window": (3, 3),
            "stride": 1,
            "activation": 'relu'
        },
        {
            "type": "conv2d",
            "size": 128,
            "window": (3, 3),
            "stride": 1,
            "activation": 'relu'
        },
        {
            'type': 'flatten'
        },
        {
            'type' : 'output',
            'name' : 'local_out_two'
        }
    ],
    [
        {
            'type': 'input',
            'names': ['stats']
        },
        {
            "type": "embedding",
            "indices": 82,
            "size": 64
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
            'type': 'output',
            'name': 'stats_out'
        }
    ],
    [
        {
            'type' : 'input',
            'names': ['global_out', 'local_out', 'local_out_two', 'stats_out'],
            'aggregation_type': 'concat',
        },
        {
            "type": "dense",
            "size": 256,
            "activation": 'relu'
        },
        {
            'type' : 'output',
            'name' : 'first_FC'
        }
    ],
    [
        {
            'type' : 'input',
            'names': ['action']
        },
        {
            'type': 'flatten'
        },
        {
            'type': 'output',
            'name': 'action_out'
        }
    ],
    [
        {
            'type' : 'input',
            'names': ['first_FC', 'action_out'],
            'aggregation_type': 'concat',
        },
        {
            "type": "internal_lstm",
            "size": 256,
        }
    ]
]

# Baseline net structure
dc2_baseline_embs = [
    [
        [
            {
                'type' : 'input',
                'names' : ['global_in']
            },
            {
                "type": "embedding",
                "indices": 7,
                "size": 16
            },
            {
                'type' : 'output',
                'name' : 'global_embs'
            }
        ],
        [
            {
                'type' : 'input',
                'names' : ['attr_global_1']
            },
            {
                "type": "embedding",
                "indices": 9,
                "size": 32
            },
            {
                'type' : 'output',
                'name' : 'attr_global_1_embs'
            }
        ],
        [
            {
                'type' : 'input',
                'names' : ['attr_global_2']
            },
            {
                "type": "embedding",
                "indices": 9,
                "size": 32
            },
            {
                'type' : 'output',
                'name' : 'attr_global_2_embs'
            }
        ],
        [
            {
                'type' : 'input',
                'names' : ['attr_global_3']
            },
            {
                "type": "embedding",
                "indices": 9,
                "size": 32
            },
            {
                'type' : 'output',
                'name' : 'attr_global_3_embs'
            }
        ],
        [
            {
                'type': 'input',
                'names': ['attr_global_4']
            },
            {
                "type": "embedding",
                "indices": 9,
                "size": 32
            },
            {
                'type': 'output',
                'name': 'attr_global_4_embs'
            }
        ],
        [
            {
                'type': 'input',
                'names': ['attr_global_5']
            },
            {
                "type": "embedding",
                "indices": 9,
                "size": 32
            },
            {
                'type': 'output',
                'name': 'attr_global_5_embs'
            }
        ],
        [
            {
                'type': 'input',
                'names': ['attr_global_1_embs', 'attr_global_2_embs', 'attr_global_3_embs', 'attr_global_4_embs', 'attr_global_5_embs'],
                'aggregation_type': 'concat',
                'axis': 3
            },
            {
                'type': 'conv2d',
                'activation': 'relu',
                'window': (1,1),
                'stride': 1,
                'size': 16
            },
            {
                'type': 'output',
                'name': 'weapon_emb'
            }
        ],
        {
            'type': 'input',
            'names': ['global_embs', 'weapon_emb'],
            'aggregation_type': 'concat',
            'axis': 3
        },
        {
            "type": "conv2d",
            "size": 32,
            "window": (3,3),
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
            'type' : 'output',
            'name' : 'global_out'
        }
    ],
    [
        [
            {
                'type' : 'input',
                'names' : ['local_in']
            },
            {
                "type": "embedding",
                "indices": 7,
                "size": 16
            },
            {
                'type' : 'output',
                'name' : 'local_embs'
            }
        ],
        [
            {
                'type' : 'input',
                'names' : ['attr_local_1']
            },
            {
                "type": "embedding",
                "indices": 9,
                "size": 32
            },
            {
                'type' : 'output',
                'name' : 'attr_local_1_embs'
            }
        ],
        [
            {
                'type' : 'input',
                'names' : ['attr_local_2']
            },
            {
                "type": "embedding",
                "indices": 9,
                "size": 32
            },
            {
                'type' : 'output',
                'name' : 'attr_local_2_embs'
            }
        ],
        [
            {
                'type' : 'input',
                'names' : ['attr_local_3']
            },
            {
                "type": "embedding",
                "indices": 9,
                "size": 32
            },
            {
                'type' : 'output',
                'name' : 'attr_local_3_embs'
            }
        ],
        [
            {
                'type': 'input',
                'names': ['attr_local_4']
            },
            {
                "type": "embedding",
                "indices": 9,
                "size": 32
            },
            {
                'type': 'output',
                'name': 'attr_local_4_embs'
            }
        ],
        [
            {
                'type': 'input',
                'names': ['attr_local_5']
            },
            {
                "type": "embedding",
                "indices": 9,
                "size": 32
            },
            {
                'type': 'output',
                'name': 'attr_local_5_embs'
            }
        ],
        [
            {
                'type': 'input',
                'names': ['attr_local_1_embs', 'attr_local_2_embs', 'attr_local_3_embs', 'attr_local_4_embs', 'attr_local_5_embs'],
                'aggregation_type': 'concat',
                'axis': 3
            },
            {
                'type': 'conv2d',
                'activation': 'relu',
                'window': (1,1),
                'stride': 1,
                'size': 16
            },
            {
                'type': 'output',
                'name': 'weapon_local_emb'
            }
        ],
        {
            'type': 'input',
            'names': ['local_embs', 'weapon_local_emb'],
            'aggregation_type': 'concat',
            'axis': 3
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
            'type' : 'output',
            'name' : 'local_out'
        }
    ],
    [
        [
            {
                'type': 'input',
                'names': ['local_in_two']
            },
            {
                "type": "embedding",
                "indices": 7,
                "size": 16
            },
            {
                'type': 'output',
                'name': 'local_two_embs'
            }
        ],
        [
            {
                'type': 'input',
                'names': ['attr_local_two_1']
            },
            {
                "type": "embedding",
                "indices": 9,
                "size": 32
            },
            {
                'type': 'output',
                'name': 'attr_local_two_1_embs'
            }
        ],
        [
            {
                'type': 'input',
                'names': ['attr_local_two_2']
            },
            {
                "type": "embedding",
                "indices": 9,
                "size": 32
            },
            {
                'type': 'output',
                'name': 'attr_local_two_2_embs'
            }
        ],
        [
            {
                'type': 'input',
                'names': ['attr_local_two_3']
            },
            {
                "type": "embedding",
                "indices": 9,
                "size": 32
            },
            {
                'type': 'output',
                'name': 'attr_local_two_3_embs'
            }
        ],
        [
            {
                'type': 'input',
                'names': ['attr_local_two_4']
            },
            {
                "type": "embedding",
                "indices": 9,
                "size": 32
            },
            {
                'type': 'output',
                'name': 'attr_local_two_4_embs'
            }
        ],
        [
            {
                'type': 'input',
                'names': ['attr_local_two_5']
            },
            {
                "type": "embedding",
                "indices": 9,
                "size": 32
            },
            {
                'type': 'output',
                'name': 'attr_local_two_5_embs'
            }
        ],
        [
            {
                'type': 'input',
                'names': ['attr_local_two_1_embs', 'attr_local_two_2_embs', 'attr_local_two_3_embs', 'attr_local_two_4_embs', 'attr_local_two_5_embs'],
                'aggregation_type': 'concat',
                'axis': 3
            },
            {
                'type': 'conv2d',
                'activation': 'relu',
                'window': (1,1),
                'stride': 1,
                'size': 16
            },
            {
                'type': 'output',
                'name': 'weapon_local_two_emb'
            }
        ],
        {
            'type': 'input',
            'names': ['local_two_embs', 'weapon_local_two_emb'],
            'aggregation_type': 'concat',
            'axis': 3
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
            'type' : 'output',
            'name' : 'local_out_two'
        }
    ],
    [
        {
            'type' : 'input',
            'names' : ['agent_stats']
        },
        {
            "type": "embedding",
            "indices": 107,
            "size": 256
        },
        {
            "type": "nonlinearity",
            "name": 'tanh'
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
            'type' : 'output',
            'name' : 'agent_stats_out'
        }
    ],
    [
        {
            'type': 'input',
            'names': ['target_stats']
        },
        {
            "type": "embedding",
            "indices": 32,
            "size": 256
        },
        {
            "type": "nonlinearity",
            "name": 'tanh'
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
            'type': 'output',
            'name': 'target_stats_out'
        }
    ],
    [
        {
            'type' : 'input',
            'names': ['global_out', 'local_out', 'local_out_two', 'agent_stats_out', 'target_stats_out'],
            'aggregation_type': 'concat',
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
dc2_baseline_conv = [
    [
        {
            'type' : 'input',
            'names' : ['global_in']
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
            'type' : 'output',
            'name' : 'global_out'
        }
    ],
    [
        {
            'type' : 'input',
            'names' : ['local_in']
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
            'type' : 'output',
            'name' : 'local_out'
        }
    ],
    [
        {
            'type' : 'input',
            'names' : ['local_in_two']
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
            'type' : 'output',
            'name' : 'local_out_two'
        }
    ],
    [
        {
            'type' : 'input',
            'names' : ['stats']
        },
        {
            "type": "embedding",
            "indices": 136,
            "size": 128
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
            'type' : 'output',
            'name' : 'stats_out'
        }
    ],
    [
        {
            'type' : 'input',
            'names': ['global_out', 'local_out', 'local_out_two', 'stats_out'],
            'aggregation_type': 'concat',
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
dc2_baseline_conv_with_eq = [
    [
        {
            'type' : 'input',
            'names' : ['global_in']
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
            'type' : 'output',
            'name' : 'global_out'
        }
    ],
    [
        {
            'type' : 'input',
            'names' : ['local_in']
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
            'type' : 'output',
            'name' : 'local_out'
        }
    ],
    [
        {
            'type' : 'input',
            'names' : ['local_in_two']
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
            'type' : 'output',
            'name' : 'local_out_two'
        }
    ],
    [
        {
            'type' : 'input',
            'names' : ['stats']
        },
        {
            "type": "embedding",
            "indices": 63,
            "size": 64
        },
{
            "type": "nonlinearity",
            "name": 'tanh'
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
            'type' : 'output',
            'name' : 'stats_out'
        }
    ],
    [
        {
            'type' : 'input',
            'names' : ['equip']
        },
        {
            "type": "embedding",
            "indices": 74,
            "size": 64
        },
{
            "type": "nonlinearity",
            "name": 'tanh'
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
            'type' : 'output',
            'name' : 'equip_out'
        }
    ],
    [
        {
            'type' : 'input',
            'names': ['global_out', 'local_out', 'local_out_two', 'stats_out', 'equip_out'],
            'aggregation_type': 'concat',
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


dc2_baseline_same_stat = [
    [
        {
            'type' : 'input',
            'names' : ['global_in']
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
            "size": 64,
            "window": (3, 3),
            "stride": 1,
            "activation": 'relu'
        },
        {
            "type": "conv2d",
            "size": 128,
            "window": (3, 3),
            "stride": 1,
            "activation": 'relu'
        },
        {
            'type': 'flatten'
        },
        {
            'type' : 'output',
            'name' : 'global_out'
        }
    ],
    [
        {
            'type' : 'input',
            'names' : ['local_in']
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
            "size": 64,
            "window": (3, 3),
            "stride": 1,
            "activation": 'relu'
        },
        {
            "type": "conv2d",
            "size": 128,
            "window": (3, 3),
            "stride": 1,
            "activation": 'relu'
        },
        {
            'type': 'flatten'
        },
        {
            'type' : 'output',
            'name' : 'local_out'
        }
    ],
    [
        {
            'type' : 'input',
            'names' : ['local_in_two']
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
            "size": 64,
            "window": (3, 3),
            "stride": 1,
            "activation": 'relu'
        },
        {
            "type": "conv2d",
            "size": 128,
            "window": (3, 3),
            "stride": 1,
            "activation": 'relu'
        },
        {
            'type': 'flatten'
        },
        {
            'type' : 'output',
            'name' : 'local_out_two'
        }
    ],
    [
        {
            'type': 'input',
            'names': ['stats']
        },
        {
            "type": "embedding",
            "indices": 82,
            "size": 64
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
            'type': 'output',
            'name': 'stats_out'
        }
    ],
    [
        {
            'type' : 'input',
            'names': ['global_out', 'local_out', 'local_out_two', 'stats_out'],
            'aggregation_type': 'concat',
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


net = [
    [
        {
            'type' : 'input',
            'names' : ['global_in']
        },
        {
            "type": "embedding",
            "indices": 12,
            "size": 32
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
            'type' : 'output',
            'name' : 'global_out'
        }
    ],
    [
        {
            'type' : 'input',
            'names' : ['local_in']
        },
        {
            "type": "embedding",
            "indices": 12,
            "size": 32
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
            'type' : 'output',
            'name' : 'local_out'
        }
    ],
    [
        {
            'type' : 'input',
            'names' : ['local_in_two']
        },
        {
            "type": "embedding",
            "indices": 12,
            "size": 32
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
            'type' : 'output',
            'name' : 'local_out_two'
        }
    ],
    [
        {
            'type' : 'input',
            'names' : ['stats']
        },
        {
            "type": "embedding",
            "indices": 82,
            "size": 64
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
            'type' : 'output',
            'name' : 'stats_out'
        }
    ],
    [
        {
            'type' : 'input',
            'names': ['global_out', 'local_out', 'local_out_two', 'stats_out'],
            'aggregation_type': 'concat',
        },
        {
            "type": "dense",
            "size": 256,
            "activation": 'relu'
        },
        {
            'type' : 'output',
            'name' : 'first_FC'
        }
    ],
    [
        {
            'type' : 'input',
            'names': ['action']
        },
        {
            'type': 'flatten'
        },
        {
            'type': 'output',
            'name': 'action_out'
        }
    ],
    [
        {
            'type' : 'input',
            'names': ['first_FC', 'action_out'],
            'aggregation_type': 'concat',
        },
        {
            "type": "internal_lstm",
            "size": 256,
        }
    ]
]

# Baseline net structure
baseline = [
    [
        {
            'type' : 'input',
            'names' : ['global_in']
        },
        {
            "type": "embedding",
            "indices": 12,
            "size": 32
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
            'type' : 'output',
            'name' : 'global_out'
        }
    ],
    [
        {
            'type' : 'input',
            'names' : ['local_in']
        },
        {
            "type": "embedding",
            "indices": 12,
            "size": 32
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
            'type' : 'output',
            'name' : 'local_out'
        }
    ],
    [
        {
            'type' : 'input',
            'names' : ['local_in_two']
        },
        {
            "type": "embedding",
            "indices": 12,
            "size": 32
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
            'type' : 'output',
            'name' : 'local_out_two'
        }
    ],
    [
        {
            'type' : 'input',
            'names' : ['stats']
        },
        {
            "type": "embedding",
            "indices": 82,
            "size": 64
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
            'type' : 'output',
            'name' : 'stats_out'
        }
    ],
    [
        {
            'type' : 'input',
            'names': ['global_out', 'local_out', 'local_out_two', 'stats_out'],
            'aggregation_type': 'concat',
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
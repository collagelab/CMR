"""
Constants for router training.

This module centralizes all magic numbers and configuration constants
used in router training to improve maintainability.
"""

# Default candidate set sizes
DEFAULT_K_TOTAL = 64
DEFAULT_K_SEMANTIC = 48
DEFAULT_K_FAR = 8
DEFAULT_K_HARD = 7

# Default router hyperparameters
DEFAULT_ROUTER_EMBEDDING_DIM = 256
DEFAULT_ROUTER_TAU = 0.07
DEFAULT_ROUTER_EPS = 0.1
DEFAULT_ROUTER_K_NEIGHBORS = 3

# Default hard negative mining
DEFAULT_MINE_EVERY_STEPS = 200
DEFAULT_K_HARD_POOL = 20
DEFAULT_SEMANTIC_POOL_SIZE = 512
DEFAULT_MAX_POOL_SIZE = 1024
DEFAULT_MAX_EXAMPLES_PER_UPDATE = 128

# Default graph regularization
DEFAULT_GRAPH_TAU = 0.07
DEFAULT_GRAPH_TAU_TARGET = 0.1
DEFAULT_GRAPH_ALPHA_DOMAIN = 0.3
DEFAULT_MAX_GRAPH_MODELS = 256

# Validation thresholds
LOSS_DIFF_TOLERANCE = 1e-5
CE_LOSS_DIFF_TOLERANCE = 1e-6

# Debug defaults
DEFAULT_DEBUG_EVERY = 100
DEFAULT_DEBUG_FIRST_STEPS = 50

# Semantic pool modes
SEMANTIC_POOL_MODE_DOMAIN_ONLY = "domain_only"
SEMANTIC_POOL_MODE_PARENT_GROUP = "parent_group"
SEMANTIC_POOL_MODE_TAXONOMY_GRAPH = "taxonomy_graph"

# Pooling strategies
POOLING_LAST_TOKEN = "last_token"
POOLING_MEAN = "mean"


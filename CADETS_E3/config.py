########################################################
#
#                   Artifacts path
#
########################################################

# The directory of the raw logs
RAW_DIR = "../data/cadets/"

# The directory to save all artifacts
ARTIFACT_DIR = "./artifact/"

# The directory to save the vectorized graphs
GRAPHS_DIR = ARTIFACT_DIR + "graph_embeddings/"

# The directory to save the models
MODELS_DIR = ARTIFACT_DIR + "models/"

# The directory to save the results after testing
TEST_RE = ARTIFACT_DIR + "test_re/"

# The directory to save all visualized results
VIS_RE = ARTIFACT_DIR + "vis_re/"

# The directory with malicious ground truth
MALICIOUS_DIR = ARTIFACT_DIR + "malicious/"

########################################################
#
#               Database settings
#
########################################################

# Database name
DATABASE = 'tc_cadet_dataset_db_001'

# Only config this setting when you have the problem mentioned
# in the Troubleshooting section in settings/environment-settings.md.
# Otherwise, set it as None
# host = '/var/run/postgresql/'
HOST = None

# Database user
USER = 'postgres'

# The password to the database user
PASSWORD = 'password'

# The port number for Postgres
PORT = '5432'


########################################################
#
#               Graph semantics
#
########################################################

# The directions of the following edge types need to be reversed
edge_reversed = [
    "EVENT_ACCEPT",
    "EVENT_RECVFROM",
    "EVENT_RECVMSG"
]

# The following edges are the types only considered to construct the
# temporal graph for experiments.
include_edge_type=[
    "EVENT_WRITE",
    "EVENT_READ",
    "EVENT_CLOSE",
    "EVENT_OPEN",
    "EVENT_EXECUTE",
    "EVENT_SENDTO",
    "EVENT_RECVFROM",
]

# The map between edge type and edge ID
rel2id = {
 1: 'EVENT_WRITE',
 'EVENT_WRITE': 1,
 2: 'EVENT_READ',
 'EVENT_READ': 2,
 3: 'EVENT_CLOSE',
 'EVENT_CLOSE': 3,
 4: 'EVENT_OPEN',
 'EVENT_OPEN': 4,
 5: 'EVENT_EXECUTE',
 'EVENT_EXECUTE': 5,
 6: 'EVENT_SENDTO',
 'EVENT_SENDTO': 6,
 7: 'EVENT_RECVFROM',
 'EVENT_RECVFROM': 7
}

########################################################
#
#                   Model dimensionality
#
########################################################

# Node Embedding Dimension
node_embedding_dim = 16

# Node State Dimension
node_state_dim = 100

# Neighborhood Sampling Size
neighbor_size = 20

# Edge Embedding Dimension
edge_dim = 100

# The time encoding Dimension
time_dim = 100


########################################################
#
#                   Train&Test
#
########################################################

# Batch size for training and testing
BATCH = 1024

# Parameters for optimizer
lr=0.00005
eps=1e-08
weight_decay=0.01

epoch_num=50

# The size of time window, 60000000000 represent 1 min in nanoseconds.
# The default setting is 15 minutes.
time_window_size = 60000000000 * 2


########################################################
#
#                   Threshold
#
########################################################

beta_day6 = 100
beta_day7 = 100

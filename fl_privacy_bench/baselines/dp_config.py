from fl_privacy_bench.core.config import DEVICE

NUM_CLIENTS = 100
CLIENTS_PER_ROUND = 10
BATCH_SIZE = 16
LEARNING_RATE = 0.01
NUM_ROUNDS = 200

epsilon = 0.1
delta = 1e-5
sensitivity = 0.01
MAX_PARAM_ABS = 10.0
MAX_GRAD_NORM = 1.0

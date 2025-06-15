### Append stop word ###
STOP_WORD = True
### Capitalization
CAPITALIZATION = True

### small number
EPSILON = 1e-100

### Inference Types ###
GREEDY = 0
BEAM = 1
BEAM_K = 2
VITERBI = 2
INFERENCE = GREEDY
# INFERENCE = VITERBI

### Smoothing Types ###
LAPLACE = 0
LAPLACE_FACTOR = 0.2
INTERPOLATION = 1
LAMBDAS = None
SMOOTHING = INTERPOLATION

### Append stop word ###
STOP_WORD = True

### Capitalization
CAPITALIZATION = True

# NGRAMM
NGRAMM = 3

## Handle unknown words TnT style
TNT_UNK = True
UNK_C = 10  # words with count to be considered
UNK_M = 10  # substring length to be considered

### Smoothing Types ###
TRIGRAM_LAMBDAS = 0.7, 0.15, 0.15
BIGRAM_LAMBDAS = 0.85, 0.15
LINEAR_INTERPOLATION = 1
LAMBDAS_LINEAR_INTERPOLATION = 0.1, 0.3, 0.6

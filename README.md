# Execution Instructions

## Data
Ensure that all files are located in the same working director and that the code is also located in this directory, otherwise, ensure that the code properly imports the data in the lines 826-828. 

## Running the Code
First, the only file that needs to be run is the pos_tagger.py file, the rest are only dependencies. Next, once you run the code, you will be prompted in the terminal to select one of three decoding methods: GREEDY, BEAM_K, VITERBI. If BEAM_K is chosen, then you will be allowed to input an integer value for beam width. GREEDY and VITERBI will skip this step. Next, once a decoding method is chosen, it will prompt you to choose a smoothing method: LAPLACE or LINEAR_INTERPOLATION. Lastly, the terminal will command you to choose a bigram or trigram model, by inputing 2 or 3, respectively. Thereafter, the code should run.

## Output
The output of the process above should be some accuracy metrics on the dev set, a confusion matrix for parts-of-speech predictions on the dev set, and finally a csv of predictions on the unseen testing dataset. 
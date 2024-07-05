# Code and Data for CIKM'16 paper: Learning Points and Routes to Recommend Trajectories - Reproduction
---------------------------------------------------------------------------------------

This repository corresponds to the paper TODO which goal was to reproduce an experiment from paper "Learning Points and Routes to Recommend Trajectories" (https://dl.acm.org/doi/10.1145/2983323.2983672). The code was originally sourced from https://github.com/computationalmedia/tour-cikm16.

The code as is was tested on Windows 10, versions of libraries etc. are described in the paper/code.

The code needs to be run in the same way as the original, 
    first 
    rank_markov.ipynb and then 
    parse_results.ipynb (it is needed to adjust path to RankSvm in first notebok).


# Code
Preprocessing Data 
- preprocessing_data.py
Metode 
- POI RANK = POIRank.py
- Popularity = POIRank.py
- Markov =  Transition_matrix.py
- Rank Markov = RankTransition_matrix.py
Helper
- Utiutils.py
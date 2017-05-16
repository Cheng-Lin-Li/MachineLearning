# This is an implementation of Hidden Markov Model (HMM) - Viterbi Algorithm in Python 3

## Machine Learning Algorithm: Hidden Markov Model - Viterbi Algorithm.

## Task:
The file hmm-data.txt contains a map of a 10-by-10 2D grid-world. The most up-left cell has a coordinate of (0, 0). The ith row has a x coordinate of i, and the jth column has a y coordinate of j. The row and column indices start from 0. The free cells are represented as '1's and the obstacles are represented as '0's. There are four towers, one in each of the four corners, as indicated in the data file. This task is to use a Hidden Markov Model to figure out the most likely trajectory of a robot in this grid-world. Assume that the initial position of the robot has a uniform prior over all free cells. In each time-step, the robot moves to one of its four neighboring free cells chosen uniformly at random. 


At a given cell, the robot measures L2 distances (Euclidean distances) to each of the towers. For a true distance d, the robot records a noisy measurement chosen uniformly at random from the set of numbers in the interval [0.7d, 1.3d] with one decimal place. These measurements for 11 time-steps are also provided in the data file. You should output the coordinates of the most likely trajectory of the robot for 11 time-steps. The Viterbi algorithm is the implemented algorithm for this task. For tie breaking, it always prefer the one with a smaller x coordinate, and a smaller y coordinate if the x coordinates are equal.


#### Usage: python DecisionTree.py dt-data.txt	

#### Input: A text file 'hmm-data.txt'.
The text file include the map of 2D grid-world, the location of towers that send out signals, four signals the robot receives from each tower.

#### Output: Output includes the coordinates of the most likely trajectory of the robot for 11 timesteps.


## Data Structure:
  1. STATES: States list to store labels of each state
  
    => [state1, state2, ..., stateN].
  2. TOWER_LOCATION: location list for towers
  
    => [tower1[x1,y1], ..., towerN[xN,yN]]
  3. NOISY_DISTANCES: The observation sequence of evidence
  
    => [Time1[distance between robot and Tower1, ..., distance between robot and TowerN], ..., TimeN[distance between robot and Tower1, ..., distance between robot and TowerN]]
  4. OBSERVATIONS = []: Observation list to store labels of evidence
  
    => [observation1, observation2,..., observationN]
  5. INITIAL_PROB: The matrix of probability from system start to first state
  
    => [[probability from start to state1, ..., probability from start to stateN]]
  6. TRANSITION_MATRIX
  
    => [Time T at state1[Probability to Time T+1 to state1,..., Probability to Time T+1 to stateN], ..., Time T at stateN[Probability to Time T+1 to state1,..., Probability to Time T+1 to stateN]]
  7. EVIDENCE_MATRIX: Store observation of evidence. It support multiple evidences in the same time step.
  
    => [[evidence1],[evidence2],...,[evidenceN]], [evidenceN]=[state1[Probability of evidence1, ..., Probability of evidenceN], ..., stateN[Probability of evidence1, ..., Probability of evidenceN]]
  8. DISTANCE_MATRIX: Store distance(observation) probability of each cell for each tower
  
    => [[tower1],...,[tower4]], [tower1]=[cell1[Probability of distance0, ..., Probability of distanceN]]
  9. VALIDCELL2COORDINATION: Store the map between state to coordination of the grid
  
    => {state1,[x,y], ...,stateN[x,y]}
  10. DISTANCE_MATRIX: Store distance(observation) probability of each cell for each tower
  
    => [[tower1],...,[tower4]]
    => [tower1]=[cell1[Probability of distance0, ..., Probability of distanceN]] stateN[Probability of evidence1, ..., Probability of evidenceN]]
  11. _state_prob: List of consolidate probability for every state at time t
  
    => [probability of state1, ..., probability of stateN]


## Process:
  1. Get input data from file. 
  2. Format the GRID_WORLD, TOWER_LOCATION, NOISY_DISTANCES from data file. 
  3. From GRID_WORLD, generate matrix STATES, matrix INITIAL_PROB, and matrixes TRANSITION_MATRIX, VALIDCELL2COORDINATION through getCPT(GRID_WORLD).
  4. Generate matrixes DISTANCE_MATRIX, OBSERVATIONS through getDistance(TOWER_LOCATION, GRID_WORLD).
  5. Instantiate a HMM variable through HMM(STATES, OBSERVATIONS, INITIAL_PROB, TRANSITION_MATRIX, DISTANCE_MATRIX, 'viterbi').
  6. Use the function viterbi (self, observation_seq) which implementing the Viterbi algorighm to calculate the maximum probability of state in the last time-step, then backtrack the state provides the condition for getting this probability, until the starting state is determined.
  7. Print out the possible route of robots.
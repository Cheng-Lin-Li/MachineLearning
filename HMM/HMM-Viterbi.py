#!/usr/bin/env python
# encoding: utf-8
'''
Machine Learning Algorithm Name: Hidden Markov Model implementation.

This is a sample program to demonstrate the implementation of Hidden Markov Model - Viterbi algorithm.

@author: Cheng-Lin Li a.k.a. Clark

@copyright:  2017 Cheng-Lin Li@University of Southern California. All rights reserved.

@license:    Licensed under the GNU v3.0. https://www.gnu.org/licenses/gpl.html

@contact:    jianfali@usc.edu, clark.cl.li@gmail.com
@version:    2.1

@create:    November 28, 2016
@updated:   February, 16, 2017

The states of this assignment are the position of cells which define from f1=(0,0), f2=(1,0), f3=(2,0)...f87=(9,9)
The symbols of this assignment are distance of each tower which define from 0.1, 0.2, ..., 12.7.
There are four evident probability tables from 4 towers.

TOWER_LOCATION: location list for towers.
    =>[tower1[x1,y1], ..., towerN[xN,yN]]
NOISY_DISTANCES: The observation sequence of evidence
    =>[Time1[distance between robot and Tower1, ..., distance between robot and TowerN], ..., TimeN[distance between robot and Tower1, ..., distance between robot and TowerN]]
STATES: States list to store labels of each state.
    => [state1, state2, ..., stateN]
    => In this implementation state1 map to (x=0, y=0), state2 map to (x=1, y=0)
OBSERVATIONS = []: Observation list to store labels of evidence.
    => [observation1, observation2,..., observationN] 
INITIAL_PROB: The matrix of probability from system start to first state.
    => [[probability from start to state1, ..., probability from start to stateN]]
TRANSITION_MATRIX 
    => [Time T at state1[Probability to Time T+1 to state1,..., Probability to Time T+1 to stateN], ..., Time T at stateN[Probability to Time T+1 to state1,..., Probability to Time T+1 to stateN]]
EVIDENCE_MATRIX: Store observation of evidence. It support multiple evidences in the same time step.
    => [[evidence1],[evidence2],...,[evidenceN]], [evidenceN]=[state1[Probability of evidence1, ..., Probability of evidenceN], ..., stateN[Probability of evidence1, ..., Probability of evidenceN]]
DISTANCE_MATRIX: Store distance(observation) probability of each cell for each tower . 
    => [[tower1],...,[tower4]], [tower1]=[cell1[Probability of distance0, ..., Probability of distanceN]]

VALIDCELL2COORDINATION: Store the map between state to coordination of the grid
    =>{state1,[x,y], ...,stateN[x,y]}
'''
import math
import numpy as np

DECIMAL_PRECISION = 1 #decimal place is 0.1**1
GRID_WORLD_ROW = 10 # Y axis row
GRID_WORLD_COLUMN = 10 # X axis elements in each row
GRID_WORLD = []
TOWER_NUMBERS = 4
TOWER_LOCATION = []
NOISY_DISTANCES = []
TIME_STEPS = 11
NOISY_MIN = 0.7
NOISY_MAX = 1.3

STATES = []
OBSERVATIONS = []
INITIAL_PROB = []
TRANSITION_MATRIX = []
DISTANCE_MATRIX = []
EVIDENCE_MATRIX = []

STATE_SEQUENCE=[]

VALIDCELL2COORDINATION={}


def load_parameters(file_name):
    _gw = []
    _tl = []
    _nd = []
    with open(file_name) as f:
        f.readline()   # skip Grid-World:
        f.readline()   # skip the blank line
        for _ in range (GRID_WORLD_ROW):
            _gw.append([int(_x) for _x in f.readline().split()])
            
        f.readline()   # skip the blank line
        f.readline()   # skip the blank line
        f.readline()   # skip Tower Locations:
        f.readline()   # skip the blank line
        for _ in range (TOWER_NUMBERS):
            _tl.append([int(_x) for _x in f.readline().split()[2:]]) #skip two label words.
                     
        f.readline()   # skip the blank line
        f.readline()   # skip the blank line
        f.readline()   # skip Noisy Distances to towers 1, 2, 3, 4:
        f.readline()   # skip the blank line
        for _ in range (TIME_STEPS):
            _nd.append([float(_x) for _x in f.readline().split()])
                                            
        return _gw, _tl, _nd

def Print_List(l):
    for i in l:
        print(i)
    
def NextStepProb(x, y, grid_world):
    # The ith row has a x coordinate of i, and the jth column has a y coordinate of j on the grid world.
    # Probability of next step is 1/(all available next steps)
    _a = 0

    if((y-1 >=0) and (grid_world[x][y-1]==1)):
        # Left cell is available.
        _a += 1
    else: pass    
    
    if ((y+1 < GRID_WORLD_COLUMN) and (grid_world[x][y+1]==1)):
        # Right cell is available.
        _a += 1
    else: pass 
            
    if ((x-1 >=0) and (grid_world[x-1][y]==1)):
        # Up cell is available.
        _a += 1
    else: pass

    if ((x+1 < GRID_WORLD_ROW) and (grid_world[x+1][y]==1)):
        # Down cell is available.
        _a += 1
    else: pass 
            
    if (_a > 0):
        return 1/_a    
    else:
        return 0  
      
    
def GetProbability(x_1, y_1, x, y, grid_world):
    #The ith row has a x coordinate of i, and the jth column has a y coordinate of j on the grid world.
    _a = 0 # available free cells based on given position x_1, y_1
    _n_x = 0 #neighbor in x axis
    _n_y = 0 #neighbor in y axis
  
    if((x-1 == x_1) and (y == y_1)):
        # Are (x, y) next to (under) the (x_1, y_1)?
        return NextStepProb(x_1, y_1, grid_world)
    elif ((x+1 == x_1) and (y == y_1)):
        # Are (x, y) next to (at top of) the (x_1, y_1)?
        return NextStepProb(x_1, y_1, grid_world)
    elif ((y-1 == y_1) and (x == x_1)):
        # Are (x, y) next to (at right of) the (x_1, y_1)?
        return NextStepProb(x_1, y_1, grid_world)
    elif ((y+1 == y_1) and (x == x_1)):
        # Are (x, y) next to (at left of) the (x_1, y_1)?
        return NextStepProb(x_1, y_1, grid_world)    
    else:
        return 0    

# I revised the line: for i in range(0, int(_max_distance * NOISY_MAX /_decimal_unit)+1).
def getDistance(towerlocation, grid_world):
    #DISTANCE_MATRIX: Store distance(observation) probability of each cell for each tower . 
    #=> [[tower1],...,[tower4]], 
    #[tower1]=[cell1[Probability of distance0, ..., Probability of distanceN]] stateN[Probability of evidence1, ..., Probability of evidenceN]]
    _prob_m = []
    _dm = [] #Distance matrix
    _o = []  #Observations
    _max_distance = 0
    _act_dist = 0 #actual distance between cell to tower
    _min_dist = 0 #Due to noise, the minimum distance return to robot will be 0.7*d
    _max_dist = 0 #Due to noise, the minimum distance return to robot will be 1.3*d
    _prob = 0.0
    _decimal_unit = 0.1**DECIMAL_PRECISION
    _tx, _ty = 0, 0 #the coordinate of tower
    GRID_WORLD_ROW, GRID_WORLD_COLUMN = np.array(grid_world).shape
    _max_distance = math.sqrt((GRID_WORLD_ROW-1)**2 + (GRID_WORLD_COLUMN-1)**2)

    for i in range(0, int(_max_distance * NOISY_MAX/_decimal_unit)+1): # _max_distance multiply NOISY_MAX will be the maximum distance in noisy environment
        _o.extend([round(i*_decimal_unit,DECIMAL_PRECISION)])
    
    for tower in towerlocation:
        _prob_m = []
        for cell in STATES:
            _tx = tower[0]
            _ty = tower[1]
            _dist = math.sqrt((VALIDCELL2COORDINATION[cell][0]-_tx)**2+(VALIDCELL2COORDINATION[cell][1]-_ty)**2)
            _min_dist = round(NOISY_MIN * _dist, DECIMAL_PRECISION)
            _max_dist = round(NOISY_MAX * _dist, DECIMAL_PRECISION)
            _prob = 1/((_max_dist - _min_dist + _decimal_unit)/_decimal_unit) # probability of distance between cell to tower for each state 
            _prob_m.append([])
            for i in _o:
                if (_min_dist <= i and i <= _max_dist):
                    _prob_m[cell-1].extend([_prob])
                else:
                    _prob_m[cell-1].extend([0])                    
        _dm.append(_prob_m)
    return _dm, _o


def getCPT(grid_world):
    #The ith row has a x coordinate of i, and the jth column has a y coordinate of j on the grid world.
    _s = [] #states
    _tm = [] #transition matrix
    _ip = [] #initial probability
    _total_cells = 0
    _valid_cells = 0
    _x = 0
    _y = 0
    _x_1 = 0
    _y_1 = 0
    _valid2coord = {}
    _state = 0 #total _valid_cells+1 states, from X(0) ~ X(_valid_cells).
    
    # Match the state to x, y coordination. 
    for _r in grid_world: #row is x
        for _c in _r: #column is y
            if _c == 1:
                _valid2coord[(_valid_cells+1)] =[_x, _y]
                _s.extend([_valid_cells+1]) #From state1
                _valid_cells += 1 
                _total_cells += 1
                _y += 1
            else:
                _total_cells +=1
                _y += 1
        _y = 0 # next row of data
        _x += 1
    # Compute probability for S0 to S1
    init_p = 1/_valid_cells #Same probability for each valid cell.
    _ip = np.ones((1, _valid_cells))*init_p
    print('total cells=%d, valid cells=%d, initial probability=%f' % (_total_cells, _valid_cells, init_p))
    # Compute probability for S1 for X2.
    _tm = np.zeros((_valid_cells, _valid_cells))
    # Define transition matrix
    # _tm = [[S0], [S1], [S2]....[Sn]] to represent the probability of robot locate in each cell if the robot locate in cell n at time (t-1).
    # [Sn]=[x0, x1, x2,....xn] to represent the probability of robot locate in each cell if the robot previous locate in cell n at time (t). 
    
    for _r in range (_valid_cells): #xt-1=f1, f2,...,f87
        for _c in range (_valid_cells):
            #Calculate from St-1 to St. xt=f1, f2,...,f87
            _x_1, _y_1 = _valid2coord[_r+1] #Get Xt-1 position.
            #print('x=%d, y=%d'%(_x, _y))
            _x, _y = _valid2coord[_c+1] #Get Xt position.
            _tm[_r][_c] = GetProbability(_x_1, _y_1, _x, _y, grid_world)
            
    return _s, _ip, _tm, _valid2coord


class HMM(object):
    '''
    Currently only support Viterbi Algorithm but can adopt other algorithms into this module in the furture.
    1. def viterbi(self, observation_seq): To support multiple evidences.
        Input: observation sequence of evidences.
        Return: decode the sequences of hidden state 
    2. def eliminate_evidences(self, observation_seq, step): To support multiple evidences      
        Input: observation sequence of evidences, what is the time step
        Return the probability of combination of all evidences.
    3. def max_prob_state(self, states_prob): To get maximum probability of state list at time t
        Input: current states of probability which constrain by the probability of previous states
        Return: max_prob list for each state at time t and its previous state at time t-1    
    self.trace_back_prob = list of the highest probability at each time steps.
        =>[Time0[highest probability for each state], ..., TimeT[highest probability for each state]]
        =>[highest probability for each state]=[highest probability of state1, ..., highest probability of stateN]
    self.trace_back_states 
        =>[Time1[previous state i, ..., previous state j], ..., TimeN[previous state i, ..., previous state j]]
        =>TimeN[previous state i, ..., previous state j]
            =[previous state i at TimeN-1 has the highest probability to state0, ..., previous state j at TimeN-1 has the highest probability to stateN ]
    '''

    def __init__(self, states, observations, init_prob=None, transition_prob=None, emission_prob=None, algorithm='viterbi'):
        self.states = states
        self.states_len = len(states)
        self.observations = observations
        self.init_prob = init_prob
        self.transition_prob = transition_prob
        self.emission_prob = emission_prob
        self.trace_back_prob = [] #record down highest probability at each time steps
        self.trace_back_states = []
        self.algorithm = algorithm
        if algorithm == 'viterbi':
            self.decode = self.viterbi
        else :
            raise ValueError('Unknown algorithm: %s'%(algorithm))           
        
    def viterbi(self, observation_seq):
        _obs_seq = observation_seq
        _time_steps = 0
        _max_prob = 0
        _states_max_prob = [] # The highest probability of each state at time_step t
        _previous_states = [] # The previous state link to current state
        _states_seq = [] # The out put of state sequences.
        _em_prob = [] #The integrated emission probability from multiple emission probability tables.
        if (np.array(_obs_seq).ndim == 1): #add sequence from one to two dimensions
            _obs_seq = [observation_seq]
        else:
            pass
        if (np.array(self.emission_prob).ndim ==1):
            self.emission_prob = [self.emission_prob]
        else:
            pass
        
        _time_steps = len(observation_seq)
        if (_time_steps == 0 or len(self.init_prob[0]) == 0 ): #If input observation sequence is zero length.
            return _states_seq
        else:
            pass
               
        for step in range(_time_steps):
            #emission_prob
            _em_prob = self.eliminate_evidences(_obs_seq, step) #Get the probability of state from given observation sequence of evidences
            if step == 0: #First step
                _states_max_prob = self.init_prob * _em_prob
#                self.trace_back_states.append(_states_max_prob[0])              
            else:
                _states_max_prob, _previous_states = self.max_prob_state(_states_max_prob * self.transition_prob) 
                _states_max_prob = _states_max_prob * _em_prob
#                print('step', step)
#                print('_states_max_prob:', _states_max_prob)
#                print('_em_prob:', _em_prob)
                self.trace_back_states.append(_previous_states)
                
        _states_seq.extend([self.max_prob(_states_max_prob)]) #Record down the highest probability of state in last time step.
        #back track to each time steps for the highest probability of state in each time step.
        self.trace_back_states.reverse()
        for _previous_state_list in self.trace_back_states:
            _states_seq.extend([_previous_state_list[_states_seq[-1]]])
        _states_seq.reverse()    
        return _states_seq
    
    def eliminate_evidences(self, observation_seq, step):
        #To support multiple evidences
        # Return the probability of combination of all evidences.
        # _tmp_state_prob: List of probability list for every state at time t for all evidences.
        #    =>[Evidence1[probability of state1, ..., probability of stateN], ..., EvidenceN[probability of state1, ..., probability of stateN]]
        # _state_prob: List of consolidate probability for every state at time t.
        #    =>[probability of state1, ..., probability of stateN]
        _state_prob = np.ones((1, self.states_len))
        _tmp_state_prob = []
        _emission_m = []
        _step = step
        
        for i, _ev in enumerate(observation_seq[step]): #evidence i at step of time
            _tmp_state_prob.append([])
            for j, _st in enumerate(self.states): # state j
                for k, _obs in enumerate(self.observations): #observation k
                    if(_ev == _obs): #If evidence value is match to the label of observations.
                        _tmp_state_prob[i].extend([self.emission_prob[i][j][k]]) #Get the probability from emission probability matrix.
                        break
                    else:
                        pass

        for _tmp_prob in _tmp_state_prob: #Consolidate evidence from multiple to one
            _state_prob *= _tmp_prob 

#        print('Print the elimination results for step %d: %s'%(step, _state_prob))
        return _state_prob
        
    def max_prob_state(self, states_prob):
        # To get maximum probability of state list at time t
        # It will return max_prob list for each state at time t and its previous state at time t-1 
        # For tie breaking, please always prefer the one with a smaller x coordinate, and a smaller y coordinate if the x coordinates are equal.
        _sts_prob = states_prob
        _prob = 0
        _max_prob = 0
        _max_pre_state = 0
        _state_len = self.states_len
        _max_probs_list = []
        _max_pre_states_list = []
        
        for _current_state in range(_state_len): #Review each column from row1, ..., rowN.
            _max_prob = 0
            _previous_state = 0
            for _previous_state in range(_state_len):
                _prob = _sts_prob[_current_state][_previous_state]
                if (_prob > _max_prob):
                    _max_prob = _prob
                    _max_pre_state = _previous_state
                else:
                    pass 
            _max_probs_list.extend([_max_prob])  
            _max_pre_states_list.extend([_max_pre_state]) 
        return _max_probs_list, _max_pre_states_list

    def max_prob(self, states_prob):
        # To get maximum probability of state at time t
        # For tie breaking, please always prefer the one with a smaller x coordinate, and a smaller y coordinate if the x coordinates are equal.
        _sts_prob = states_prob
        _state = 0
        _max_prob = 0
        
        for _st, _prob in enumerate(_sts_prob[0]): #Review row0 first, then each column. row1, column 0 ~ 9...etc.
            if (_prob > _max_prob):
                _max_prob = _prob
                _state = _st
            else:
                pass    
        return _state
    
def state2coordinate(sequence_list, labels_dictionary):
    _seq_list = sequence_list
    _label_dic = labels_dictionary
    _label_seq = []
    
    for _seq in _seq_list:
        _label_seq.extend([_label_dic[_seq+1]]) #starting state from 1 rather than 0.
    
    return _label_seq
'''
    Main program for the HMM class execution.

'''
           
if __name__ == '__main__':
    '''
        Main program.
            Read the observation data from hmm-data.txt.
            Construct HMM with states, observation...etc. data.
            Calculate the sequences of the state.
    '''      
    GRID_WORLD, TOWER_LOCATION, NOISY_DISTANCES = load_parameters('hmm-data.txt')
#    Print_List(GRID_WORLD)
#    print ('TOWER_LOCATION', TOWER_LOCATION)
#    NOISY_DISTANCES.reverse()
    print ('NOISY_DISTANCES', NOISY_DISTANCES)

    STATES, INITIAL_PROB, TRANSITION_MATRIX, VALIDCELL2COORDINATION = getCPT(GRID_WORLD)
    print('STATES,', STATES)
#    print('INITIAL_PROB,', INITIAL_PROB)
#    print('TRANSITION_MATRIX,')
#    Print_List(TRANSITION_MATRIX)    
    print('VALIDCELL2COORDINATION,', VALIDCELL2COORDINATION)

    DISTANCE_MATRIX, OBSERVATIONS = getDistance(TOWER_LOCATION, GRID_WORLD)
    print('OBSERVATIONS=', OBSERVATIONS)
#    print ('DISTANCE_MATRIX for each tower')
#    Print_List(DISTANCE_MATRIX)
    
    hmm = HMM(STATES, OBSERVATIONS, INITIAL_PROB, TRANSITION_MATRIX, DISTANCE_MATRIX, 'viterbi')
    state_sequence = hmm.decode(NOISY_DISTANCES)
    print ('State sequence=', [x+1 for x in state_sequence]) #Because we define the first state counts from 1
    print ('Coordination sequence=', state2coordinate(state_sequence, VALIDCELL2COORDINATION))
 
        
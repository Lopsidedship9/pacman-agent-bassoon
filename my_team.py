# my_team.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import random
import contest.util as util
from contest.capture_agents import CaptureAgent
from contest.game import Directions
from contest.util import nearest_point
from enum import Enum


class Role(Enum):
    OFFENSIVE = "Offensive"
    DEFENSIVE = "Defensive"

class GameStateVariables:
    def __init__(self):
        self.initial_food = 0
        self.actual_food = 0
        self.food_eat = 0
        self.have_pased_pos = (0.0,0.0)
        self.first_iteration_back = False
        self.timer = 0

        

class RoleManager:
    def __init__(self):
        self.roles = {}  # Diccionario para almacenar roles por índice de agente
        self.need_change = {} # Diccionario para almacenar los valores de si se ha de canviar de rol

    def set_roles(self, agent1_index, role1, agent2_index, role2):
        self.roles[agent1_index] = role1
        self.need_change[agent1_index] = False
        self.roles[agent2_index] = role2
        self.need_change[agent2_index] = False


    def get_role(self, agent_index):
        return self.roles.get(agent_index, None)

    def change_role(self, agent_index, agent_instance, n_food_left):
        if self.roles[agent_index] == Role.OFFENSIVE:
            self.roles[agent_index] = Role.DEFENSIVE
            self.need_change[agent_index] = False
            agent_instance.food_obtained(n_food_left)
        else:
            self.roles[agent_index] = Role.OFFENSIVE
            self.need_change[agent_index] = False
    
    def change_signal(self):
        for agent_index in self.need_change:
            self.need_change[agent_index] = True
'''
    def need_change(self, agent_index):
        return self.need_change[agent_index]
'''


#################
# Team creation #
#################

def create_team(first_index, second_index, is_red,
                first='ReflexCaptureAgent', second='ReflexCaptureAgent', num_training=0):
    """
    This function should return a list of two agents that will form the
    team, initialized using firstIndex and secondIndex as their agent
    index numbers.  isRed is True if the red team is being created, and
    will be False if the blue team is being created.

    As a potentially helpful development aid, this function can take
    additional string-valued keyword arguments ("first" and "second" are
    such arguments in the case of this function), which will come from
    the --redOpts and --blueOpts command-line arguments to capture.py.
    For the nightly contest, however, your team will be created without
    any extra arguments, so you should make sure that the default
    behavior is what you want for the nightly contest.
    """

    # The following line is an example only; feel free to change it.
    role_manager = RoleManager()
    game_state_vars = GameStateVariables()
    role_manager.set_roles(first_index, Role.OFFENSIVE, second_index, Role.DEFENSIVE)
    return [
        eval(first)(first_index, game_state_vars, role_manager),
        eval(second)(second_index, game_state_vars, role_manager)
    ]

##########
# Agents #
##########

class ReflexCaptureAgent(CaptureAgent):
    


    def __init__(self, index, game_state_vars, role_manager, time_for_computing=.1):
        super().__init__(index, time_for_computing)
        self.start = None
        self.index = index
        self.role_manager = role_manager  # Shared role manager
        self.game_state_vars = game_state_vars  # Shared game state variables
    

    def register_initial_state(self, game_state):
        self.start = game_state.get_agent_position(self.index)
        CaptureAgent.register_initial_state(self, game_state)
        self.game_state_vars.initial_food = len(self.get_food(game_state).as_list())      
        #print(self.initial_food)
        
        print(self.role_manager.roles[self.index])

    def choose_action(self, game_state):
        """
        Picks among the actions with the highest Q(s,a).
        """
        actions = game_state.get_legal_actions(self.index)
        '''if self.game_state_vars.first_iteration == False:
            self.game_state_vars.first_iteration = True
            successor = self.get_successor(game_state, actions[0])
            food_list = self.get_food_you_are_defending(successor).as_list()
            my_pos = successor.get_agent_state(self.index).get_position()

            if len(food_list) > 0: # This should always be True,  but better safe than sorry
                max_distance = 0
                for food in food_list:
                    aux_distance = self.get_maze_distance(my_pos, food)
                    if max_distance < aux_distance:
                        max_distance = aux_distance
                        save_food = food
                self.game_state_vars.have_pased_pos = save_food'''

        # You can profile your evaluation time by uncommenting these lines
        # start = time.time()food_left
        values = [self.evaluate(game_state, a) for a in actions]
        # print (eval time for agent %d: %.4f' % (self.index, time.time() - start)
        
        max_value = max(values)
        best_actions = [a for a, v in zip(actions, values) if v == max_value]

        food_left = len(self.get_food(game_state).as_list())

        if food_left <= 2:
            best_dist = 9999
            best_action = None
            for action in actions:
                successor = self.get_successor(game_state, action)
                pos2 = successor.get_agent_position(self.index)
                dist = self.get_maze_distance(self.start, pos2)
                if dist < best_dist:
                    best_action = action
                    best_dist = dist
            
            return best_action

        


        return random.choice(best_actions)

    def go_safe_zone(self, game_state, action):
        
        successor = self.get_successor(game_state, action)
        my_pos = successor.get_agent_state(self.index).get_position()
        my_pos = (int(my_pos[0]), int(my_pos[1]))
        x = 0
        walls= game_state.get_walls()
        mid_line = int(walls.width//2)
        if self.red:
            if walls[mid_line][my_pos[1]] == False:
                min_distance = self.get_maze_distance(my_pos, (mid_line, my_pos[1]))
            else:
                for i in range(1, mid_line):
                    if walls[mid_line-i][my_pos[1]] == False:
                        min_distance = self.get_maze_distance(my_pos, (mid_line-i, my_pos[1]))
        else:
            if walls[mid_line+1][my_pos[1]] == False:
                min_distance = self.get_maze_distance(my_pos, (mid_line, my_pos[1]))
            else:
                for i in range(1, mid_line):
                    if walls[mid_line+i+1][my_pos[1]] == False:
                        min_distance = self.get_maze_distance(my_pos, (mid_line+i+1, my_pos[1]))
        
        """if self.game_state_vars.first_iteration_back == False:
            min_distance = float('-inf')
            food_list = self.get_food_you_are_defending(successor).as_list()

            if len(food_list) > 0: # This should always be True,  but better safe than sorry
                for food in food_list:
                    aux_distance = self.get_maze_distance(my_pos, food)
                    if min_distance < aux_distance:
                        min_distance = aux_distance
                        min_food = food
            self.game_state_vars.first_iteration_back = True
            self.game_state_vars.have_pased_pos = min_food
        else:
            min_distance = self.get_maze_distance(my_pos, self.game_state_vars.have_pased_pos)"""
        return min_distance   

    def get_successor(self, game_state, action):
        """
        Finds the next successor which is a grid position (location tuple).
        """
        successor = game_state.generate_successor(self.index, action)
        pos = successor.get_agent_state(self.index).get_position()
        if pos != nearest_point(pos):
            # Only half a grid position was covered
            return successor.generate_successor(self.index, action)
        else:
            return successor

    def evaluate(self, game_state, action):
        """
        Computes a linear combination of features and feature weights
        """
        if(self.role_manager.need_change[self.index] == True): #Aqui a idea que tinc es posar que si hi a algun ghost aprop no canviï de rol.
            self.role_manager.change_role(self.index,self,len(self.get_food(game_state).as_list()))

        if(self.role_manager.get_role(self.index) == Role.OFFENSIVE):
            features = self.get_features_offensive(game_state, action)
            weights = self.get_weights_offensive(game_state, action)
        elif(self.role_manager.get_role(self.index) == Role.DEFENSIVE):
            features = self.get_features_defensive(game_state, action)
            weights = self.get_weights_defensive(game_state, action)
        return features * weights
   
    def manage_food_counter(self, food_left):
        if food_left == self.game_state_vars.initial_food:
            self.game_state_vars.food_eat = 0
            self.game_state_vars.actual_food = self.game_state_vars.initial_food
        elif food_left < self.game_state_vars.actual_food:
            self.game_state_vars.actual_food = food_left
            self.game_state_vars.food_eat += 1
        print("Food_eat: ",self.game_state_vars.food_eat," Food_left: ", food_left, " Food_to_eat: ", self.game_state_vars.actual_food)

    def food_obtained(self, n_food):
        #print('Nfood es: ', n_food)
        self.game_state_vars.initial_food = n_food
        self.game_state_vars.food_eat = 0
    '''
*************************************************************************************************************************
    Here is the defensive agent
********************************************************************************************************************************
    '''
    def get_features_defensive(self, game_state, action):
        features = util.Counter()
        successor = self.get_successor(game_state, action)

        my_state = successor.get_agent_state(self.index)
        my_pos = my_state.get_position()

        # Computes whether we're on defense (1) or offense (0)
        features['on_defense'] = 1
        if my_state.is_pacman: features['on_defense'] = 0
#######################################
        map_width = game_state.get_walls().width  # Width of the grid (map)
        walls = game_state.get_walls()
        midline_x = map_width // 2 

        # Computes distance to invaders we can see
        enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]

        invaders = [a for a in enemies if a.is_pacman and a.get_position() is not None]
        features['num_invaders'] = len(invaders)
        if len(invaders) > 0:
            dists = [self.get_maze_distance(my_pos, a.get_position()) for a in invaders]
            closest_invader_distance = min(dists)
            features['invader_distance'] = min(dists)
            # Check if any invader is within five squares
            if closest_invader_distance <= 5:
                # Reward moving towards the closest invader
                closest_invader = invaders[dists.index(closest_invader_distance)]
                target_pos = closest_invader.get_position()

                # Check if the target is on the enemy side of the map
                if target_pos[0] > midline_x:
                    # If the target is on the enemy's side, prevent crossing the midline
                    if my_pos[0] <= midline_x:
                        # If we're on our side, continue chasing the invader
                        features['chasing_invader'] = self.get_maze_distance(my_pos, target_pos)
                    else:
                        # If we're on the wrong side (past the midline), don't chase
                        features['chasing_invader'] = -float('inf')  # Penalize for crossing the line
                else:
                    # If the invader is still on our side, chase it
                    features['chasing_invader'] = self.get_maze_distance(my_pos, target_pos)

        else:
            # No invaders visible
            features['invader_distance'] = 0
            features['move_towards_invader'] = 0
##########################################
        if action == Directions.STOP: features['stop'] = 1
        rev = Directions.REVERSE[game_state.get_agent_state(self.index).configuration.direction]
        if action == rev: features['reverse'] = 1

        return features
    #added 'move_towards_invader': 50
    def get_weights_defensive(self, game_state, action):
        return {'num_invaders': -1000, 'on_defense': 100, 'invader_distance': -10, 'move_towards_invader': 50, 'stop': -100, 'reverse': -2}
    
    '''
*************************************************************************************************************************
    Here is the Offensive agent
********************************************************************************************************************************
    '''
    
    def get_features_offensive(self, game_state, action):
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        food_list = self.get_food(successor).as_list()
        features['successor_score'] = -len(food_list)  # self.get_score(successor)
        my_pos = successor.get_agent_state(self.index).get_position()
        if my_pos == self.start and self.game_state_vars.timer >= 20:
            self.role_manager.change_signal()
        elif self.game_state_vars.timer < 20:
            self.game_state_vars.timer += 1
        
        # Compute distance to the nearest food      
        if len(food_list) > 0: # This should always be True,  but better safe than sorry
            if(self.game_state_vars.food_eat >= 1):
                min_distance = self.go_safe_zone(game_state, action)
                features['distance_to_food'] = min_distance
                if ( self.game_state_vars.have_pased_pos[0] == my_pos[0]):
                    self.game_state_vars.food_eat = 0
                    #self.role_manager.change_signal()
            else:
                min_distance = min([self.get_maze_distance(my_pos, food) for food in food_list])
                features['distance_to_food'] = min_distance 
        self.manage_food_counter(len(food_list))
        return features

    def get_weights_offensive(self, game_state, action):
        return {'successor_score': 100, 'distance_to_food': -1} 
      
        


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

        if self.game_state_vars.first_iteration_back == False:
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
            min_distance = self.get_maze_distance(my_pos, self.game_state_vars.have_pased_pos)
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
        """
        Extract defensive features for evaluating the current state after a given action.
        """
        features = util.Counter()
        successor = self.get_successor(game_state, action)

        # Current agent's position and state
        my_state = successor.get_agent_state(self.index)
        my_pos = my_state.get_position()

        # Defense state: 1 if on defense, 0 if a pacman
        features['on_defense'] = 1 if not my_state.is_pacman else 0

        # Get visible enemies who are Pacmen (invaders)
        enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
        invaders = [enemy for enemy in enemies if enemy.is_pacman and enemy.get_position() is not None]
        features['num_invaders'] = len(invaders)

        # Handle visible invaders
        if len(invaders) > 0:
            dists = [self.get_maze_distance(my_pos, invader.get_position()) for invader in invaders]
            features['invader_distance'] = min(dists)
        else:
            # Handle unseen invaders using noisy distances
            noisy_distances = successor.get_agent_distances()
            estimated_positions = self.estimate_opponent_positions(game_state, noisy_distances)
            if estimated_positions:
                dists = [self.get_maze_distance(my_pos, pos) for pos in estimated_positions]
                features['invader_distance'] = min(dists)

        # Penalize stopping
        if action == Directions.STOP:
            features['stop'] = 1

        # Penalize reversing direction
        reverse_dir = Directions.REVERSE[game_state.get_agent_state(self.index).configuration.direction]
        if action == reverse_dir:
            features['reverse'] = 1

        return features


    def estimate_opponent_positions(self, game_state, noisy_distances):
        """
        Estimate possible positions of opponents using noisy distance readings.
        """
        possible_positions = []
        for opponent_idx in self.get_opponents(game_state):
            noisy_distance = noisy_distances[opponent_idx]
            previous_beliefs = self.get_previous_beliefs(opponent_idx)

            # Iterate through potential positions in the belief distribution
            for pos, prob in previous_beliefs.items():
                if prob > 0 and abs(self.get_maze_distance(pos, self.get_position())) == noisy_distance:
                    possible_positions.append(pos)

        # Optionally, display beliefs for debugging
        self.display_distributions_over_positions([self.get_previous_beliefs(idx) for idx in self.get_opponents(game_state)])

        return possible_positions


    def get_previous_beliefs(self, opponent_idx):
        """
        Retrieve or initialize belief distributions for an opponent.
        """
        if not hasattr(self, 'beliefs'):
            self.beliefs = {opponent_idx: util.Counter() for opponent_idx in self.get_opponents(self.get_current_observation())}

        if len(self.observation_history) < 2:
            return self.beliefs[opponent_idx]

        previous_state = self.get_previous_observation()
        current_state = self.get_current_observation()
        return self.update_beliefs(previous_state, current_state, opponent_idx)


    def update_beliefs(self, prev_state, curr_state, opponent_idx):
        """
        Update belief distributions based on noisy readings and observations.
        """
        beliefs = self.beliefs[opponent_idx]
        noisy_distance = curr_state.get_agent_distances()[opponent_idx]

        # Update the belief distribution using Manhattan distance and game rules
        all_positions = [(x, y) for x in range(curr_state.data.layout.width) for y in range(curr_state.data.layout.height)]
        new_beliefs = util.Counter()
        for pos in all_positions:
            if noisy_distance == self.get_maze_distance(pos, self.get_position()):
                new_beliefs[pos] = 1.0

        # Normalize beliefs
        new_beliefs.normalize()
        self.beliefs[opponent_idx] = new_beliefs
        return new_beliefs

    
    def get_weights_defensive(self, game_state, action):
        return {'num_invaders': -1000, 'on_defense': 100, 'invader_distance': -10, 'stop': -100, 'reverse': -2}
    
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
      
        


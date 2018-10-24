import numpy as np
from itertools import combinations
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from math import ceil
from collections import Counter
from agents import *


class Game():
    def __init__(self, row_player=None, col_player=None, row_payoff_matrix = [[3, 1], [5, 2]], col_payoff_matrix = [[3, 5], [1, 2]], row_name=None, col_name=None):
        """defaults to prisoner's dilemma"""
        self.row_payoff_matrix = row_payoff_matrix
        self.col_payoff_matrix = col_payoff_matrix

        self.row_player = row_player
        self.col_player = col_player

        self.row_payoff = []
        self.col_payoff = []

        self.move_history = []

        self.row_name = row_name #currently unused
        self.col_name = col_name

    def step(self):
        row_move = self.row_player.move()
        col_move = self.col_player.move()

        if type(self.row_player) == TitforTat or type(self.row_player) == TitforTwoTats or type(self.row_player) == NeverForgive or type(self.row_player) == WinStayLoseShift:
            self.row_player.update_previous_move(col_move)
        elif type(self.row_player) == PavlovAgent:
            self.row_player.update_previous_move(row_move, col_move)

        if type(self.col_player) == TitforTat or type(self.col_player) == TitforTwoTats or type(self.col_player) == NeverForgive or type(self.col_player) == WinStayLoseShift:
            self.col_player.update_previous_move(row_move)
        elif type(self.col_player) == PavlovAgent:
            self.col_player.update_previous_move(row_move, col_move)


        self.move_history.append([row_move, col_move])
        row_temp_payoff = self.row_payoff_matrix[row_move][col_move]
        col_temp_payoff = self.col_payoff_matrix[row_move][col_move]



        self.row_payoff.append(row_temp_payoff)
        self.col_payoff.append(col_temp_payoff)


    def set_players(self, row_player, col_player):
        self.row_player = row_player
        self.col_player = col_player

    def get_row_payoff(self):
        return np.sum(self.row_payoff)
    
    def get_col_payoff(self):
        return np.sum(self.col_payoff)

def get_agent_name(agent_object):
    t = type(agent_object)
    if t == AlwaysDefect:
        return "AlwaysDefect"
    elif t == AlwaysCoorperate: 
        return "AlwaysCoorperate"
    elif t == TitforTat: 
        return "TitforTat"
    elif t == NotTitforTat: 
        return "NotTitforTat"
    else:
        return None

def main():
    filename = "data{}.csv".format(np.random.randint(1,10000))
    agents = [
        ["AlwaysDefect", AlwaysDefect],
        ["AlwaysCoorperate", AlwaysCoorperate],
        ["TitforTat", TitforTat],
        ["NotTitforTat", NotTitforTat]
    ]
    n_agents = 900
    gammas = [0.95, 0.99]
    lr = 0.5
    n_steps = 10
    initial_population_distribution = [
        [0.25, 0.25, 0.25, 0.25],
        [0.40, 0.40, 0.10, 0.10],
        [0.10, 0.10, 0.40, 0.40],
        [0.40, 0.10, 0.40, 0.10],
        [0.10, 0.40, 0.10, 0.40]

    ]


    prisdel_row_payoff_matrix = [[3, 1], [5, 2]]
    prisdel_col_payoff_matrix = [[3, 5], [1, 2]]

    staghunt_row_payoff_matrix = [[4, 0], [2, 2]]
    staghunt_col_payoff_matrix = [[4, 2], [0, 2]]

    battle_row_payoff_matrix = [[0, 1], [2, 0]]
    battle_col_payoff_matrix = [[0, 2], [1, 0]]

    games = [
        ["Prisoner's Dilemma", [prisdel_row_payoff_matrix, prisdel_col_payoff_matrix]],
        ["Stag Hunt", [staghunt_row_payoff_matrix, staghunt_col_payoff_matrix]],
        ["Battle of the Sexes", [battle_row_payoff_matrix, battle_col_payoff_matrix]]
    ]

    # crap code, sorry
    agent1 = agents[0][1]
    agent2 = agents[1][1]
    agent3 = agents[2][1]
    agent4 = agents[3][1]
    
    # agent_pairs = list(combinations(agents, 2))
    payoffs = []
    print("replicator dynamics ")
    
    #replicator dynamics with random pairings, yes these should be functions, but they're not
    for game_name, game in games:
        print(game_name)
        row_payoff_matrix = game[0]
        col_payoff_matrix = game[1]
        for gamma in gammas:
            for theta1, theta2, theta3, theta4 in initial_population_distribution:
                payoff = {"AlwaysDefect":0, "AlwaysCoorperate": 0, "TitforTat": 0, "NotTitforTat": 0 }
                initial_thetas = [theta1, theta2, theta3, theta4]
                # play through to a steady state
                n = 0
                theta_progress = []
                for _ in tqdm(range(n_steps)):
                    n += 1
                    # calculate phi for each agent
                    n_agent1 = int(theta1*n_agents)
                    n_agent2 = int(theta2*n_agents)
                    n_agent3 = int(theta3*n_agents)
                    n_agent4 = int(theta4*n_agents)
                    theta_progress.append([theta1, theta2, theta3, theta4])

                    # log, initialize, and shuffle agent space
                    current_agents_generation = [agent1 for x in range(n_agent1)]+[agent2 for x in range(n_agent2)]+[agent3 for x in range(n_agent3)]+[agent4 for x in range(n_agent4)]
                    np.random.shuffle(current_agents_generation) # occurs in place

                    # play random pairs, works because of shuffle, every agent randomly plays another
                    for i in range(0, len(current_agents_generation)-1):
                        row_player = current_agents_generation[i]()
                        col_player = current_agents_generation[i+1]()
                        row_name = get_agent_name(row_player)
                        col_name = get_agent_name(col_player)
                        game_obj = Game(row_player, col_player, row_payoff_matrix, col_payoff_matrix)
                        it = 0
                        #play game between two agents
                        while True:
                            it += 1
                            game_obj.step()
                            if np.random.random() > gamma:
                                # game is over between two agents
                                break
                        payoff[row_name] += game_obj.get_row_payoff()
                        # payoff[col_name] += game_obj.get_col_payoff()
                        # payoff[row_name] /= it
                        # payoff[col_name] /= it
                    
                    # replicator dynamics for fitness of next generation
                    payoff["AlwaysDefect"] /= n_agent1
                    payoff["AlwaysCoorperate"] /= n_agent2
                    payoff["TitforTat"] /= n_agent3
                    payoff["NotTitforTat"] /= n_agent4
                    total_average_payoff = (payoff["AlwaysDefect"] + payoff["AlwaysCoorperate"] + payoff["TitforTat"] + payoff["NotTitforTat"])/4 # u*

                    # change in population = current_percentage * (average_payoff - total_average_payoff)
                    theta1_prime = theta1 * (payoff["AlwaysDefect"] - total_average_payoff)
                    theta2_prime = theta2 * (payoff["AlwaysCoorperate"] - total_average_payoff)
                    theta3_prime = theta3 * (payoff["TitforTat"] - total_average_payoff)
                    theta4_prime = theta4 * (payoff["NotTitforTat"] - total_average_payoff)

                    theta1 = theta1 + lr * theta1_prime
                    theta2 = theta2 + lr * theta2_prime
                    theta3 = theta3 + lr * theta3_prime
                    theta4 = theta4 + lr * theta4_prime

                
                theta_progress = np.array(theta_progress)
                # record data here
                payoffs.append({
                    "game": game_name,
                    "agent1": list(theta_progress[:,0]),
                    "agent2": list(theta_progress[:,1]),
                    "agent3": list(theta_progress[:,2]),
                    "agent4": list(theta_progress[:,3]),
                    "initial_distribution": initial_thetas,
                    "generations_until_stability": n,
                    "gamma": gamma,
                    "dynamics": "replicator"
                    })



    # print("imitator dynamics")
    # cardinal_directions = lambda row, col, row_size, col_size: [[(row-1)%row_size, col],[(row+1)%row_size, col],[row, (col-1)%col_size],[row, (col+1)%col_size],[(row-1)%row_size, (col+1)%col_size],[(row-1)%row_size, (col-1)%col_size],[(row+1)%row_size, (col+1)%col_size],[(row+1)%row_size, (col-1)%col_size]]
    # #imitator dynamics with local lattice, again, not a function
    # for game_name, game in games:
    #     print(game_name)
    #     row_payoff_matrix = game[0]
    #     col_payoff_matrix = game[1]
    #     for gamma in gammas:
    #         for theta1, theta2, theta3, theta4 in initial_population_distribution:
    #             payoff = {"AlwaysDefect":0, "AlwaysCoorperate": 0, "TitforTat": 0, "NotTitforTat": 0 }
    #             initial_thetas = [theta1, theta2, theta3, theta4]

    #             # play through to a steady state
    #             theta_progress = []

    #             # calculate phi for each agent
    #             n_agent1 = int(theta1*n_agents)
    #             n_agent2 = int(theta2*n_agents)
    #             n_agent3 = int(theta3*n_agents)
    #             n_agent4 = int(theta4*n_agents)
    #             theta_progress.append([theta1, theta2, theta3, theta4])

    #             # log, initialize, and shuffle agent space
    #             current_agents_generation = [agent1 for x in range(n_agent1)]+[agent2 for x in range(n_agent2)]+[agent3 for x in range(n_agent3)]+[agent4 for x in range(n_agent4)]
    #             np.random.shuffle(current_agents_generation) # occurs in place, good randomization
    #             lattice = np.reshape(current_agents_generation, (30,30)) # this is now our lattice
    #             n = 0
    #             for _ in tqdm(range(n_steps)):
    #                 n += 1
    #                 scores = np.zeros_like(lattice) # used for imitator selection
    #                 row_size, col_size = lattice.shape
    #                 for row in range(row_size):
    #                     for col in range(col_size):
    #                         curr_payoff = 0
    #                         for direction in cardinal_directions(row, col, row_size, col_size):
    #                             row_player = lattice[row][col]()
    #                             col_player = lattice[direction[0]][direction[1]]()
    #                             row_name = get_agent_name(row_player)
    #                             col_name = get_agent_name(col_player)
    #                             game_obj = Game(row_player, col_player, row_payoff_matrix, col_payoff_matrix)

    #                             it = 0
    #                             #play game between two agents
    #                             while True: 
    #                                 it += 1
    #                                 game_obj.step()
    #                                 if np.random.random() > gamma:
    #                                     # game is over between two agents
    #                                     break
    #                             curr_payoff += game_obj.get_row_payoff()
    #                             curr_payoff /= it
    #                         scores[row][col] = curr_payoff

    #                 # imitator dynamics for fitness of next generation
    #                 new_lattice = lattice.copy()
    #                 for row in range(row_size):
    #                     for col in range(col_size):
    #                         fitest_neighbor = np.argmax([scores[row][col],
    #                             scores[(row-1)%row_size][col],
    #                             scores[(row+1)%row_size][col],
    #                             scores[row][(col-1)%col_size],
    #                             scores[row][(col+1)%col_size],
    #                             scores[(row-1)%row_size][(col+1)%col_size],
    #                             scores[(row-1)%row_size][(col-1)%col_size],
    #                             scores[(row+1)%row_size][(col+1)%col_size],
    #                             scores[(row+1)%row_size][(col-1)%col_size]])

    #                         direction = [[row,col]]+cardinal_directions(row, col, row_size, col_size)
    #                         direction = direction[fitest_neighbor]
                            
    #                         new_lattice[row][col] = lattice[direction[0]][direction[1]]

    #                 lattice = new_lattice.copy()

    #                 c = Counter(np.reshape(lattice, (-1)))
    #                 n_agent1 = c[agent1]
    #                 n_agent2 = c[agent2]
    #                 n_agent3 = c[agent3]
    #                 n_agent4 = c[agent4]

    #                 theta1 = n_agent1/(n_agent1 + n_agent2 + n_agent3 + n_agent4)
    #                 theta2 = n_agent2/(n_agent1 + n_agent2 + n_agent3 + n_agent4)
    #                 theta3 = n_agent3/(n_agent1 + n_agent2 + n_agent3 + n_agent4)
    #                 theta4 = n_agent4/(n_agent1 + n_agent2 + n_agent3 + n_agent4)

    #                 theta_progress.append([theta1, theta2, theta3, theta4])

                    
    #             theta_progress = np.array(theta_progress)
    #             # record data here
    #             payoffs.append({
    #                 "game": game_name,
    #                 "agent1": list(theta_progress[:,0]),
    #                 "agent2": list(theta_progress[:,1]),
    #                 "agent3": list(theta_progress[:,2]),
    #                 "agent4": list(theta_progress[:,3]),
    #                 "initial_distribution": initial_thetas,
    #                 "generations_until_stability": n,
    #                 "gamma": gamma,
    #                 "dynamics": "imitator"
    #                 })

    df = pd.DataFrame.from_dict(payoffs)
    df.to_csv(filename, index=False)

if __name__ == "__main__":
    main()

import numpy as np
from itertools import combinations
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from math import ceil
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
    agents = [
        ["AlwaysDefect", AlwaysDefect],
        ["AlwaysCoorperate", AlwaysCoorperate],
        ["TitforTat", TitforTat],
        ["NotTitforTat", NotTitforTat]
    ]
    n_agents = 900
    gammas = [0.95, 0.99]
    initial_population_distribution = [
        [0.5, 0.5],
        [0.25, 0.75],
        [0.75, 0.25],
        [0.1, 0.9],
        [0.9, 0.1]
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
    
    agent_pairs = list(combinations(agents, 2))

    print("replicator dynamics ")
    payoffs = []
    #replicator dynamics with random pairings, yes these should be functions, but they're not
    # for game_name, game in games:
    #     print(game_name)
    #     row_payoff_matrix = game[0]
    #     col_payoff_matrix = game[1]
    #     for (agent1_name, agent1), (agent2_name, agent2)  in agent_pairs:
    #         print("{} vs {}".format(agent1_name, agent2_name))
    #         for gamma in gammas:
    #             for theta1, theta2 in initial_population_distribution:
    #                 payoff = {agent1_name:0, agent2_name: 0}
    #                 initial_theta1, initial_theta2 = theta1, theta2

    #                 # play through to a steady state
    #                 n = 0
    #                 population_progress = []
    #                 theta_progress = []
    #                 while True:
    #                     n += 1
    #                      # calculate phi for each agent
    #                     n_agent1 = int(theta1*n_agents)
    #                     n_agent2 = int(theta2*n_agents)
    #                     population_progress.append([n_agent1, n_agent2])
    #                     theta_progress.append([theta1, theta2])

    #                     # log, initialize, and shuffle agent space
    #                     # print("\t{}: {}\n\t{}: {}".format(n_agent1, agent1_name, n_agent2, agent2_name))
    #                     current_agents_generation = [agent1 for x in range(n_agent1)]+[agent2 for x in range(n_agent2)]
    #                     np.random.shuffle(current_agents_generation) # occurs in place

    #                     # play random pairs, works because of shuffle, every agent randomly plays another
    #                     for i in range(0, len(current_agents_generation)-2, 2):
    #                         row_player = current_agents_generation[i]()
    #                         col_player = current_agents_generation[i+1]()
    #                         row_name = get_agent_name(row_player)
    #                         col_name = get_agent_name(col_player)
    #                         game_obj = Game(row_player, col_player, row_payoff_matrix, col_payoff_matrix)
    #                         it =0
    #                         #play game between two agents
    #                         while True: 
    #                             it += 1
    #                             game_obj.step()
    #                             if np.random.random() > gamma:
    #                                 # game is over between two agents
    #                                 break
    #                         payoff[row_name] += game_obj.get_row_payoff()
    #                         payoff[col_name] += game_obj.get_col_payoff()
    #                         payoff[row_name] /= it
    #                         payoff[col_name] /= it
                        
    #                     # replicator dynamics for fitness of next generation
    #                     payoff[agent1_name] /= n_agent1
    #                     payoff[agent2_name] /= n_agent2
    #                     total_average_payoff = (payoff[agent1_name] + payoff[agent2_name])/2 # u*

    #                     # change in population = current_percentage * (average_payoff - total_average_payoff)
    #                     theta1_prime = theta1 * (payoff[agent1_name] - total_average_payoff)
    #                     theta2_prime = theta2 * (payoff[agent2_name] - total_average_payoff)

    #                     prev1, prev2 = theta1, theta2
    #                     theta1 = theta1 + theta1_prime
    #                     theta2 = theta2 + theta2_prime
    #                     if abs(prev1 - theta1) < 10e-6 and abs(prev2 - theta2) < 10e-6:
    #                         break

                       
    #                 # record data here
    #                 payoffs.append({
    #                     "game": game_name,
    #                     "agent1": agent1_name,
    #                     "agent2": agent2_name,
    #                     "initial_theta1": initial_theta1,
    #                     "initial_theta2": initial_theta2,
    #                     "generations_until_stability": n,
    #                     "gamma": gamma,
    #                     "final_theta1": theta1,
    #                     "final_theta2": theta2,
    #                     "population_progress": population_progress,
    #                     "theta_progress": theta_progress
    #                     })
    
    # df = pd.DataFrame.from_dict(payoffs)
    # df.to_csv("replicator_data.csv", index=False)


    print("imitator dynamics")
    payoffs = []
    #imitator dynamics with local lattice, again, not a function
    for game_name, game in games:
        print(game_name)
        row_payoff_matrix = game[0]
        col_payoff_matrix = game[1]
        for (agent1_name, agent1), (agent2_name, agent2)  in agent_pairs:
            print("{} vs {}".format(agent1_name, agent2_name))
            for gamma in gammas:
                for theta1, theta2 in initial_population_distribution:
                    payoff = {agent1_name:0, agent2_name: 0}
                    initial_theta1, initial_theta2 = theta1, theta2

                    # play through to a steady state
                    n = 0
                    population_progress = []
                    theta_progress = []
                    while True:
                        n += 1
                         # calculate phi for each agent
                        n_agent1 = ceil(theta1*n_agents)
                        n_agent2 = ceil(theta2*n_agents)
                        normalizing_difference = n_agents - (n_agent1 + n_agent2)

                        if np.random.random() > 0.5:
                            n_agent1 += normalizing_difference
                        else:
                            n_agent2 += normalizing_difference

                        population_progress.append([n_agent1, n_agent2])
                        theta_progress.append([theta1, theta2])

                        # log, initialize, and shuffle agent space
                        # print("\t{}: {}\n\t{}: {}".format(n_agent1, agent1_name, n_agent2, agent2_name))
                        current_agents_generation = np.array([agent1 for x in range(n_agent1)]+[agent2 for x in range(n_agent2)])
                        np.random.shuffle(current_agents_generation) # occurs in place, good randomization
                        lattice = np.reshape(current_agents_generation, (-1, 2)) # this is now our lattice
                        scores = np.zeros_like(current_agents_generation) # used for imitator selection
                        row_size, col_size = lattice.shape
                        for row in range(row_size):
                            for col in range(col_size):
                                for direction in [[(row-1)%row_size, col],[(row+1)%row_size, col],[row, (col-1)%col_size],[row, (col+1)%col_size],[(row-1)%row_size, (col+1)%col_size],[(row-1)%row_size, (col-1)%col_size],[(row+1)%row_size, (col+1)%col_size],[(row+1)%row_size, (col-1)%col_size]]:
                                    row_player = lattice[row][col]()
                                    col_player = lattice[direction[0]][direction[1]]()
                                    row_name = get_agent_name(row_player)
                                    col_name = get_agent_name(col_player)
                                    game_obj = Game(row_player, col_player, row_payoff_matrix, col_payoff_matrix)
                                    it =0
                                    #play game between two agents
                                    while True: 
                                        it += 1
                                        game_obj.step()
                                        if np.random.random() > gamma:
                                            # game is over between two agents
                                            break
                                    payoff[row_name] += game_obj.get_row_payoff()
                                    payoff[col_name] += game_obj.get_col_payoff()
                                    payoff[row_name] /= it
                                    payoff[col_name] /= it
                        
                        # replicator dynamics for fitness of next generation
                        payoff[agent1_name] /= n_agent1
                        payoff[agent2_name] /= n_agent2
                        total_average_payoff = (payoff[agent1_name] + payoff[agent2_name])/2 # u*

                        # change in population = current_percentage * (average_payoff - total_average_payoff)
                        theta1_prime = theta1 * (payoff[agent1_name] - total_average_payoff)
                        theta2_prime = theta2 * (payoff[agent2_name] - total_average_payoff)

                        prev1, prev2 = theta1, theta2
                        theta1 = theta1 + theta1_prime
                        theta2 = theta2 + theta2_prime
                        if abs(prev1 - theta1) < 10e-6 and abs(prev2 - theta2) < 10e-6:
                            break

                       
                    # record data here
                    payoffs.append({
                        "game": game_name,
                        "agent1": agent1_name,
                        "agent2": agent2_name,
                        "initial_theta1": initial_theta1,
                        "initial_theta2": initial_theta2,
                        "generations_until_stability": n,
                        "gamma": gamma,
                        "final_theta1": theta1,
                        "final_theta2": theta2,
                        "population_progress": population_progress,
                        "theta_progress": theta_progress
                        })

    df = pd.DataFrame.from_dict(payoffs)
    df.to_csv("imitator_data.csv", index=False)

if __name__ == "__main__":
    main()

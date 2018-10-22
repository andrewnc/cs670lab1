import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from agents import *


class Game():
    def __init__(self, row_player, col_player, row_payoff_matrix = [[3, 1], [5, 2]], col_payoff_matrix = [[3, 5], [1, 2]]):
        """defaults to prisoner's dilemma"""
        self.row_payoff_matrix = row_payoff_matrix
        self.col_payoff_matrix = col_payoff_matrix

        self.row_player = row_player
        self.col_player = col_player

        self.row_payoff = []
        self.col_payoff = []

        self.move_history = []

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

    def get_row_payoff(self):
        return np.sum(self.row_payoff)
    
    def get_col_payoff(self):
        return np.sum(self.col_payoff)

def main():
    agents = [
        ["AlwaysDefect", AlwaysDefect],
        ["AlwaysCoorperate", AlwaysCoorperate],
        ["TitforTat", TitforTat],
        ["NeverForgive", NeverForgive]
    ]
    gammas = [0.95, 0.99]
    payoffs = []

    prisdel_row_payoff_matrix = [[3, 1], [5, 2]]
    prisdel_col_payoff_matrix = [[3, 5], [1, 2]]

    staghunt_row_payoff_matrix = [[4, 0], [2, 2]]
    staghunt_col_payoff_matrix = [[4, 2], [0, 2]]

    battle_row_payoff_matrix = [[0, 1], [2, 0]]
    battle_col_payoff_matrix = [[0, 2], [1, 0]]

    games = [
        ["Prisoner's Dilemma", prisdel_row_payoff_matrix, prisdel_col_payoff_matrix],
        ["Stag Hunt", staghunt_row_payoff_matrix, staghunt_col_payoff_matrix],
        ["Battle of the Sexes", battle_row_payoff_matrix, battle_col_payoff_matrix]
    ]

    # build lattice here

    for agent1_name, agent1 in agents:
        for agent2_name, agent2 in agents:
            for gamma in gammas:
                n = 0
                row_player = agent1()
                col_player = agent2()
                game = Game(row_player, col_player)
                while True:
                    n += 1
                    game.step()
                    if np.random.random() > gamma:
                        break
                payoffs.append({"prob": gamma, "agent1": agent1_name, "agent2": agent2_name, "n_plays": n, "agent1_payoff": game.get_row_payoff(), "agent2_payoff": game.get_col_payoff()})

    # plt.plot(np.array(payoffs)[:,0])
    # plt.show()
    df = pd.DataFrame.from_dict(payoffs)
    df.to_csv("data.csv", index=False)

if __name__ == "__main__":
    main()

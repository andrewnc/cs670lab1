import numpy as np
from collections import deque
import matplotlib.pyplot as plt
import pandas as pd

class AlwaysDefect():
    def __init__(self):
        pass
    
    def move(self):
        return 1

class AlwaysCoorperate():
    def __init__(self):
        pass
    
    def move(self):
        return 0

class RandomAgent():
    def __init__(self):
        pass

    def move(self):
        return np.random.choice([0,1])

class TitforTat():
    def __init__(self):
        self.previous_move = None
    
    def move(self):
        if self.previous_move is None:
            return 0
        elif self.previous_move == 0:
            return 0
        elif self.previous_move == 1:
            return 1

    def update_previous_move(self, move):
        self.previous_move = move

class TitforTwoTats():
    def __init__(self):
        self.past_two_moves = deque([0,0], maxlen=2)
    
    def move(self):
        if np.sum(self.past_two_moves) == 2:
            return 1
        else:
            return 0

    def update_previous_move(self, move):
        self.past_two_moves.append(move)

class PavlovAgent():
    def __init__(self):
        self.agreed = True
    
    def move(self):
        if self.agreed:
            return 0
        else:
            return 1

    def update_previous_move(self, move1, move2):
        if move1 == move2:
            self.agreed = True
        else:
            self.agreed = False
    
class WinStayLoseShift():
    def __init__(self):
        self.move_to_play = 0

    def move(self):
        return self.move_to_play

    def update_previous_move(self, move):
        if move == 1:
            self.move_to_play = 1 - self.move_to_play
        else:
            self.move_to_play = self.move_to_play

class NeverForgive():
    def __init__(self):
        self.angry = False

    def move(self):
        if self.angry:
            return 1
        else:
            return 0

    def update_previous_move(self, move):
        if move:
            self.angry = True

class MyAgent():
    def __init__(self):
        self.next_move = 0
    
    def move(self):
        self.next_move = 1 - self.next_move
        return self.next_move

class Game():
    def __init__(self, row_player, col_player):
        self.row_payoff_matrix = [[3, 1], [5, 2]]
        self.col_payoff_matrix = [[3, 5], [1, 2]]

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
    agents = [["AlwaysDefect", AlwaysDefect], ["RandomAgent", RandomAgent], ["AlwaysCoorperate", AlwaysCoorperate], ["TitforTat", TitforTat], ["TitforTwoTats", TitforTwoTats], ["PavlovAgent", PavlovAgent], ["WinStayLoseShift", WinStayLoseShift], ["NeverForgive", NeverForgive], ["MyAgent", MyAgent]]
    n_values = [5, 100, 200]
    payoffs = []
    for agent1_name, agent1 in agents:
        for agent2_name, agent2 in agents:
            for n in n_values:
                row_player = agent1()
                col_player = agent2()
                game = Game(row_player, col_player)
                for i in range(n):
                    game.step()
                payoffs.append({"agent1": agent1_name, "agent2": agent2_name, "n_plays": n, "agent1_payoff": game.get_row_payoff(), "agent2_payoff": game.get_col_payoff()})
    # plt.plot(np.array(payoffs)[:,0])
    # plt.show()
    df = pd.DataFrame.from_dict(payoffs)
    with open("results.html", 'w') as f:
        f.write(df.to_html())

    df.to_csv("data.csv", index=False)

def main_prob():
    agents = [["AlwaysDefect", AlwaysDefect], ["RandomAgent", RandomAgent], ["AlwaysCoorperate", AlwaysCoorperate], ["TitforTat", TitforTat], ["TitforTwoTats", TitforTwoTats], ["PavlovAgent", PavlovAgent], ["WinStayLoseShift", WinStayLoseShift], ["NeverForgive", NeverForgive], ["MyAgent", MyAgent]]
    probs = [0.75, 0.9, 0.99]
    payoffs = []
    for agent1_name, agent1 in agents:
        for agent2_name, agent2 in agents:
            for p in probs:
                n = 0
                row_player = agent1()
                col_player = agent2()
                game = Game(row_player, col_player)
                while True:
                    n += 1
                    game.step()
                    if np.random.random() > p:
                        break
                payoffs.append({"prob": p, "agent1": agent1_name, "agent2": agent2_name, "n_plays": n, "agent1_payoff": game.get_row_payoff(), "agent2_payoff": game.get_col_payoff()})

    # plt.plot(np.array(payoffs)[:,0])
    # plt.show()
    df = pd.DataFrame.from_dict(payoffs)
    with open("prob_results.html", 'w') as f:
        f.write(df.to_html())

    df.to_csv("prob_data.csv", index=False)

if __name__ == "__main__":
    main_prob()

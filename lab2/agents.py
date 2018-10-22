import numpy as np
from collections import deque

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


class NotTitforTat():
    def __init__(self):
        self.previous_move = None
    
    def move(self):
        if self.previous_move is None:
            return 1
        elif self.previous_move == 0:
            return 1
        elif self.previous_move == 1:
            return 0

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
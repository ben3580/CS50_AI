"""
Tic Tac Toe Player
Shuyan Liu
CS50's Introduction to AI 2020
"""
import copy
import random

X = "X"
O = "O"
EMPTY = ""

def initial_state():
    """
    Returns starting state of the board.
    """
    return [[EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY]]


def player(board):
    """
    Returns player who has the next turn on a board.
    """
    XSpots = 0
    OSpots = 0
    # Count the number of moves made
    for i in range(3):
        for j in range(3):
            if board[i][j] == X:
                XSpots += 1
            elif board[i][j] == O:
                OSpots += 1
    # X always goes first, so it is a X move if the number of X and O moves are equal
    if XSpots == OSpots:
        return X
    else:
        return O


def actions(board):
    """
    Returns set of all possible actions (i, j) available on the board.
    """
    allActions = set()
    for i in range(3):
        for j in range(3):
            # You can choose any empty space on the board
            if board[i][j] == EMPTY:
                allActions.add((i, j))
    return allActions


def result(board, action):
    """
    Returns the board that results from making move (i, j) on the board.
    """
    # Move cannot be made on a space that already has a X or O
    if board[action[0]][action[1]] != EMPTY:
        raise ValueError("Invalid action")
    # Create a copy
    newBoard = copy.deepcopy(board)
    # Find which player is making the move
    p = player(newBoard)
    # Update the board
    newBoard[action[0]][action[1]] = p
    return newBoard


def winner(board):
    """
    Returns the winner of the game, if there is one.
    """
    isWinner = None
    if board[0][0] == X and board[0][1] == X and board[0][2] == X:
        isWinner = X
    elif board[1][0] == X and board[1][1] == X and board[1][2] == X:
        isWinner = X
    elif board[2][0] == X and board[2][1] == X and board[2][2] == X:
        isWinner = X
    elif board[0][0] == X and board[1][0] == X and board[2][0] == X:
        isWinner = X
    elif board[0][1] == X and board[1][1] == X and board[2][1] == X:
        isWinner = X
    elif board[0][2] == X and board[1][2] == X and board[2][2] == X:
        isWinner = X
    elif board[0][0] == X and board[1][1] == X and board[2][2] == X:
        isWinner = X
    elif board[0][2] == X and board[1][1] == X and board[2][0] == X:
        isWinner = X
    elif board[0][0] == O and board[0][1] == O and board[0][2] == O:
        isWinner = O
    elif board[1][0] == O and board[1][1] == O and board[1][2] == O:
        isWinner = O
    elif board[2][0] == O and board[2][1] == O and board[2][2] == O:
        isWinner = O
    elif board[0][0] == O and board[1][0] == O and board[2][0] == O:
        isWinner = O
    elif board[0][1] == O and board[1][1] == O and board[2][1] == O:
        isWinner = O
    elif board[0][2] == O and board[1][2] == O and board[2][2] == O:
        isWinner = O
    elif board[0][0] == O and board[1][1] == O and board[2][2] == O:
        isWinner = O
    elif board[0][2] == O and board[1][1] == O and board[2][0] == O:
        isWinner = O
    return isWinner


def terminal(board):
    """
    Returns True if game is over, False otherwise.
    """
    total = 0
    # Find the number of moves made
    for i in range(3):
        for j in range(3):
            if board[i][j] == O or board[i][j] == X:
                total += 1
    # A game cannot end in under 5 turns
    if total < 5:
        return False
    # A game with the board fill in is complete
    elif total == 9:
        return True
    # Otherwise check is there is a winner
    else:
        isWinner = winner(board)
        if isWinner == None:
            return False
        else:
            return True


def utility(board):
    """
    Returns 1 if X has won the game, -1 if O has won, 0 otherwise.
    """
    isWinner = winner(board)
    if isWinner == X:
        return 1
    elif isWinner == O:
        return -1
    else:
        return 0


def minimax(board):
    """
    Returns the optimal action for the current player on the board.
    """
    p = player(board)
    # X is the max player and O is the min player
    if p == X:
        optimalActions = maxAction(board)
    else:
        optimalActions = minAction(board)
    # Choose a action randomly from the list
    action = optimalActions[random.randrange(0, len(optimalActions))]
    return action


def maxAction(board):
    """
    Returns a list of all optimal actions for the X player
    """
    value = -1 # Value cannot be lower than -1 or higher than 1
    optimalActions = []
    for action in actions(board):
        # Calls the recursive minimax function
        tempValue = minValue(result(board, action), value)
        # If the value is the same, add it to the list of best actions
        if tempValue == value:
            optimalActions.append(action)
        # If the value is higher, start a new list of best actions
        if tempValue > value:
            optimalActions.clear()
            optimalActions.append(action)
            value = tempValue
    return optimalActions


def minAction(board):
    """
    Returns a list of all optimal actions for the O player
    """
    value = 1
    optimalActions = []
    for action in actions(board):
        tempValue = maxValue(result(board, action), value)
        if tempValue == value:
            optimalActions.append(action)
        if tempValue < value:
            optimalActions.clear()
            optimalActions.append(action)
            value = tempValue
    return optimalActions

    
def maxValue(board, previous):
    """
    Returns the maximum value (most X wins) of a board and its possible actions
    """
    if terminal(board):
        return utility(board)
    value = -1
    for action in actions(board):
        value = max(value, minValue(result(board, action), value))
        # Alpha-Beta pruning
        if previous < value:
            return value
    return value


def minValue(board, previous):
    """
    Returns the minimum value (most O wins) of a board and its possible actions
    """
    if terminal(board):
        return utility(board)
    value = 1
    for action in actions(board):
        value = min(value, maxValue(result(board, action), value))
        if previous > value:
            return value
    return value


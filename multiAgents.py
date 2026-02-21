# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        score = successorGameState.getScore()

        if action == Directions.STOP:
            score -= 8 #penalty for stopping, can tweak if not strong enough

        foodList = newFood.asList() #from note
        numFoodLeft = len(foodList)
        if numFoodLeft > 0:
            #use manhattan distance to closest food for heuristic base
            minFoodDist = min(manhattanDistance(newPos, food) for food in foodList)
            score -= 2 * minFoodDist #closer to food
            score -= 4 * numFoodLeft #less food left

        #pac is circling around food
        curFoodCount = len(currentGameState.getFood().asList())
        newFoodCount = len(newFood.asList())
        if newFoodCount < curFoodCount:
            #really make it want to eat the food
            score += 100

        #should behave dif based on if scared
        #lwk use same thing as food
        ghostDists = [manhattanDistance(newPos, ghostState.getPosition()) for ghostState in newGhostStates]
        for dist, scared in zip(ghostDists, newScaredTimes):
            #swap signs based on scared or not
            if scared > 0:
                # scared -> closer good
                score += 10.0 / (dist + 1)
            else:
                # not -> closer bad
                score -= 5.0 / (dist + 1) #adjusted to lower value as it seemed to over penalize ghost distance
                if dist <= 1:
                    score -= 200 #death is rly bad

        return score

def scoreEvaluationFunction(currentGameState: GameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        def value(state, depth, agentIndex):
            #return end state or max depth
            if state.isWin() or state.isLose() or depth == self.depth: #use self.depth from initi
                return self.evaluationFunction(state)

            nextAgent = agentIndex + 1
            nextDepth = depth

            #cycle agents
            if nextAgent == gameState.getNumAgents():
                nextAgent = 0
                nextDepth = depth + 1 #only by 1 then into next cycle

            # pac (max) else ghost (min)
            if agentIndex == 0:
                best = float("-inf")
                for a in state.getLegalActions(agentIndex):
                    best = max(best, value(state.generateSuccessor(agentIndex, a), nextDepth, nextAgent))
                return best
            else:
                best = float("inf")
                for a in state.getLegalActions(agentIndex):
                    best = min(best, value(state.generateSuccessor(agentIndex, a), nextDepth, nextAgent))
                return best

        bestAction = None
        bestScore = float('-inf')

        for action in gameState.getLegalActions(0):
            # after pacman moves, next is ghost 1 (if it exists)
            nextAgent = 1 if gameState.getNumAgents() > 1 else 0
            val = value(gameState.generateSuccessor(0, action), 0, nextAgent)

            if bestAction is None or val > bestScore:
                bestScore = val
                bestAction = action

        return bestAction

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        def value(state, depth, agentIndex, alpha, beta):
            #copy top portion of minimax
            #return end state or max depth
            if state.isWin() or state.isLose() or depth == self.depth: #use self.depth from initi
                return self.evaluationFunction(state)

            nextAgent = agentIndex + 1
            nextDepth = depth

            #cycle agents
            if nextAgent == gameState.getNumAgents():
                nextAgent = 0
                nextDepth = depth + 1

            # Pacman (max)
            if agentIndex == 0:
                best = float("-inf")
                for a in state.getLegalActions(agentIndex):
                    best = max(best, value(state.generateSuccessor(agentIndex, a), nextDepth, nextAgent, alpha, beta))
                    #used pseudocode
                    if best > beta:
                        return best
                    alpha = max(alpha, best)
                return best

            # Ghosts (min)
            else:
                best = float("inf")
                for a in state.getLegalActions(agentIndex):
                    best = min(best, value(state.generateSuccessor(agentIndex, a), nextDepth, nextAgent, alpha, beta))
                    # used pseudocode
                    if best < alpha:
                        return best
                    beta = min(beta, best)
                return best

        bestAction, bestVal = None, float("-inf")
        alpha, beta = float("-inf"), float("inf")

        for action in gameState.getLegalActions(0):
            nextAgent = 1 if gameState.getNumAgents() > 1 else 0
            val = value(gameState.generateSuccessor(0, action), 0, nextAgent, alpha, beta)

            if bestAction is None or val > bestVal:
                bestVal = val
                bestAction = action

            alpha = max(alpha, bestVal)

        return bestAction

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        def value(state, depth, agentIndex):
            #return end state or max depth
            if state.isWin() or state.isLose() or depth == self.depth: #use self.depth from initi
                return self.evaluationFunction(state)

            nextAgent = agentIndex + 1
            nextDepth = depth

            #cycle agents
            if nextAgent == gameState.getNumAgents():
                nextAgent = 0
                nextDepth = depth + 1

            # Pacman (max)
            if agentIndex == 0:
                best = float("-inf")
                for a in state.getLegalActions(agentIndex):
                    best = max(best, value(state.generateSuccessor(agentIndex, a), nextDepth, nextAgent))
                return best

            # Ghosts (expecti min) - avg of all actions
            else:
                total = 0
                prob = 1.0 / len(state.getLegalActions(agentIndex)) #uniform prob
                for a in state.getLegalActions(agentIndex):
                    total += prob * value(state.generateSuccessor(agentIndex, a), nextDepth, nextAgent)
                return total

        bestAction, bestVal = None, float("-inf")

        for action in gameState.getLegalActions(0):
            nextAgent = 1 if gameState.getNumAgents() > 1 else 0
            val = value(gameState.generateSuccessor(0, action), 0, nextAgent)

            if bestAction is None or val > bestVal:
                bestVal = val
                bestAction = action

        return bestAction

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    base of game score
    reward for less food and distance to it
    penalty for dist to ghost (reward if scared)
    penalty for leaving capsules
    penalty for stopping (to help push it when frozen)
    """
    "*** YOUR CODE HERE ***"
    if currentGameState.isWin():
        return float("inf")
    elif currentGameState.isLose():
        return float("-inf")

    pos = currentGameState.getPacmanPosition()
    foodList = currentGameState.getFood().asList()
    capsules = currentGameState.getCapsules()
    ghostStates = currentGameState.getGhostStates()
    score = currentGameState.getScore()

    score -= 20 * len(foodList)

    if foodList:
        minFoodDist = min(manhattanDistance(pos, f) for f in foodList)
        score += 10.0 / minFoodDist

    for ghost in ghostStates:
        dist = manhattanDistance(pos, ghost.getPosition())
        if ghost.scaredTimer > 0:
            score += 100.0 / (dist + 1)
        else:
            if dist < 2:
                score -= 1000
            elif dist < 4:
                score -= 5.0 / dist

    score -= 100 * len(capsules)

    #might not be allowed but helps push it when stopped/frozen
    if currentGameState.getPacmanState().getDirection() == 'Stop':
        score -= 50

    return score

# Abbreviation
better = betterEvaluationFunction

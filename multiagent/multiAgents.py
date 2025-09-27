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

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
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

    def evaluationFunction(self, currentGameState, action):
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
        # Return best possible score (positive infinity) if win, worst possible score (negative infinity) if loss
        if currentGameState.isWin():
            return float('inf')  
        if currentGameState.isLose():
            return float('-inf') 
        
        score = successorGameState.getScore()
        foodList = newFood.asList()
        
        # Find distance to the closest food particle
        distToClosestFood = float('inf')
        for food in foodList:
            distToCurFood = manhattanDistance(newPos, food)
            if distToCurFood < distToClosestFood:
                distToClosestFood = distToCurFood

        # Add reciprocal of distance to the closest food particle (closer food = better score)
        score += 1 / distToClosestFood
                
        # For each ghost, add/subtract reciprocal of distance based on ghost's scared status
        for i in range(0, len(newGhostStates)):
            curGhost = newGhostStates[i]
            curGhostScaredTime = newScaredTimes[i]
            curGhostPos = curGhost.getPosition()
            distToCurGhost = manhattanDistance(newPos, curGhostPos)

            if curGhostScaredTime > 0:
                # Add reciprocal of distance to each scared ghost (closer scared ghost = better score)
                score += 1 / distToCurGhost
            else:
                if distToCurGhost > 0:  
                    # Subtract reciprocal of distance to each normal ghost (closer normal ghost = worse score)
                    score -= 1 / distToCurGhost
                else: 
                    # Return worst possible score if overlapping with normal ghost (Pacman death)
                    return float('-inf')

        return score

def scoreEvaluationFunction(currentGameState):
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

    def getAction(self, gameState):
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
        def isTerminal(state, agentIndex, depth):
            """
            Checks whether the current state is terminal for search.

            Terminal if:
            - The depth limit for tree expansion is reached;
            - The game state is a win or lose;
            - The agent has no legal actions (cannot move further).
            """
            # check for win or lose
            if state.isWin() or state.isLose():
                return True
            # if we finished a full round, check depth
            if depth == self.depth:
                return True
            # if the agent cannot move, game basically over
            if len(state.getLegalActions(agentIndex)) == 0:
                return True
            return False

        def minimaxValue(state, agentIndex, depth):
            """
            Recursively computes the minimax value for a given state.

            Pacman (agent 0) tries to maximize the score.
            Ghosts (agents 1+) try to minimize Pacman's score.

            When all agents have moved, the search depth will increase.
            """
            if isTerminal(state, agentIndex, depth):
                # use evaluation function when terminal state or cut-off reached
                return self.evaluationFunction(state)

            numAgents = state.getNumAgents()
            # move to next agent; if we're back at Pacman, increase depth
            nextAgent = (agentIndex + 1) % numAgents
            nextDepth = depth + 1 if nextAgent == 0 else depth

            actions = state.getLegalActions(agentIndex)

            # pacman acts as a maximizer: picks the best scoring move
            if agentIndex == 0:
                bestValue = float('-inf')
                for action in actions:
                    successor = state.generateSuccessor(agentIndex, action)
                    value = minimaxValue(successor, nextAgent, nextDepth)
                    bestValue = max(bestValue, value)
                return bestValue
            # ghost acts as minimizer: picks the worst scoring move for Pacman
            else:
                bestValue = float('inf')
                for action in actions:
                    successor = state.generateSuccessor(agentIndex, action)
                    value = minimaxValue(successor, nextAgent, nextDepth)
                    bestValue = min(bestValue, value)
                return bestValue

        # Pacman turn: choose action with highest minimax value
        bestScore = float('-inf')
        bestAction = None
        for action in gameState.getLegalActions(0):  # Pacman's legal actions
            successor = gameState.generateSuccessor(0, action)
            score = minimaxValue(successor, 1, 0)     # Start search with ghost 1, depth 0
            if score > bestScore:
                bestScore = score
                bestAction = action

        # return the optimal move for Pacman
        return bestAction
    
class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        def isTerminal(state, agentIndex, depth):
            """
            Checks if the search should stop:
            - Game state is win or lose.
            - Maximum search depth reached (full ply).
            - No legal actions left for agent.
            """
            if state.isWin() or state.isLose():
                return True # game is over, so cutoff
            if depth == self.depth:
                return True  # max search depth has been reached
            if len(state.getLegalActions(agentIndex)) == 0:
                return True # no way for agent to continue
            return False #otherwise, continue searching

        def alphaBetaMinimaxValue(state, agentIndex, depth, alpha, beta):
            """
            Recursive alpha-beta minimax.
            - Pacman maximizes value.
            - Ghosts minimize value.
            """
            # terminal state or cutoff: we need to evaluate
            if isTerminal(state, agentIndex, depth):
                return self.evaluationFunction(state)

            numAgents = state.getNumAgents() 
            nextAgent = (agentIndex + 1) % numAgents #moves to the next agent in the cycle
            nextDepth = depth + 1 if nextAgent == 0 else depth # increase depth if we're back at pacman (all agents moved)

            actions = state.getLegalActions(agentIndex)

            # PACMAN (max)
            if agentIndex == 0:
                value = float('-inf')
                for action in actions:
                    successor = state.generateSuccessor(agentIndex, action) # get next state after action
                    #recursively get minimax value of successor state
                    value = max(value, alphaBetaMinimaxValue(successor, nextAgent, nextDepth, alpha, beta))
                    # prune branch if value is definitely better for Pacman than best so far for min player
                    if value > beta:
                        return value  # cut off value
                    alpha = max(alpha, value) # update alpha
                return value # return best value found for Pacman
            # GHOST (min)
            else:
                value = float('inf')
                for action in actions:
                    successor = state.generateSuccessor(agentIndex, action) # get next state after action
					#recursively get minimax value of successor state
                    value = min(value, alphaBetaMinimaxValue(successor, nextAgent, nextDepth, alpha, beta))
                    # prume if value is definitely worse for Pacman than max so far for max player
                    if value < alpha:
                        return value  # Cut off
                    beta = min(beta, value)
                return value# return best value found for ghost

        # find Pacman move with best alpha-beta minimax value
        bestScore = float('-inf')
        bestAction = None
        alpha = float('-inf')
        beta = float('inf')
        for action in gameState.getLegalActions(0):  # pacmans's available moves
            successor = gameState.generateSuccessor(0, action) # get state after pacman move
            score = alphaBetaMinimaxValue(successor, 1, 0, alpha, beta) # get minimax value starting with ghost 1, depth 0
            if score > bestScore: # update best score and action if we found a new best
                bestScore = score
                bestAction = action
            # update alpha for root (important for correct pruning on siblings)
            alpha = max(alpha, bestScore)

        return bestAction

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        def isTerminal(state, agentIndex, depth):
            """
            Checks whether the current state is terminal for search.

            Terminal if:
            - The depth limit for tree expansion is reached;
            - The game state is a win or lose;
            - The agent has no legal actions (cannot move further).
            """
            # check for win or lose
            if state.isWin() or state.isLose():
                return True
            # if we finished a full round, check depth
            if depth == self.depth:
                return True
            # if the agent cannot move, game basically over
            if len(state.getLegalActions(agentIndex)) == 0:
                return True
            return False

        def expectimaxValue(state, agentIndex, depth):
            """
            Recursively computes the expectimax value for a given state.

            Pacman (agent 0) tries to maximize the score.
            Ghosts (agents 1+) choose a legal move uniformly at random.

            When all agents have moved, the search depth will increase.
            """
            if isTerminal(state, agentIndex, depth):
                # use evaluation function when terminal state or cut-off reached
                return self.evaluationFunction(state)

            numAgents = state.getNumAgents()
            # move to next agent; if we're back at Pacman, increase depth
            nextAgent = (agentIndex + 1) % numAgents
            nextDepth = depth + 1 if nextAgent == 0 else depth

            actions = state.getLegalActions(agentIndex)

            # pacman acts as a maximizer: picks the best scoring move
            if agentIndex == 0:
                bestValue = float('-inf')
                for action in actions:
                    successor = state.generateSuccessor(agentIndex, action)
                    value = expectimaxValue(successor, nextAgent, nextDepth)
                    bestValue = max(bestValue, value)
                return bestValue
            # Ghost acts randomly: takes the mean score over all legal moves
            else:
                totalValue = 0
                for action in actions:
                    successor = state.generateSuccessor(agentIndex, action)
                    value = expectimaxValue(successor, nextAgent, nextDepth)
                    totalValue += value
                meanValue = totalValue / len(actions)
                return meanValue

        # Pacman turn: choose action with highest expectimax value
        bestScore = float('-inf')
        bestAction = None
        for action in gameState.getLegalActions(0):  # Pacman's legal actions
            successor = gameState.generateSuccessor(0, action)
            score = expectimaxValue(successor, 1, 0)     # Start search with ghost 1, depth 0
            if score > bestScore:
                bestScore = score
                bestAction = action

        # return the optimal move for Pacman
        return bestAction

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: A better evaluation function that guides Pacman to:
    - win quickly,
    - eat food efficiently,
    - avoid getting caught by ghosts unless they're scared,
    - and take capsules wisely.
    - and incorporates a small random factor to break ties.
    """
    from util import manhattanDistance

    # return best/worst possible values for immediate win/loss
    if currentGameState.isWin():
        return float('inf')
    if currentGameState.isLose():
        return float('-inf')

    # start with the current built-in score as our base
    score = currentGameState.getScore()

    # gather positions and lists needed for features
    pacmanPos = currentGameState.getPacmanPosition()
    foodList = currentGameState.getFood().asList()
    ghostStates = currentGameState.getGhostStates()
    capsules = currentGameState.getCapsules()

    # our evaluation function will consider several features: the first is food collection
    if foodList:
        # find closest food distance; motivates Pacman to quickly eat available food
        closestFoodDist = min(manhattanDistance(pacmanPos, food) for food in foodList)
        score += 2.5 / closestFoodDist
        # Fewer food pellets left is always good
        score -= 4 * len(foodList)

    # the second feature is capsule collection
    # fewer capsules left means Pacman is progressing towards winning
    score -= 8 * len(capsules)
    if capsules:
        closestCapDist = min(manhattanDistance(pacmanPos, cap) for cap in capsules)
        score += 1.5 / (closestCapDist + 1)  # +1 just in case Pacman is on a capsule

    # the third feature is ghost interaction
    for ghost in ghostStates:
        ghostDist = manhattanDistance(pacmanPos, ghost.getPosition())
        if ghost.scaredTimer > 0:
            # reward being close to scared ghosts (so Pacman chases them)
            score += 18.0 / (ghostDist + 1)
        else:
            # punish for being too close to active ghosts
            if ghostDist == 0:
                return float('-inf')
            elif ghostDist < 2:
                score -= 300  # large penalty if death is likely
            else:
                score -= 2.0 / ghostDist  # small penalty when within range but not both adjacent

    # small random factor to break ties randomly so Pacman doesn't get stuck
    score += random.random() * 0.01

    return score

# Abbreviation
better = betterEvaluationFunction

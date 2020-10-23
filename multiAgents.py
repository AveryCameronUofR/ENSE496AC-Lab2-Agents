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
import random
import util

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
        scores = [self.evaluationFunction(
            gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(
            len(scores)) if scores[index] == bestScore]
        # Pick randomly among the best
        chosenIndex = random.choice(bestIndices)

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

        #Set up starting values
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = currentGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [
            ghostState.scaredTimer for ghostState in newGhostStates]
        score = 0
        foodScore = 0
        distScore = 0
        foodList = newFood.asList()

        #Punish staying in the same place
        currentPos = currentGameState.getPacmanPosition()
        if (currentPos == newPos):
            score = -50
        else:
            foodScore = 50

        #check the closest food(dist) and check the number of close pellets (distScore) 
        from math import inf
        dist = inf
        for food in foodList:
            distTemp = manhattanDistance(food, newPos)
            if (distTemp == 0):
                foodScore += 100
            else:
                dist = min(dist, distTemp)
                distScore += (10.0/distTemp)
        #Scale number of close pellets
        distScore = distScore*5

        #Check ghosts
        ghostScore = 0
        i = 0
        while (i < len(newGhostStates)):
            ghost = newGhostStates[i]
            scared = newScaredTimes[i]
            ghostPosition = ghost.getPosition()
            #Get the distance from ghost to pacman
            distGhost = manhattanDistance(ghostPosition, newPos)
            if (distGhost < 2):
                #if the ghost is close and can kill pacman, reduce score, otherwise increase it
                if (scared == 0):
                    ghostScore = -500
                else:
                    ghostScore = scared*5
            #if the ghost is close but not too close, reduce score if it isn't scared
            #if it is further than 10, ignore it, it is far away and we can focus food
            elif distGhost < 10:
                if (scared):
                    ghostScore = distGhost * 5
                else:
                    ghostScore = distGhost * -5
            i += 1
        #Return food, closest food, distance pellets, ghost scores 
        return foodScore + ghostScore + distScore + score + dist

    def manhattanDistance(self, xy1, xy2):
        "Returns the Manhattan distance between points xy1 and xy2"
        return abs(xy1[0] - xy2[0]) + abs(xy1[1] - xy2[1])


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

    def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """
    # return the action to get to the score

    def getAction(self, gameState, currDepth=-1, index=0):
        "*** YOUR CODE HERE ***"
        import math
        index = index % gameState.getNumAgents()
        maximizingPlayer = False
        #if index is 0, maximizing agent, Pacman
        if index == 0:
            currDepth += 1
            maximizingPlayer = True
        
        if currDepth == self.depth or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)
        
        #Get legal actions of next agent and check successors
        nextIndex = index + 1
        actions = gameState.getLegalActions(index)
        successors = []
        for action in actions:
            successor = gameState.generateSuccessor(index, action)
            successors.append(
                (action, self.getAction(successor, currDepth, nextIndex)))

        action = ''
        value = math.inf
        if (maximizingPlayer):
            value = -value
        #update max/min value
        for successorAction, successorScore in successors:
            if (maximizingPlayer and successorScore > value) or (not maximizingPlayer and successorScore < value):
                value = successorScore
                action = successorAction
        #if it is the start, pacman and depth 0 return action, otherwise use value for max and mins
        if currDepth == 0 and index == 0:
            return action
        else:
            return value


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """
    import math

    def getAction(self, gameState, currDepth=-1, index=0, a=-math.inf, b=math.inf):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        import math
        index = index % gameState.getNumAgents()
        maximizingPlayer = False
        valueCheck = math.inf
        #if index is 0, maximizing agent, Pacman
        if index == 0:
            valueCheck = -math.inf
            currDepth += 1
            maximizingPlayer = True

        tempIndex = index + 1
        if currDepth == self.depth or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)

        #Get legal actions of next agent and check successors
        successors = []
        actions = gameState.getLegalActions(index)
        for action in actions:
            successor = gameState.generateSuccessor(index, action)
            value = self.getAction(successor, currDepth, tempIndex, a, b)
            #Compare to alpha and beta as needed, prune by returning early, otherwise append
            if (maximizingPlayer):
                valueCheck = max(valueCheck, value)
                a = max(a, value)
                if (value > b):
                    return value
            else:
                valueCheck = min(valueCheck, value)
                b = min(b, value)
                if (value < a):
                    return value
            successors.append((action, value))

        action = ''
        value = math.inf
        if (maximizingPlayer):
            value = -math.inf
        #Update max/min of layer
        for successorAction, successorScore in successors:
            if (maximizingPlayer and successorScore > value) or (not maximizingPlayer and successorScore < value):
                value = successorScore
                action = successorAction
        #return action if pacman depth 0, otherwise return max/min of layer
        if currDepth == 0 and index == 0:
            return action
        else:
            return value


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState, currDepth=-1, index=0):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        import math
        import random
        import math
        index = index % gameState.getNumAgents()
        maximizingPlayer = False
        if index == 0:
            currDepth += 1
            maximizingPlayer = True

        if currDepth == self.depth or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)
        
        #Generate Successors from legal actions
        actions = gameState.getLegalActions(index)
        actionCount = len(actions)
        successors = []
        for action in actions:
            successor = gameState.generateSuccessor(index, action)
            tempIndex = index + 1
            successors.append(
                (action, self.getAction(successor, currDepth, tempIndex)))

        action = ''
        value = math.inf
        if (maximizingPlayer):
            value = -value
            #find max value
            for successorAction, successorScore in successors:
                if maximizingPlayer and successorScore > value:
                    value = successorScore
                    action = successorAction
        else:
            value = 0
            #Average score and pick a random action
            for successorAction, successorScore in successors:
                value += successorScore
            action = actions[random.randint(0, actionCount - 1)]
            value = float(value)/float(actionCount)

        #Return action if pacman and depth 0, value (max or random minimizing agent value)
        if currDepth == 0 and index == 0:
            return action
        else:
            return value


def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    First we check the closest food, it is better to be close
    to food than far away from it, after fiddling with the score
    modifier, 10 seemed to provide the best results

    Then check for Super pellets, these let us increase our score
    by eating the ghosts, eating these is good but a secondary
    goal, 10 also seemed a good modifier
    
    Then check the ghosts, if they are within on the same pos, reduce the 
    score, -500, since that is a loss.

    Then check the distance to ghosts, 
    If they are withing 1 spot, that is dangerous, reduce the score
    10 modifier again

    If they are withing 10 spots, reduce score as this is a risk,
    use 5 as a modifier
    
    if they are scared,
    check and see if you can reach them in time and eat them 
    this will increase the score and a modifier of 50 worked well.

    Add these scores to the current game score,
    since the current game score matters most this is great, 
    since the others you can take 1 divided by the value and 
    use a modifier these are small values but changes have
    a large impact.

    By adjusting the modifiers separately, the average score 
    was able to be 1141.6
    
    """
    from math import inf
    currentPos = currentGameState.getPacmanPosition()
    currentFood = currentGameState.getFood()
    currentGhostStates = currentGameState.getGhostStates()
    currentScaredTimes = [ghostState.scaredTimer for ghostState in currentGhostStates]
    superPellets = currentGameState.getCapsules()
    
    foodList = currentFood.asList()
    foodScore = 0 
    if (len(foodList) > 0):
        foodScore = 1/len(foodList) * 10
    #check the closest food
    minFoodDist = inf
    for food in foodList:
        distTemp = manhattanDistance(food, currentPos)
        minFoodDist = min(minFoodDist, distTemp)
    minFoodDist = 1/minFoodDist *10

    #check the closest Super Pellet/Capsule
    minSuperDist = inf
    for superPellet in superPellets:
        distTemp = manhattanDistance(superPellet, currentPos)
        minSuperDist = min(minSuperDist, distTemp)
    minSuperDist = 1/minSuperDist *10
    #Check ghosts
    ghostScore = 0
    
    i = 0
    while (i < len(currentGhostStates)):
        ghost = currentGhostStates[i]
        scared = currentScaredTimes[i]
        ghostPosition = ghost.getPosition()
        #Get the distance from ghost to pacman
        distGhost = manhattanDistance(ghostPosition, currentPos)
        if (scared == 0):
            #if distance is 0, it is a loss 
            if (distGhost == 0):
                ghost = -500
            elif (distGhost < 2):               #this is dangerous
                ghostScore = -1/distGhost*10
            elif distGhost < 10:                #this is risky
                ghostScore = -1/distGhost*5
        else:                                   #you can maybe eat them!
            if (scared > distGhost):
                ghostScore = 40 / (scared - distGhost) * 50
        i += 1
    
    #Return gameScore, food, closest food, closest Super Capsule and ghost scores 
    return currentGameState.getScore() + foodScore + minFoodDist + minSuperDist + ghostScore

# Abbreviation
better = betterEvaluationFunction

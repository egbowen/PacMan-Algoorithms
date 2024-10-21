"""
CS311 Programming Assignment 2: Adversarial Search

Full Name: Liz Bowen

Brief description of my evaluation function:

My evaluation function takes the shortest distance to food, capsules, and ghosts, then weights and adds them to the score.
The move with the highest score is chosen. It takes into account keeping away from threatening ghosts, and highly incentivizes
getting capsules, then chasing after scared ghosts, an action that improves the win rate. It also pushes pacman to go in search 
of food. This improves the win rate by minimizing being caught by ghosts and maximizing food outtake and getting scared
ghosts.
"""

import math, random, typing

import util
from game import Agent, Directions
from pacman import GameState



class ReflexAgent(Agent):
    """
    A reflex agent chooses the best action at each choice point by examining its alternatives via a state evaluation
    function.

    The code below is provided as a guide. You are welcome to change it as long as you don't modify the method
    signatures.
    """

    def __init__(self):
        self.lastPositions = []
        self.dc = None

    def getAction(self, gameState: GameState) -> str:
        """Choose the best action according to an evaluation function.

        Review pacman.py for the available methods on GameState.

        Args:
            gameState (GameState): Current game state

        Returns:
            str: Chosen legal action in this state
        """
        # Collect legal moves
        legalMoves = gameState.getLegalActions()

        # Compute the score for the successor states, choosing the highest scoring action
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [
            index for index in range(len(scores)) if scores[index] == bestScore
        ]
        # Break ties randomly
        chosenIndex = random.choice(bestIndices)

        return legalMoves[chosenIndex]

    def evaluationFunction(self, gameState: GameState, action: str):
        """Compute score for current game state and proposed action"""
        successorGameState = gameState.generatePacmanSuccessor(action)
        return successorGameState.getScore()


def scoreEvaluationFunction(gameState: GameState) -> float:
    """
    Return score of gameState (as shown in Pac-Man GUI)

    This is the default evaluation function for adversarial search agents (not reflex agents)
    """
    return gameState.getScore()


class MultiAgentSearchAgent(Agent):
    """
    Abstract Base Class for Minimax, AlphaBeta and Expectimax agents.

    You do not need to modify this class, but it can be a helpful place to add attributes or methods that used by
    all your agents. Do not remove any existing functionality.
    """

    def __init__(self, evalFn=scoreEvaluationFunction, depth=2):
        self.index = 0  # Pac-Man is always agent index 0
        self.evaluationFunction = globals()[evalFn] if isinstance(evalFn, str) else evalFn
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):
    """Minimax Agent"""


    def getAction(self, gameState: GameState) -> str:
        """Return the minimax action from the current gameState.

        * Use self.depth (depth limit) and self.evaluationFunction.
        * A "terminal" state is when Pac-Man won, Pac-Man lost or there are no legal moves.
        """

        """
        Some potentially useful methods on GameState (recall Pac-Man always has an agent index of 0, the ghosts >= 1):

        getLegalActions(agentIndex): Returns a list of legal actions for an agent
        generateSuccessor(agentIndex, action): Returns the successor game state after an agent takes an action
        getNumAgents(): Return the total number of agents in the game
        getScore(): Return the score corresponding to the current state of the game
        isWin(): Return True if GameState is a winning state
        gameState.isLose(): Return True if GameState is a losing state
        """     
        def doMiniMax(state: GameState, index, depth):
            if (depth >= self.depth) or state.isWin() or state.isLose():
                return self.evaluationFunction(state)
            
            if index == 0:
                return maxVal(state, depth)
            else:
                return minVal(state, index, depth)
                
        #pac man's best option
        def maxVal(state: GameState, depth):   
            '''
            Get pacman's best choice
            Args:
                state: Gamestate we are in
                depth: our current depth in tree
            '''    
            actions = state.getLegalActions(0)
            maxScore = float('-inf')
           
            for action in actions:
                nextState = state.generateSuccessor(0, action)
                currScore = doMiniMax(nextState, 1, depth) 
                maxScore = max(maxScore, currScore)
            return maxScore

        def minVal (state: GameState, index, depth):
            '''
            Get ghosts's best choice
            Args:
                state: Gamestate we are in
                depth: our current depth in tree
            '''   
            #need to go through all ghost options 
            numGhosts = state.getNumAgents() - 1
            actions = state.getLegalActions(index)
            minScore = float('inf')
            
            for action in actions:
                nextState = state.generateSuccessor(index, action)
                if index >= numGhosts:  # last ghost --> go to pacman
                    currScore = doMiniMax(nextState, 0, depth + 1)
                else: # next value to go to is another ghost
                    currScore = doMiniMax(nextState, index + 1, depth)
            minScore = min(minScore, currScore)
            
            #return best for ghost
            return minScore
        
        #start off the recursion with pacman: index = 0
        initActions = gameState.getLegalActions(0)
        maxScore = float('-inf') 
        bestAction = None
        
        for action in initActions:
            nextState = gameState.generateSuccessor(0, action)
            currScore = doMiniMax(nextState, 1, 0) 
            
            if bestAction is None or currScore > maxScore:
                maxScore = currScore
                bestAction = action
        return bestAction
    

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Minimax agent with alpha-beta pruning
    """
    def getAction(self, gameState: GameState) -> str:
        """Return the minimax action with alpha-beta pruning from the current gameState.

        * Use self.depth (depth limit) and self.evaluationFunction.
        * A "terminal" state is when Pac-Man won, Pac-Man lost or there are no legal moves.
        """
        # TODO: Implement your Minimax Agent with alpha-beta pruning
        def doMiniMax(state: GameState, index, depth, alpha, beta):
            if (depth >= self.depth) or state.isWin() or state.isLose():
                return self.evaluationFunction(state)
            
            if index == 0:
                return maxVal(state, depth, alpha, beta)
            else:
                return minVal(state, index, depth, alpha, beta)
                
        #pac man's best option
        def maxVal(state: GameState, depth, alpha, beta):   
            '''
            Get pacman's best choice
            Args:
                state: Gamestate we are in
                depth: our current depth in tree
            '''    
            actions = state.getLegalActions(0)
            currScore = float('-inf')
           
            for action in actions:
                nextState = state.generateSuccessor(0, action)
                currScore = max (currScore, doMiniMax(nextState, 1, depth, alpha, beta))
                if currScore >= beta:
                    return currScore
                alpha = max(alpha, currScore)
            return currScore

        def minVal (state: GameState, index, depth, alpha, beta):
            '''
            Get ghosts's best choice
            Args:
                state: Gamestate we are in
                depth: our current depth in tree
            '''   
            #need to go through all ghost options 
            numGhosts = state.getNumAgents() - 1
            actions = state.getLegalActions(index)
            currScore = float('inf')
            
            for action in actions:
                nextState = state.generateSuccessor(index, action)
                if index >= numGhosts:  # last ghost --> go to pacman
                    currScore = min(currScore, doMiniMax(nextState, 0, depth + 1, alpha, beta))
                else: # next value to go to is another ghost
                    currScore = min(currScore, doMiniMax(nextState, index + 1, depth, alpha, beta)) 
                if currScore <= alpha:
                    return currScore
                beta = min(beta, currScore)
            #return best for ghost
            return currScore
        
        #start off the recursion with pacman: index = 0
        initActions = gameState.getLegalActions(0)
        maxScore = float('-inf') 
        bestAction = None
        alphaInit = float('-inf') #I asked chatGPT if beta is - or + infinity
        betaInit = float('inf')
        
        for action in initActions:
            nextState = gameState.generateSuccessor(0, action)
            currScore =  doMiniMax(nextState, 1, 0, alphaInit, betaInit)
            
            if bestAction is None or currScore > maxScore:
                maxScore = currScore
                bestAction = action
            alphaInit = max(alphaInit, maxScore)
        print(maxScore)
        return bestAction


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
    Expectimax agent
    """
    def getAction(self, gameState):
        """Return the expectimax action from the current gameState.

        All ghosts should be modeled as choosing uniformly at random from their legal moves.

        * Use self.depth (depth limit) and self.evaluationFunction.
        * A "terminal" state is when Pac-Man won, Pac-Man lost or there are no legal moves.
        """
        def doMiniMax(state: GameState, index, depth):
            if (depth >= self.depth) or state.isWin() or state.isLose():
                return self.evaluationFunction(state)
            
            if index == 0:
                return maxVal(state, depth)
            else:
                return minVal(state, index, depth)
                
        #pac man's best option
        def maxVal(state: GameState, depth):   
            '''
            Get pacman's best choice
            Args:
                state: Gamestate we are in
                depth: our current depth in tree
            '''    
            actions = state.getLegalActions(0)
            maxScore = float('-inf')
           
            for action in actions:
                nextState = state.generateSuccessor(0, action)
                currScore = doMiniMax(nextState, 1, depth) 
                maxScore = max(maxScore, currScore)
            return maxScore

        def minVal (state: GameState, index, depth):
            '''
            Get ghosts's best choice
            Args:
                state: Gamestate we are in
                depth: our current depth in tree
            '''   
            #need to go through all ghost options 
            numGhosts = state.getNumAgents() - 1
            actions = state.getLegalActions(index)
            minScore = float('inf')
            totalScores = 0
            
            for action in actions:
                nextState = state.generateSuccessor(index, action)
                if index >= numGhosts:  # last ghost --> go to pacman
                    currScore = doMiniMax(nextState, 0, depth + 1)
                else: # next value to go to is another ghost
                    currScore = doMiniMax(nextState, index + 1, depth)
                totalScores+=currScore
            #get expected value of children
            expVal = totalScores/len(actions)
            return expVal
        
        #start off the recursion with pacman: index = 0
        initActions = gameState.getLegalActions(0)
        maxScore = float('-inf') 
        bestAction = None
        
        for action in initActions:
            nextState = gameState.generateSuccessor(0, action)
            currScore = doMiniMax(nextState, 1, 0) 
            
            if bestAction is None or currScore > maxScore:
                maxScore = currScore
                bestAction = action
        return bestAction


def betterEvaluationFunction(gameState: GameState) -> float:
    """
    Return score of gameState using custom evaluation function that improves agent performance.
    """

    """
    The evaluation function takes the current GameStates (pacman.py) and returns a number,
    where higher numbers are better.

    Some methods/functions that may be useful for extracting game state:
    gameState.getPacmanPosition() # Pac-Man position
    gameState.getGhostPositions() # List of ghost positions
    gameState.getFood().asList() # List of positions of current food
    gameState.getCapsules() # List of positions of current capsules
    gameState.getGhostStates() # List of ghost states, including if current scared (via scaredTimer)
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
    """
    # Implement your evaluation function
    
    pacPos = gameState.getPacmanPosition()
    ghostStates = gameState.getGhostStates()
    foodList = gameState.getFood().asList()
    capsules = gameState.getCapsules()
    ghostPositions = gameState.getGhostPositions()

    ghostWeight = 10
    scaredWeight = 5
    foodWeight = 3
    capWeight =  5
    
    metric = 0

    for ghostState, ghostPos in zip(ghostStates, ghostPositions):
        ghostDist = abs(pacPos[0] - ghostPos[0]) + abs(pacPos[1] - ghostPos[1])

        if ghostState.scaredTimer == 0:
            if  ghostDist > 0:
                if ghostDist <= 4:
                    metric -= (ghostWeight*4) / ghostDist
                else:
                    metric -= ghostWeight / ghostDist
        else:
            metric += scaredWeight / (ghostDist + 1)

    if foodList:
        foodDistances = [abs(pacPos[0] - food[0]) + abs(pacPos[1] - food[1]) for food in foodList]
        nearestFoodDist = min(foodDistances)
        metric += foodWeight / (nearestFoodDist + 1)

    if capsules:
        capDist = [abs(pacPos[0] - cap[0]) + abs(pacPos[1] - cap[1]) for cap in capsules]
        nearestCap = min(capDist)
        metric += capWeight / (nearestCap + 1)
    
    return gameState.getScore() + metric    
    


# Create short name for custom evaluation function
better = betterEvaluationFunction

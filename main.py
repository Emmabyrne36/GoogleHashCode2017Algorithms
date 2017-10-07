import numpy as np
from read_input import read_google
from astropy.wcs.docstrings import row
from copy import deepcopy
import random, math, time
from heapq import nlargest

# Generate the input data
data = read_google("input/hash_code_sample_file.in")

# Defining the number of rows and columns in the matrix - x is rows, y is columns
x = data["number_of_videos"]
y = data["number_of_caches"]

# The matrix output for hash_code_sample_file.in
testMatrix = [[0,0,1,0,0],[0,1,0,1,0],[1,1,0,0,0]]

# Create the matrix
def createMatrix(x,y):
    ''' Function that creates a 2D matrix based depending on the input '''

    # Initial matrix which is populated with all 0's
    matrix = [[0] * x for i in range(y)]

    return matrix

matrix = createMatrix(x,y)

def printMatrix(matrix):
    ''' Function to print out the 2D matrix neatly '''
    for row in matrix:
        print(row)

printMatrix(createMatrix(x,y))
print()
printMatrix(testMatrix)
print()

# =====================================================================================================
# ==============================  Feasibility, Fitness, Hill Climbing =================================
# =====================================================================================================

def is_feasible(matrix):
    ''' A funciton that tests for each cache server, the sum of the files is not

    overflowing the cache server. If this is the case and the cache is larger than
    the cache limit, then the solution is not possible and the function returns -1 

     '''
    for row in range(0, len(matrix)):
        sumVids = 0
        for col in range(0, len(matrix[0])):
            if matrix[row][col] == 1:
                sumVids += data["video_size_desc"][col]
                if sumVids > data["cache_size"]:
                    return -1


def fitness(matrix):
    """ Finds the fitness score of an input matrix if it is feasible """ 
    is_feasible(matrix)

    fileID = 0
    endpointID = 0
    numRequests = 0     
    totalRequests = 0
    gains = 0
    
    # Iterate over the dictionary, get values for fileId, endpointId and number of requests
    for key, value in data["video_ed_request"].items():
        fileID = int(key[0])
        endpointID = int(key[1])
        caches = []
        epToCacheLatency = 0
        inCache = False
        
        for cacheNum in range(0, len(matrix)):
            if matrix[cacheNum][fileID] == 1: # change this to 1 when the array is populated
                caches.append(cacheNum)
                inCache = True
      
        numRequests = int(value)
        totalRequests += numRequests
        dc_latency = data["ep_to_dc_latency"][endpointID]
        
        if inCache:
            epToCacheLatency = data["ep_to_cache_latency"][endpointID][caches[0]]
        else:
            epToCacheLatency = dc_latency
        
        # Sets the minimum cache latency value
        for i in caches:
            if data["ep_to_cache_latency"][endpointID][i] < epToCacheLatency:
                epToCacheLatency = data["ep_to_cache_latency"][endpointID][i]
                    
        difference = dc_latency - epToCacheLatency

        # If the difference is negative, make it 0, as it is not being streamed from one of the caches
        if difference < 0:
            difference = 0
        gains += (difference * numRequests)
        score = int((gains/totalRequests) * 1000)

  
    return score

def HillClimbing(matrix):
    """ Generates multiple solutions by creating neighbours of the input matrix (by changing 0 to 1 and 1 to 0) 
    
    and obtains the fitness of the neighbour matrices 
    """
    allScores = [] # Store the scores of the feasible matrices here
    bestCoordinate = [] # Store the coordinates of the best score, then make another grid based on these coordinates, change the value of the original grid to match the change in these coordinates
    matrixCopy = deepcopy(matrix) # Create a deepcopy of the original matrix

    for cache in range(0, len(matrixCopy)):
        for file in range(0, len(matrixCopy[0])):
            if matrixCopy[cache][file] == 1:
                matrixCopy[cache][file] = 0
            
            elif matrixCopy[cache][file] == 0:
                matrixCopy[cache][file] = 1
                
            updatedScore = fitness(matrixCopy) 
            if fitness(matrixCopy) != -1:
                allScores.append(updatedScore)
                bestCoordinate.append((cache,file))
            
            matrixCopy = deepcopy(matrix)

    maxScore = max(allScores)
    # Getting the index of a value obtained from - http://stackoverflow.com/questions/364621/how-to-get-items-position-in-a-list
    bestScoreIndex = allScores.index(maxScore)
    row  = bestCoordinate[bestScoreIndex][0]
    col = bestCoordinate[bestScoreIndex][1] # make new grid out of these coordinates
    
    # Create a new matrix with one thing changed - http://stackoverflow.com/questions/6532881/how-to-make-a-copy-of-a-2d-array-in-python  
    newMatrix = deepcopy(matrix)
    if newMatrix[row][col] == 1:
        newMatrix[row][col] = 0
        
    elif newMatrix[row][col] == 0:
        newMatrix[row][col] = 1
    
    return newMatrix, maxScore


def multipleClimbing(matrix, bestScore):
    """ Call on the hill climbing function multiple times to generate a better score """
    moreScores = []
    size = np.shape(matrix) # this gets the dimensions of the array
    # This decides how many iterations to do
    totalBits = size[0] * size[1]
    numTests = int(totalBits/3)
    solution = bestScore
    bestMatrix = matrix
    count = 0

    while count < numTests:
        bestMatrix, newScore = HillClimbing(bestMatrix)
        moreScores.append(newScore)
        if newScore > bestScore:
            solution = newScore
        count += 1
        
        # If the score is the same value as the previous value, break the loop - because the fitness isn't getting any better
        if len(moreScores) > 2:
            if newScore == moreScores[-1]:
                break
    
    print("Iterations:", count)
    print("This is the highest score:", solution)
    print("This is the matrix:", bestMatrix)
    return solution, bestMatrix
    
# ===================================================================================================================
# ===================================  Simulated Annealing ==========================================================
# ===================================================================================================================
def switchValues(x,y):
    """ Used in the genetic function and other functions.
    
    Gets the values of the dimensions of the arrays and if x > y, swaps them.
    This is useful for generating correct coordinates in the genetic function for limiting
    the number of infeasible arrays 
    """
    x, y = x-1, y-1 # decrement x and y by 1 so as to create coordinates within range of the grid in the functions
    if x > y:
        x, y = y, x
    
    return x, y

def neighbours(matrix):
    """ Generates feasible neighbours of the matrix - flips 1 bit """
    a, b = switchValues(x, y)
    foo = False
    while foo == False:
        row = random.randint(0,a)
        col = random.randint(0,b)
        if matrix[row][col] == 1:
            matrix[row][col] = 0
        
        elif matrix[row][col] == 0:
            matrix[row][col] = 1
            
        if is_feasible(matrix) != -1:
            foo = True
        
    return matrix

def acceptance_probability(old, new, T):
    """ Generates a number between 0 and 1 """
    if new > old:
        new, old = old, new
    calculation = old-new/T
    a = math.e*calculation
    if a < 0:
        a = 0
    elif a > 1:
        a = 1
    return a
    
def generateRandomMatrix():
    """ This is adapted from the createParents() function.
    Generates a random matrix and populates it. Returns a feasible matrix """
    potentialMatrix = [[(0) for i in range(x)] for j in range(y)]
    # Generate random index, then add 1 to that index, if it's feasible add it to the list
    a, b = switchValues(x, y)
    # Number of values which will be turned into 1 - and generate coordinates
    numChanges = random.randint(5, (x*y))
    for j in range(0, numChanges//2): # iterate until half of this value to improve chances of getting a viable matrix
        rows = random.randint(0,a)
        cols = random.randint(0,b)
        potentialMatrix[rows][cols] = 1
        
        # If it's valid, add to the multipleMatrices array
        if is_feasible(potentialMatrix) != -1:
            return matrix
    

# Simulated annealing algorithms obtained from - http://katrinaeg.com/simulated-annealing.html
def simulatedAnnealing1(matrix):
    print("Please wait. Calculations in progress...")
    old_score = fitness(matrix)
    T = 1.0
    T_min = 0.00001
    alpha = 0.9
    while T > T_min:
        i = 1
        while i <= 100:
            new_matrix = neighbours(matrix)
            new_score = fitness(new_matrix)
            ap = acceptance_probability(old_score, new_score, T)
            if ap > random.uniform(0, 1):
                matrix = new_matrix
                old_score = new_score
            i += 1
        T = T*alpha
    return matrix, old_score 

def simulatedAnnealing2():
    """ This function runs the simulated annealing by generating its own matrix """
    print("Please wait. Calculations in progress...")
    matrix = generateRandomMatrix()
    old_score = fitness(matrix)
    T = 1.0
    T_min = 0.00001
    alpha = 0.9
    while T > T_min:
        i = 1
        while i <= 100:
            new_matrix = neighbours(matrix)
            new_score = fitness(new_matrix)
            ap = acceptance_probability(old_score, new_score, T)
            if ap > random.uniform(0, 1):
                matrix = new_matrix
                old_score = new_score
            i += 1
        T = T*alpha
    return matrix, old_score


# =================================================================================================================
# =======================================  Genetic Algorigthms  ===================================================
# =================================================================================================================
def createParents():
    """ Find valid matrices and add them to the matrix array.
    
    This function generates 50 parent and 50 children matrices. It combines the parent and children
    matrices and only retains the ones with the best score. The final result is the best overall matrix
    in terms of fitness score.

     """
    multipleMatrices = [] # This will store the valid random arrays
    while len(multipleMatrices) < 50:
        # Create a 2D array with random numbers - http://stackoverflow.com/questions/24108417/simple-way-of-creating-a-2d-array-with-random-numbers-python
        potentialMatrix = [[(0) for i in range(x)] for j in range(y)]
        # Get the dimensions of the matrices
        a, b = switchValues(x, y)
        # Number of values which will be turned into 1 - and generate coordinates
        numChanges = random.randint(5, (x*y))
        for j in range(0, numChanges//2): # iterate until half of this value to improve chances of getting a viable matrix
            rows = random.randint(0,a)
            cols = random.randint(0,b)
            potentialMatrix[rows][cols] = 1
        
        # If it's valid, add to the multipleMatrices array
        if is_feasible(potentialMatrix) != -1:
            #fitness(potentialMatrix)
            multipleMatrices.append(potentialMatrix) # could make deepcopy and append the deepcopy?

    
    return multipleMatrices

def splitLists(myList):
    """ Splits a list into 2 halves """
    half = len(myList)//2
    return myList[:half], myList[half:]

def thirdList(myList):
    """ Splits a list into thirds, just returns the first third """
    third = len(myList)//3
    return myList[:third]

def mutation(matrix):
    """ To generate mutation
    
    Generates the probability of a mutation, if the probability is met, a mutation occurs.
    The mutations are checked for feasibility and if it isn't feasible, a different mutation
    occurs until it generates a feasible solution. If the mutation conditions are not met,
    the original matrix is returned """
    probability = round(1/(x*y), 2)  # round to 2 decimal points - https://gist.github.com/jackiekazil/6201722
    randomNum = round(random.uniform(0,1), 2)
    a, b = switchValues(x, y)
    # Using math.isclose inspired from - https://docs.python.org/3/library/math.html
    foo = False
    while foo == False:
        if math.isclose(randomNum, probability, rel_tol=probability//2, abs_tol=probability//2): # gives a low probability of a mutation
            # If it is to be mutated, swap values at a random position in the matrix - as long as it is feasible
            # This selects what row and column will be mutated
            row = random.randint(0,a) 
            col = random.randint(0,b)
            matrix[row][col]
            if matrix[row][col] == 1:
                matrix[row][col] = 0
            
            elif matrix[row][col] == 0:
                matrix[row][col] = 1
            
        if is_feasible(matrix) != -1:
            foo = True
            
    return matrix


def generateChildren(parents):
    """ This function generates children from the parent input (a 3D matrix) """
    # First generate 2 subparents - split parents into 2
    subParentA, subParentB = splitLists(parents)
    potentialChildren = []
    
    # Iterate through the 2 subParents and append the relative values to a new list
    for index, item in enumerate(subParentA):
        potentialChildren.append(subParentA[index] + subParentB[index])
    
    list4 = [] # stores some of the children which are a mix of subParent A and B
    for i in range(0, len(potentialChildren)):
        for j in range(0, len(potentialChildren[0])):
            list4.append(potentialChildren[i][1::2])
            
    # The potential children list will contain 150 values. We only want 50 so need to delete some matrices
    children = thirdList(list4) # This is now the list containing 50 children
    finalChildren = []
    # Check for mutation probability
    for i in children:
        mutate = mutation(i)
        finalChildren.append(mutate)
        
    return finalChildren

def naturalSelection():
    """ Selects and keeps the 50 fittest individuals """
    print("Please wait. Calculations in progress...")
    # Creates parents and their children
    parents = createParents()
    children = generateChildren(parents) # generates the parents and children
    parentsAndChildren = parents + children # creates 1 list of parents and children - makes the iteration easier
    counter = 0
    # Do 50 generations
    while counter < 50:
        onTheOriginOfLists = [] # This will store the 50 fittest individuals
        correspondingMatrix = []
        for i in range(0, len(parentsAndChildren)):
            m = parentsAndChildren[i]
            onTheOriginOfLists.append(fitness(m)) # append the score to this list
            correspondingMatrix.append(m) # append the corresponding matrix to this list
    
        # To get the 50 largest values in an array adapted from - http://stackoverflow.com/questions/2243542/how-to-efficiently-get-the-k-bigger-elements-of-a-list-in-python
        bestScores = nlargest(50, onTheOriginOfLists)
        
        # Reset parents
        parents = []
        
        # Get the index of the score values
        for score in bestScores:
            matrixIndex = onTheOriginOfLists.index(score) # keeping the fittest ones
            parents.append(correspondingMatrix[matrixIndex]) # parents now contains the fittest matrices, children will be made of this    
 
        # Reset children and parents
        children = generateChildren(parents)
        parentsAndChildren = parents + children
        
        counter += 1
    
    maxNum = max(onTheOriginOfLists)
    return maxNum

# ===========================================================================================================
# ===================================  Random Search  =======================================================
# ===========================================================================================================


def randomSearch1():
    """ Generate a random number of random matrices, check if they're feasible and get score. 
    
    A lot of these solutions are infeasible so randomSearch2() shows how difficult it is to randomly
    generate feasible matrices. 
    """
    randomMatrices = []
    scores = []
    # Number of matrices to make
    numIterations = random.randint(0, 100)
    print("We will be performing", numIterations, "iterations.\nCalculating...")
    # Only take feasible solutions
    count = 0
    while count < numIterations:
        potentialMatrix = [[random.randint(0,1) for i in range(x)] for j in range(y)]
       
        # Check for feasibility, then fitness
        if is_feasible(potentialMatrix) != -1:
            scores.append(fitness(potentialMatrix))
            randomMatrices.append(potentialMatrix)
        
        count += 1
            
    print("Of those", numIterations, "iterations", len(randomMatrices), "was/were viable solution(s)")
            
    return scores, randomMatrices

def randomSearch2():
    """ Calculating the time it takes to generate a certain number of random feasible matrices """
    randomMatrices = []
    scores = []
    # Number of matrices to make - this is the stopping condition determines how many feasible solutions we need to make
    numIterations = random.randint(0, 100)
    print("We will create", numIterations, "feasible matrices.\nCalculating...")
    # Only take feasible solutions
    count = 0
    start = round(time.clock(), 3)
    while len(randomMatrices) < numIterations:
        potentialMatrix = [[random.randint(0,1) for i in range(x)] for j in range(y)]
        
        # Check for feasibility, then fitness
        if is_feasible(potentialMatrix) != -1:
            scores.append(fitness(potentialMatrix))
            randomMatrices.append(potentialMatrix)
        
        else:
            count += 1
    end = round(time.clock(), 3)
    overallTime = end - start
    
    print("It took", overallTime, "seconds to generate", numIterations, "feasible matrices of size", x, "x", y)
    print("There were", count, "number of infeasible solutions")
    
    return scores, randomMatrices
    
# =======================================================================================================================
# =======================================  Run Algorithms Here  =========================================================
# =======================================================================================================================

#Run using the testMatrix
print(fitness(testMatrix)) # The score should be 462,500 when run through the fitness function
a, b = HillClimbing(testMatrix)
multipleClimbing(a,b)
print(simulatedAnnealing1(testMatrix))
print(simulatedAnnealing2())
parents = createParents()
generateChildren(parents)
print(naturalSelection())
randomSearch1()
randomSearch2()
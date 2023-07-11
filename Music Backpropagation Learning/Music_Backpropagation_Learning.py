import numpy as np
import copy


Music = [[0.3, -0.5, -0.5, 0.2, 0.1, -0.5, 0.2, -0.5],
         [0.3, -0.5, 0.3, -0.5, 0.3, -0.5, -0.5, -0.5],
         [0.2, -0.5, 0.2, -0.5, 0.2, -0.5, -0.5, -0.5],
         [0.3, -0.5, 0.3, -0.5, 0.3, -0.5, -0.5, -0.5],
         [0.3, -0.5, -0.5, 0.2, 0.1, -0.5, 0.2, -0.5],
         [0.3, -0.5, 0.3, -0.5, 0.3, -0.5, -0.5, -0.5],
         [0.2, -0.5, 0.2, -0.5, 0.3, -0.5, 0.2, -0.5],
         [0.1, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5],]
layer = [16, 50, 50, 50, 8]
speed = 0.1
MiniBatchExtractionNumber = 5

activeFunction              = lambda x: (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
activeFunction_Differential = lambda x: 1 - (x**2)
StoredOutput = [np.zeros(lay) for lay in layer]

Policy = [[(np.random.randn(layer[lay-1]*layer[lay]) * (1/np.sqrt(layer[lay-1]))), #--/2
            (np.random.randn(layer[lay]) * 0.01)] 
            for lay in range(1, len(layer))]

def ForwardPropagation(Policy, state):
    VariableOutput = [np.array(state)] + copy.deepcopy(StoredOutput[1:])
    
    for lay in range(1, len(VariableOutput)):
        for z in range(len(VariableOutput[lay])): VariableOutput[lay][z] = np.sum(VariableOutput[lay-1] * np.array([Policy[lay-1][0][(z+1)*(h+1)-1] for h in range(len(VariableOutput[lay-1]))])) + Policy[lay-1][1][z]
        VariableOutput[lay] = activeFunction(VariableOutput[lay])
    
    return VariableOutput

def BackPropagation(Policy, state, true_reward):
    VariableOutput = ForwardPropagation(Policy, state)

    VariableMiniBatch = [[np.zeros(layer[lay-1]*layer[lay]), np.zeros(layer[lay])] for lay in range(1, len(layer))]

    differentialOverlap = [(VariableOutput[-1][i] - true_reward[i]) * activeFunction_Differential(VariableOutput[-1][i]) for i in range(layer[-1])]
    zSizeList = [[l for l in range(i)] for i in layer]

    for lay in reversed(range(len(Policy))):
        VariableMiniBatch[lay] = [np.array([VariableOutput[lay][h] * differentialOverlap[z] for z in range(layer[lay+1]) for h in zSizeList[lay]]), differentialOverlap]
        differentialOverlap = np.array([np.sum(differentialOverlap * np.array([Policy[lay][0][(z+1)*(h+1)-1] for z in zSizeList[lay+1]])) * activeFunction_Differential(VariableOutput[lay][h]) for h in zSizeList[lay]])
    return VariableMiniBatch


MiniBatch = [[[np.zeros(layer[lay-1]*layer[lay]), np.zeros(layer[lay])] for lay in range(1, len(layer))] for i in range(MiniBatchExtractionNumber)]

for n in range(500):
    for number in range(MiniBatchExtractionNumber):
        startPoint = np.random.randint(2, len(Music))
        state = Music[startPoint-2]+Music[startPoint-1]
        MiniBatch[number] = BackPropagation(Policy, state, Music[startPoint])

    Policy = [[Policy[L][N] - (speed * np.array([np.mean([MiniBatch[M][L][N][B] for M in range(len(MiniBatch))]) for B in range(len(MiniBatch[0][L][N]))])) for N in range(2)] for L in range(len(MiniBatch[0]))]
    MiniBatch = [[[np.zeros(layer[lay-1]*layer[lay]), np.zeros(layer[lay])] for lay in range(1, len(layer))] for i in range(MiniBatchExtractionNumber)]
    
    startPoint = np.random.randint(2, len(Music))
    state = Music[startPoint-2]+Music[startPoint-1]
    print(f"{Music[startPoint]}\t{np.round(ForwardPropagation(Policy, state)[-1], 1)}")

for i in Policy:
    i[0] = i[0].tolist()
    i[1] = i[1].tolist()
print(Policy)
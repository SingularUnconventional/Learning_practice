import numpy        as np
import random
import math

OutputLayer = [0]
InputLayer = [0, 0]

layer = [4, 4]
A = 0.2

#함수의 정리-
def valueTheoremFunction():
    valueTheorem = []

    '''
    예
    한 계층 결과값       가중치             역치
    [
    [[54, 10]                                       ],
    [[0, 0],            [1, 1, 1, 1],       [2, 2]  ],
    [[0],               [1, 1],             [2]     ],
    [[0, 0],            [1, 1],             [2, 2]  ]
    ]
    '''

    def valueTheorem_(layer_,layer__):
        valueTheorem.append([[], [], []])

        for x in range(layer__):
            #각 계층 갯수 설정
            valueTheorem[layer_+1][0].append(0)

            #역치의 초기값 설정
            valueTheorem[layer_+1][2].append(np.random.randn()*0.001)
        
        #가중치의 초기값 설정
        for x in range(len(valueTheorem[layer_][0])*len(valueTheorem[layer_+1][0])):
            valueTheorem[layer_+1][1].append(np.random.randn() / math.sqrt(len(OutputLayer)))

    #입력층
    valueTheorem.append([InputLayer])

    #은닉층
    for lay in range(len(layer)):
        valueTheorem_(lay, layer[lay])

    #출력층
    valueTheorem_(lay+1, len(OutputLayer))

    return valueTheorem
valueTheorem = valueTheoremFunction()

def activeFunction(x): 
    #ReLU function
    #if x <= 0: return 0
    #else: return x

    #tanh
    #return (math.exp(x) - math.exp(-x)) / (math.exp(x) + math.exp(-x))

    #sigmoid function
    return 1 / (1 + math.exp(-x))

#활성함수 미분
def a_F_D(x): 
    #ReLU function
    #if x <= 0: return 0
    #else: return 1

    #tanh
    #return 1 - (x**2)

    #sigmoid function
    return x * (1 - x)

def ForwardPropagation():
    #계산해야 할 계층 수
    for lay in range(1, len(valueTheorem)):
        #한 계층의 계산결과
        for z in range(len(valueTheorem[lay][0])):
            #전 계층의 입력값, 혹은 계산결과
            for h in range(len(valueTheorem[lay-1][0])):
                valueTheorem[lay][0][z] += valueTheorem[lay-1][0][h] * valueTheorem[lay][1][(z+1)*(h+1)-1]

            #역치 계산
            #valueTheorem[lay][0][z] += valueTheorem[lay][2][z]

            #활성함수
            valueTheorem[lay][0][z] = activeFunction(valueTheorem[lay][0][z])
            
def BackPropagation(InputLayer, OutputLayer, reward):
    differentialOverlap = [0]

    #입력층 값 대입
    valueTheorem[0][0] = InputLayer
           
    #순전파
    ForwardPropagation()

    #결과값의 미분
    for z in range(1):
        differentialOverlap[z] = valueTheorem[-1][0][z] - OutputLayer[z]

    #계산해야 할 계층 수
    for lay in reversed(range(len(valueTheorem)-1)):

        #계산완료된 미분값을 초기화
        differentialOverlap_c = []
        for x in range(len(valueTheorem[lay][0])): differentialOverlap_c.append(0)

        for z in range(len(valueTheorem[lay+1][0])):
            #현 계층의 중복되는 미분값
            Ed = differentialOverlap[z] * a_F_D(valueTheorem[lay+1][0][z])

            #역치 값 수정
            #valueTheorem[lay+1][2][z] -= A * Ed

            #현재 계층의 입력값, 혹은 계산결과
            for h in range(len(valueTheorem[lay][0])):

                #다음 계층의 중복되는 미분값 계산
                differentialOverlap_c[h] += Ed * valueTheorem[lay+1][1][(z+1)*(h+1)-1]

                #가중치 값 수정
                valueTheorem[lay+1][1][(z+1)*(h+1)-1] -= reward * A *Ed * valueTheorem[lay][0][h]

        #중복되는 미분값을 다음 계층으로 전달
        differentialOverlap = differentialOverlap_c


repeatCount = 100000

reward = 0

result = 0
while True:
    valueTheorem = valueTheoremFunction()

    for i in range(repeatCount):
        InputLayer = [random.randint(0, 1), random.randint(0, 1)]
        OutputLayer = [random.random()]

        if (InputLayer[0] == 1 and InputLayer[1] == 1) or (InputLayer[0] == 0 and InputLayer[1] == 0): result = 0
        if (InputLayer[0] == 0 and InputLayer[1] == 1) or (InputLayer[0] == 1 and InputLayer[1] == 0): result = 1

        reward = 0.7-np.abs(OutputLayer[0] - result)

        BackPropagation(InputLayer, OutputLayer, reward)

    print(valueTheorem)
    V = [[0, 0], [1, 1], [1, 0], [0, 1]]
    for x in V:
        valueTheorem[0][0] = x
        ForwardPropagation()
        print(f"{x}\t{valueTheorem[-1][0]}")



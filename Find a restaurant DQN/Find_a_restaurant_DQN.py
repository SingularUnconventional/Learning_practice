import numpy    as np
import copy
import random
import tkinter
import threading

#모든 좌표는 y x 순서

place_Y, place_X = 20, 20

food        = [10, 10]
reward      = 0.5
unreward    = -0.1
RewardBias  = 0.1

DiscountFactor = 0.2

#이 횟수만큼 진행하면 값이 0.5가 되는 횟수
percentageA = 2000
speedA = 10

MaxA = 1

layer = [2, 8, 8, 4]

TimeLimit = 100

MiniBatchVolume = 5000
MiniBatchExtractionNumber = 500

RewardGraph = np.array([[(-(((food[0]-y)**2)**RewardBias)-(((food[1]-x)**2)**RewardBias)).real for x in range(place_X)]for y in range(place_Y)])
RewardGraph = ((reward - unreward) / (-1 - np.min(RewardGraph)) * (RewardGraph - np.min(RewardGraph)))+unreward
for y in range(20): 
    for x in range(20): print('{:.1f}'.format(RewardGraph[y][x]), end='\t')
    print()

clearCount = 0
Time = 0
percentage = 1
activeFunction              = lambda x: (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))#np.where(x <= 0, x*0.1, x)
activeFunction_Differential = lambda x: 1 - (x**2)#np.where(x <= 0, 0.1, 1)
#x가 최소 18은 넘어야 반올림으로 1 출력, 710에서 오버플로우 발생
StoredOutput = [np.zeros(lay) for lay in layer]
Policy = [[(np.random.randn(layer[lay-1]*layer[lay]) * (1/np.sqrt(layer[lay-1])/2)), #--/2
           (np.random.randn(layer[lay]) * 0.01)] 
          for lay in range(1, len(layer))]
print(Policy)

MiniBatchSample = [[] for i in range(MiniBatchVolume)]
MiniBatch = [[[np.zeros(layer[lay-1]*layer[lay]), np.zeros(layer[lay])] for lay in range(1, len(layer))] for i in range(MiniBatchExtractionNumber)]


def ForwardPropagation(state):
    VariableOutput = [np.array(state)/20] + copy.deepcopy(StoredOutput[1:])

    for lay in range(1, len(VariableOutput)):
        for z in range(len(VariableOutput[lay])): VariableOutput[lay][z] = np.sum(VariableOutput[lay-1] * np.array([Policy[lay-1][0][(z+1)*(h+1)-1] for h in range(len(VariableOutput[lay-1]))])) + Policy[lay-1][1][z]
        VariableOutput[lay] = activeFunction(VariableOutput[lay])

    return VariableOutput

def BackPropagation(state, true_reward, reward_node):
    VariableOutput = ForwardPropagation(state)

    VariableMiniBatch = [[np.zeros(layer[lay-1]*layer[lay]), np.zeros(layer[lay])] for lay in range(1, len(layer))]

    differentialOverlap = np.zeros(layer[-1])
    differentialOverlap[reward_node] = (VariableOutput[-1][reward_node] - true_reward) * activeFunction_Differential(VariableOutput[-1][reward_node])

    zSizeList = [[l for l in range(i)] for i in layer[:-1]]+[[reward_node]]

    for lay in reversed(range(len(Policy))):
        VariableMiniBatch[lay] = [np.array([VariableOutput[lay][h] * differentialOverlap[z] for z in range(layer[lay+1]) for h in zSizeList[lay]]), differentialOverlap]#[0 for h in zSizeList[lay+1]]

        differentialOverlap = np.array([np.sum(differentialOverlap * np.array([Policy[lay][0][(z+1)*(h+1)-1] for z in zSizeList[lay+1]])) for h in zSizeList[lay]])

    return VariableMiniBatch



import time
def time_check(function, number):
    start = time.time()
    start_ = time.process_time()

    for i in range(number-1): function()#(np.array([0, 0]))
    a = function()#(np.array([0, 0]))

    end = time.time()
    end_ = time.process_time()

    print(function.__name__)
    print("time:        ", end - start)
    print("process time:", end_ - start_)
    print("output:", a)
    print()

#time_check(100, ForwardPropagation)

#MiniBatch[episode] = BackPropagation_(1, 2)
#for lay in range(len(Policy)):
#    Policy[lay][0] = Policy[lay][0] - (A * np.array(MiniBatch[episode][lay][0]))
#    Policy[lay][1] = Policy[lay][1] - (A * np.array(MiniBatch[episode][lay][1]))


#상황과 행동이 주어지면 보상과 다음 행동 출력
def Game(state, action):
    
    S = []

    if      action == 0 and state[0] != 0:          S = [state[0] - 1, state[1]]
    elif    action == 1 and state[1] != place_X-1:  S = [state[0], state[1] + 1]
    elif    action == 2 and state[0] != place_Y-1:  S = [state[0] + 1, state[1]]
    elif    action == 3 and state[1] != 0:          S = [state[0], state[1] - 1]
    else:                                           S = state

    if S==food: return RewardGraph[S[0]][S[1]], None
    else:       return RewardGraph[S[0]][S[1]], S


def sampling():
    if np.random.choice([True, False], p = [percentage, 1-percentage]): action = np.random.randint(4)
    else: action = np.argmax(ForwardPropagation(state)[-1])

    reward, next_state = Game(state, action)

    return next_state, [state, action, reward, next_state]


def main_():
    def Color(color, power):
        color_ = color
        color__ = ['', '', '']
        power_ = 255 * (1 - power)
        for i in range(3):
            if color_[i] == 0: color_[i] = power_
            color__[i] = format(int(color_[i]), '02x')
        return "#" + "".join(color__)

    def E():
        place = [[[[], 0] for x in range(place_X)] for y in range(place_Y)]
        Range = [0 for i in range(place_Y * place_X)]
        
        while True:
            Range_ = []
            LA = [[ForwardPropagation([y, x])[-1] for x in range(place_X)] for y in range(place_Y)]

            for y in range(place_Y):
                for x in range(place_X):
                    if [y, x] in [food]: place[y][x] = [[255, 255, 255], 0]
                    else:
                        L = np.argmax(LA[y][x])
                        A = (max(LA[y][x]) - np.mean(LA[y][x]))
                        Range[y*place_X+x] = A
                        for i in range(4): Range_.append(LA[y][x][i])
                        if      L == 0: place[y][x] = [[255,  0,   0], A]
                        elif    L == 1: place[y][x] = [[255,255,   0], A]
                        elif    L == 2: place[y][x] = [[0,    0, 255], A]
                        elif    L == 3: place[y][x] = [[  0,255,   0], A]

            Min = np.min(Range)
            Max = np.max(Range)
            label.config(text="Max: {:.3f} Min: {:.3f}  p: {:.2f}%  a: {:.4f}  {}".format(np.max(Range_), np.min(Range_), percentage*100, speed, episode))

            Range_ = lambda A: 1 / (Max - Min) * (A - Min)

            canvas.delete("all")
            
            for y in range(place_Y):
                for x in range(place_X):
                    color = Color(place[y][x][0], Range_(place[y][x][1]))
                    P = canvas.create_rectangle(x*20, y*20, x*20+20, y*20+20, fill=color, outline=color)
            time.sleep(5)

    win=tkinter.Tk()
    win.title("키 입력")
    win.resizable(False, False)

    label=tkinter.Label(win, text="")
    label.pack()

    canvas=tkinter.Canvas(win, width = place_X*20, height = place_Y*20, bg = '#001133')

    P = [[canvas.create_rectangle(x*20, y*20, x*20+20, y*20+20, fill='#000000', outline='#000000') for x in range(place_X)] for y in range(place_Y)]

    start = threading.Thread(target=E)
    start.start()

    canvas.pack()
    win.mainloop()




while MiniBatchSample[0] == []:
    state = [random.randint(0, place_Y-1), random.randint(0, place_X-1)]
    while state is not None:
        state, MiniBatchSample_ = sampling()
        MiniBatchSample = MiniBatchSample[1:] + [MiniBatchSample_]


main_start = threading.Thread(target=main_)
main_start.start()

for episode in range(1000000):
    state = [random.randint(0, place_Y-1), random.randint(0, place_X-1)]
    if Time is TimeLimit: print(clearCount, end='')
    else: 
        print("⬛", end='')
        clearCount += 1
    Time = 0
    print("\t{}\t{}\t{}\t{}".format(episode, state, ForwardPropagation(state)[-1], np.random.choice(Policy[-1][0], 1, False)))

    percentage = 1/((episode/percentageA)+1)
    speed = MaxA/((episode/speedA)+1)

    while not state is None and Time != TimeLimit:
        state, MiniBatchSample_ = sampling()
        MiniBatchSample = MiniBatchSample[1:] + [MiniBatchSample_]

        number = 0
        for D in random.sample(MiniBatchSample, MiniBatchExtractionNumber):
            if D[-1] is None: y = D[2]
            else: y = D[2] + (DiscountFactor*np.max(ForwardPropagation(D[3])[-1]))

            MiniBatch[number] = BackPropagation(D[0], y, D[1])

            number+=1

        Policy = [[Policy[L][N] - (speed * np.array([np.mean([MiniBatch[M][L][N][B] for M in range(len(MiniBatch))]) for B in range(len(MiniBatch[0][L][N]))])) for N in range(2)] for L in range(len(MiniBatch[0]))]
        
        MiniBatch = [[[np.zeros(layer[lay-1]*layer[lay]), np.zeros(layer[lay])] for lay in range(1, len(layer))] for i in range(MiniBatchExtractionNumber)]

        Time += 1
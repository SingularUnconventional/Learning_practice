import numpy    as np
import copy
import random
import tkinter
import threading

import pygame


class variable:
    def __init__(self):
        self.hungerLimit = 10
        self.reward      = 0.5
        self.unreward    = -0.001
        self.loseReward  = -0.1

        self.DiscountFactor = 0.9

        #이 횟수만큼 진행하면 값이 0.5가 되는 횟수
        self.percentageA = 50
        self.speedA = 2
        
        self.MaxA = 3
Variable = variable()

#모든 좌표는 y x 순서

place_Y, place_X = 10, 10

layer = [6, 12, 12, 3]

MiniBatchVolume = 5000
MiniBatchExtractionNumber = 500


activeFunction              = lambda x: (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
activeFunction_Differential = lambda x: 1 - (x**2)
StoredOutput = [np.zeros(lay) for lay in layer]


class Game:
    def __init__(self):
        self.Policy = [[(np.random.randn(layer[lay-1]*layer[lay]) * (1/np.sqrt(layer[lay-1])/2)), #--/2
                        (np.random.randn(layer[lay]) * 0.01)] 
                        for lay in range(1, len(layer))]

    def reset(self):
        self.currentDirection   = 1
        self.Head               = [random.randint(1, place_Y-2), random.randint(1, place_X-2)]
        self.Food               = [random.randint(0, place_Y-1), random.randint(0, place_X-1)]
        self.tail               = []
        self.score              = 0
        self.hunger             = 0
        
        self.place              = [[0 for x in range(place_X)] for y in range(place_Y)]
        self.place[self.Food[0]][self.Food[1]] = 1

    def State(self):
        def distance_measurement_UP():
            for i in reversed(range(self.Head[0])):#^
                if self.place[i][self.Head[1]] == 2: return (self.Head[0] - i)/10

            return (self.Head[0])/10
        def distance_measurement_LEFT():
            for i in reversed(range(self.Head[1])):#<
                if self.place[self.Head[0]][i] == 2: return (self.Head[1] - i)/10

            return (self.Head[1])/10
        def distance_measurement_DOWN():
            for i in range(self.Head[0], place_Y):#∨
                if self.place[i][self.Head[1]] == 2: return (i - self.Head[0])/10

            return (place_Y - self.Head[0] -1)/10
        def distance_measurement_RIGHR():
            for i in range(self.Head[1], place_X):#>
                if self.place[self.Head[0]][i] == 2: return (i - self.Head[1])/10

            return (place_X - self.Head[1] -1)/10

        distance_measurement = [distance_measurement_LEFT(),
                                distance_measurement_UP(),
                                distance_measurement_RIGHR(),
                                distance_measurement_DOWN()]


        state = [distance_measurement[i%4] for i in range(self.currentDirection, self.currentDirection+3)]
        
        Food_list = [[1, -1, -1, 0, 1], [0, 1, -1, 0, 1], [1, 1, 1, 0, -1], [0, -1, 1, 0, -1]]

        for i in range(4):
            if self.currentDirection == i:
                if np.sign(self.Food[(Food_list[i][0]-1)**2] - self.Head[(Food_list[i][0]-1)**2]) == Food_list[i][1]:
                    if      np.sign(self.Food[Food_list[i][0]] - self.Head[Food_list[i][0]]) == Food_list[i][2]: return state + [1, 0, 0]
                    elif    np.sign(self.Food[Food_list[i][0]] - self.Head[Food_list[i][0]]) == Food_list[i][3]: return state + [0, 1, 0]
                    elif    np.sign(self.Food[Food_list[i][0]] - self.Head[Food_list[i][0]]) == Food_list[i][4]: return state + [0, 0, 1]
                else: return state + [0, 0, 0]

    #상황과 행동이 주어지면 보상과 다음 상황 출력(보이는 상황과 보이지 않는 상황 모두)
    def Game_(self, action):
        reward_ = Variable.unreward

        #상황추가

        #머리부분을 꼬리로 바꾸고, 꼬리 추가
        self.place[self.Head[0]][self.Head[1]] = 2
        self.tail = self.tail + [[self.Head[0], self.Head[1]]]

        self.currentDirection = (self.currentDirection + action)%4

        #방향에 따른 이동, 막힐 시 아웃
        if   self.currentDirection == 0:
            if self.Head[0] == 0:        return Variable.loseReward, None
            else: self.Head[0]-=1
        elif self.currentDirection == 1:
            if self.Head[1] == place_X-1:return Variable.loseReward, None
            else: self.Head[1]+=1
        elif self.currentDirection == 2:
            if self.Head[0] == place_Y-1:return Variable.loseReward, None
            else: self.Head[0]+=1
        elif self.currentDirection == 3:
            if self.Head[1] == 0:        return Variable.loseReward, None
            else: self.Head[1]-=1
        
        next_state = self.State()

        #음식을 먹을 시 점수 추가와 보상, 다시 음식 생성
        if self.place[self.Head[0]][self.Head[1]] == 1: 
            self.score += 1
            reward_ = Variable.reward
            self.hunger = 0

            #머리에 생성이 되면 다시 반복
            while True:
                x = random.randint(0, place_X-1)
                y = random.randint(0, place_Y-1)
                if self.Head[0] != y or self.Head[1] != x:
                    self.Food = [y, x]
                    self.place[y][x] = 1
                    break

        #꼬리에 닿을 시 아웃
        elif self.place[self.Head[0]][self.Head[1]] == 2: return Variable.loseReward, None

        #꼬리 끝 지우기
        else: 
            if self.place[self.tail[0][0]][self.tail[0][1]] != 1: self.place[self.tail[0][0]][self.tail[0][1]] = 0
            self.tail = self.tail[1:]

            if self.hunger == Variable.hungerLimit * (self.score+3): return Variable.unreward, None
            self.hunger += 1

        #머리 표시
        self.place[self.Head[0]][self.Head[1]] = 3
        

        #시각화
        #print(f'\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n{self.score}\t{self.Head[2]}\t{count}\t{self.currentDirection}\t{self.SituationsAndActions[-1][0]}')
        #for y in range(self.place_width):
        #    for x in range(self.place_height):
        #        print(self.place[y][x], end = '\t')
        #    print()
        #    print()
        #    print()

        return reward_, next_state


def ForwardPropagation(Policy, state):
    VariableOutput = [np.array(state)] + copy.deepcopy(StoredOutput[1:])

    for lay in range(1, len(VariableOutput)):
        for z in range(len(VariableOutput[lay])): VariableOutput[lay][z] = np.sum(VariableOutput[lay-1] * np.array([Policy[lay-1][0][(z+1)*(h+1)-1] for h in range(len(VariableOutput[lay-1]))])) + Policy[lay-1][1][z]
        VariableOutput[lay] = activeFunction(VariableOutput[lay])

    return VariableOutput

def BackPropagation(Policy, state, true_reward, reward_node):
    VariableOutput = ForwardPropagation(Policy, state)

    VariableMiniBatch = [[np.zeros(layer[lay-1]*layer[lay]), np.zeros(layer[lay])] for lay in range(1, len(layer))]

    differentialOverlap = np.zeros(layer[-1])
    differentialOverlap[reward_node] = (VariableOutput[-1][reward_node] - true_reward) * activeFunction_Differential(VariableOutput[-1][reward_node])

    zSizeList = [[l for l in range(i)] for i in layer[:-1]]+[[reward_node]]

    for lay in reversed(range(len(Policy))):
        VariableMiniBatch[lay] = [np.array([VariableOutput[lay][h] * differentialOverlap[z] for z in range(layer[lay+1]) for h in zSizeList[lay]]), differentialOverlap]

        differentialOverlap = np.array([np.sum(differentialOverlap * np.array([Policy[lay][0][(z+1)*(h+1)-1] for z in zSizeList[lay+1]])) * activeFunction_Differential(VariableOutput[lay][h]) for h in zSizeList[lay]])

    return VariableMiniBatch


def sampling(state, Policy, percentage, function):
    if np.random.choice([True, False], p = [percentage, 1-percentage]): action = np.random.randint(-1, 2)
    else: action = np.argmax(ForwardPropagation(Policy, state)[-1])-1

    reward, next_state = function(action)
    return next_state, [state, action, reward, next_state]

def visualize():
    percentage = 1

    
    print(Learning.Policy)

    MiniBatchSample = [[] for i in range(MiniBatchVolume)]
    MiniBatch = [[[np.zeros(layer[lay-1]*layer[lay]), np.zeros(layer[lay])] for lay in range(1, len(layer))] for i in range(MiniBatchExtractionNumber)]


    while MiniBatchSample[0] == []:
        Learning.reset()
        state = Learning.State()

        while state is not None:
            state, MiniBatchSample_ = sampling(state, Learning.Policy, percentage, Learning.Game_)
            MiniBatchSample = MiniBatchSample[1:] + [MiniBatchSample_]

    episode = 0
    while True:
        print(Learning.score, end="  ")

        Learning.reset()
        state = Learning.State()

        print("{}\t{} {}\t{}".format(episode, state, ForwardPropagation(Learning.Policy, state)[-1], np.random.choice(Learning.Policy[-1][0], 1, False)))

        percentage = 1/((episode/Variable.percentageA)+1)
        speed = Variable.MaxA/((episode/Variable.speedA)+1)

        while not state is None:
            state, MiniBatchSample_ = sampling(state, Learning.Policy, percentage, Learning.Game_)
            MiniBatchSample = MiniBatchSample[1:] + [MiniBatchSample_]
            number = 0
            for D in random.sample(MiniBatchSample, MiniBatchExtractionNumber):
                if D[-1] is None: y = D[2]
                else: y = D[2] + (Variable.DiscountFactor*np.max(ForwardPropagation(Learning.Policy, D[3])[-1]))

                MiniBatch[number] = BackPropagation(Learning.Policy, D[0], y, D[1])

                number+=1

            Learning.Policy = [[Learning.Policy[L][N] - (speed * np.array([np.mean([MiniBatch[M][L][N][B] for M in range(len(MiniBatch))]) for B in range(len(MiniBatch[0][L][N]))])) for N in range(2)] for L in range(len(MiniBatch[0]))]
        
            MiniBatch = [[[np.zeros(layer[lay-1]*layer[lay]), np.zeros(layer[lay])] for lay in range(1, len(layer))] for i in range(MiniBatchExtractionNumber)]

        episode+=1

#신경망 시각화, 가중치 2차원 이미지 시각화, 실시간으로 변수 조정 가능하게, 만들어진 정책 txt파일로 저장 등등
def learning():
    NNY = 270
    NNX = 150

    NNW = 200
    NNH = 40

    NNS = 15


    pygame.init()

    size = (1000, 500)
    screen = pygame.display.set_mode(size)
    pygame.display.set_caption("Game")
    clock = pygame.time.Clock()
    color = [(10, 10, 10), (255, 255, 129), (0, 102, 153), (0, 153, 204)]

    
    while True:
        
        #pygame.draw.rect(screen, color[0], [0, 0, 200, 200])

        Visualize.reset()
        state = Visualize.State()

        while state is not None:
            clock.tick(5)
            screen.fill((0, 0, 0))

            VariableOutput = ForwardPropagation(Learning.Policy, state)

            for y in range(place_Y):
                for x in range(place_X):
                    pygame.draw.rect(screen, color[Visualize.place[y][x]], [x*20, y*20, 18, 18])

            
            nodeMin = np.min([x for y in VariableOutput for x in y])
            nodeMax = np.max([x for y in VariableOutput for x in y])

            Policy_ = [x for y in sum(Learning.Policy, []) for x in y]
            PolicyMin = np.min(Policy_)
            PolicyMax = np.max(Policy_)

            Range = lambda A, L, Max, Min: L / (Max - Min) * (A - Min)

            for lay in range(1, len(layer)):
                for z in range(layer[lay]): 
                    for h in range(layer[lay-1]): 
                        Policy_w = Learning.Policy[lay-1][0][(z+1)*(h+1)-1]
                        Range_w = Range(Policy_w, 200, PolicyMax, PolicyMin)

                        if Policy_w < 0:lineColor = (Range_w, int(Range_w/3), int(Range_w/4))
                        else:           lineColor = (int(Range_w/4), int(Range_w/3), Range_w)
                        pygame.draw.line(screen, lineColor, ((lay*NNW)+NNW+NNX, (z*NNH - (layer[lay]*NNH/2))+NNY), (((lay-1)*NNW)+NNW+NNX, (h*NNH - (layer[lay-1]*NNH/2))+NNY), int(Range(Policy_w, 3, PolicyMax, PolicyMin)))

            for lay in range(len(layer)-1): 
                for node in range(layer[lay]): 
                    nodeColor = Range(VariableOutput[lay][node], 255, 1, -1)
                    pygame.draw.circle(screen, (255, 255, 255), [(lay*NNW)+NNW+NNX, (node*NNH - (layer[lay]*NNH/2))+NNY], NNS)
                    pygame.draw.circle(screen, (nodeColor, nodeColor, nodeColor), [(lay*NNW)+NNW+NNX, (node*NNH - (layer[lay]*NNH/2))+NNY], NNS-1)

            for node in range(layer[(len(layer)-1)]): 
                nodeColor = Range(VariableOutput[(len(layer)-1)][node], 255, 1, -1)
                pygame.draw.circle(screen, (200, 200, 200), [((len(layer)-1)*NNW)+NNW+NNX, (node*NNH - (layer[(len(layer)-1)]*NNH/2))+NNY], NNS)
                pygame.draw.circle(screen, (nodeColor, nodeColor, nodeColor), [((len(layer)-1)*NNW)+NNW+NNX, (node*NNH - (layer[(len(layer)-1)]*NNH/2))+NNY], NNS-1)
                if np.max(VariableOutput[(len(layer)-1)]) == VariableOutput[(len(layer)-1)][node]: pygame.draw.circle(screen, color[3], [((len(layer)-1)*NNW)+NNW+NNX, (node*NNH - (layer[(len(layer)-1)]*NNH/2))+NNY], NNS-1)


            state, MiniBatchSample_ = sampling(state, Learning.Policy, 0, Visualize.Game_)
            
            for event in pygame.event.get():
                if event.type==pygame.KEYDOWN:
                    if event.key==pygame.K_p:
                        print(Learning.Policy)

                if event.type==pygame.QUIT:
                    pygame.quit()
                    sys.exit()
            pygame.display.update()

    pygame.quit()

Learning  = Game()
Visualize = Game()

visualizeStart = threading.Thread(target=visualize)
visualizeStart.start()

learningStart = threading.Thread(target=learning)
learningStart.start()


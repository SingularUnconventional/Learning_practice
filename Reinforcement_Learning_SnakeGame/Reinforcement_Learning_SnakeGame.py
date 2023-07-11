import numpy        as np
import threading
import tkinter
import time
import math
import random

AdventureProbability = 100  #모험률
DiscountFactor = 0.8        #보상감가율


layer = [6, 6]
A = 0.1 #속도

foodDiscount = 1        #음식보상
delayDiscount = -0.005   #딜레이보상
endDiscount = -0.01        #탈락보상

place_height= 10
place_width = 10

count = 0

speed = 0.1


class Game:
    def __init__(self):
        
        self.valueTheorem = self.valueTheoremFunction()

        self.score = 0
        self.SituationsAndActions = []
        self.InGame = True

        #초기 뱀의 위치와 방향값
        self.Head = [random.randint(1, place_height-2), random.randint(1, place_width-2), [0.4, 0.7, 0.4]]
        self.tail = []
        self.tail_len = 0
        self.currentDirection = 1

        self.Food = [random.randint(0, place_height-1), random.randint(0, place_height-1)]

        #판 설정과 음식 설정
        self.place=[[1 for x in range(place_width)] for y in range(place_height)]
        self.place[self.Food[0]][self.Food[1]] = 2


    #함수의 정리-
    def valueTheoremFunction(self):
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
                valueTheorem[layer_+1][1].append(np.random.randn() / math.sqrt(3))

        #입력층
        valueTheorem.append([[0 for i in range(place_height*place_width)]])

        #은닉층
        for lay in range(len(layer)):
            valueTheorem_(lay, layer[lay])

        #출력층
        valueTheorem_(lay+1, 3)

        return valueTheorem
    #------------


    def ForwardPropagation(self):
        #계산해야 할 계층 수
        for lay in range(1, len(self.valueTheorem)):
            #한 계층의 계산결과
            for z in range(len(self.valueTheorem[lay][0])):
                #전 계층의 입력값, 혹은 계산결과
                for h in range(len(self.valueTheorem[lay-1][0])):
                    self.valueTheorem[lay][0][z] += self.valueTheorem[lay-1][0][h] * self.valueTheorem[lay][1][(z+1)*(h+1)-1]

                #역치 계산
                #self.valueTheorem[lay][0][z] += self.valueTheorem[lay][2][z]

                #활성함수
                self.valueTheorem[lay][0][z] = activeFunction(self.valueTheorem[lay][0][z])
            
    def BackPropagation(self, InputLayer, OutputLayer, reward):
        differentialOverlap = [0, 0, 0]

        #입력층 값 대입
        self.valueTheorem[0][0] = InputLayer
           
        #순전파
        self.ForwardPropagation()

        #결과값의 미분
        for z in range(3):
            differentialOverlap[z] = self.valueTheorem[-1][0][z] - OutputLayer[z]

        #계산해야 할 계층 수
        for lay in reversed(range(len(self.valueTheorem)-1)):

            #계산완료된 미분값을 초기화
            differentialOverlap_c = []
            for x in range(len(self.valueTheorem[lay][0])): differentialOverlap_c.append(0)

            for z in range(len(self.valueTheorem[lay+1][0])):
                #현 계층의 중복되는 미분값
                Ed = differentialOverlap[z] * a_F_D(self.valueTheorem[lay+1][0][z])

                #역치 값 수정
                #self.valueTheorem[lay+1][2][z] -= A * Ed

                #현재 계층의 입력값, 혹은 계산결과
                for h in range(len(self.valueTheorem[lay][0])):

                    #다음 계층의 중복되는 미분값 계산
                    differentialOverlap_c[h] += Ed * self.valueTheorem[lay+1][1][(z+1)*(h+1)-1]

                    #가중치 값 수정
                    self.valueTheorem[lay+1][1][(z+1)*(h+1)-1] -= reward * A *Ed * self.valueTheorem[lay][0][h]

            #중복되는 미분값을 다음 계층으로 전달
            differentialOverlap = differentialOverlap_c

    #모험
    def Adventure(self):
        self.Head[2] = [0.4, 0.4, 0.4]
        self.Head[2][random.randint(0, 2)] = 0.7

    #경험
    def experience(self):
        self.Head[2] = [0.4, 0.4, 0.4]

        self.ForwardPropagation()

        Head_direction = sorted([[np.abs(0.7 - self.valueTheorem[-1][0][i]), self.valueTheorem[-1][0][i]] for i in range(3)])
        for i in range(3):
            if Head_direction[0][1] == self.valueTheorem[-1][0][i]:
                self.Head[2][i] = 0.7
                break

    def SnakeGame(self):

        i = 0
        def distance_measurement_UP(l):
            global i
            for i in reversed(range(self.Head[0])):#^
                if self.place[i][self.Head[1]] == 0:
                    self.SituationsAndActions[-1][0][l] = (self.Head[0] - i)/10
                    i = -1
                    break
            if i != -1: self.SituationsAndActions[-1][0][l] = (self.Head[0])/10
        def distance_measurement_LEFT(l):
            global i
            for i in reversed(range(self.Head[1])):#<
                if self.place[self.Head[0]][i] == 0:
                    self.SituationsAndActions[-1][0][l] = (self.Head[1] - i)/10
                    i = -1
                    break
            if i != -1: self.SituationsAndActions[-1][0][l] = (self.Head[1])/10
        def distance_measurement_DOWN(l):
            global i
            for i in range(self.Head[0], place_height):#∨
                if self.place[i][self.Head[1]] == 0:
                    self.SituationsAndActions[-1][0][l] = (i - self.Head[0])/10
                    i = -1
                    break
            if i != -1: self.SituationsAndActions[-1][0][l] = (place_height - self.Head[0] -1)/10
        def distance_measurement_RIGHR(l):
            global i
            for i in range(self.Head[1], place_width):#>
                if self.place[self.Head[0]][i] == 0:
                    self.SituationsAndActions[-1][0][l] = (i - self.Head[1])/10
                    i = -1
                    break
            if i != -1: 
                self.SituationsAndActions[-1][0][l] = (place_width - self.Head[1] -1)/10

        
            
        #상황추가

        self.SituationsAndActions.append([[0, 0, 0, 0, 0, 0], self.Head[2], delayDiscount])

        if   self.currentDirection == 1:
            distance_measurement_LEFT(0)
            distance_measurement_UP(1)
            distance_measurement_RIGHR(2)
            if self.Food[0] < self.Head[0]:
                if      self.Food[1] <  self.Head[1]: self.SituationsAndActions[-1][0][3] = 1
                elif    self.Food[1] == self.Head[1]: self.SituationsAndActions[-1][0][4] = 1
                elif    self.Food[1] >  self.Head[1]: self.SituationsAndActions[-1][0][5] = 1
        elif self.currentDirection == 2:
            distance_measurement_UP(0)
            distance_measurement_RIGHR(1)
            distance_measurement_DOWN(2)
            if self.Food[1] > self.Head[1]:
                if      self.Food[0] <  self.Head[0]: self.SituationsAndActions[-1][0][3] = 1
                elif    self.Food[0] == self.Head[0]: self.SituationsAndActions[-1][0][4] = 1
                elif    self.Food[0] >  self.Head[0]: self.SituationsAndActions[-1][0][5] = 1
        elif self.currentDirection == 3:
            distance_measurement_RIGHR(0)
            distance_measurement_DOWN(1)
            distance_measurement_LEFT(2)
            if self.Food[0] > self.Head[0]:
                if      self.Food[1] >  self.Head[1]: self.SituationsAndActions[-1][0][3] = 1
                elif    self.Food[1] == self.Head[1]: self.SituationsAndActions[-1][0][4] = 1
                elif    self.Food[1] <  self.Head[1]: self.SituationsAndActions[-1][0][5] = 1
        elif self.currentDirection == 4:
            distance_measurement_DOWN(0)
            distance_measurement_LEFT(1)
            distance_measurement_UP(2)
            if self.Food[1] < self.Head[1]:
                if      self.Food[0] >  self.Head[0]: self.SituationsAndActions[-1][0][3] = 1
                elif    self.Food[0] == self.Head[0]: self.SituationsAndActions[-1][0][4] = 1
                elif    self.Food[0] <  self.Head[0]: self.SituationsAndActions[-1][0][5] = 1


        #머리부분을 꼬리로 바꾸고, 꼬리 추가
        self.place[self.Head[0]][self.Head[1]] = 0
        self.tail.append([self.Head[0], self.Head[1]])

        for i in range(3): self.currentDirection += np.round(self.Head[2][i])*(i+1)
        self.currentDirection = (self.currentDirection - 2)%4
        if self.currentDirection == 0: self.currentDirection = 4

        #방향에 따른 이동, 막힐 시 아웃
        if   self.currentDirection == 1:
            if self.Head[0] == 0:                self.InGame = False
            else: self.Head[0]-=1
        elif self.currentDirection == 2:
            if self.Head[1] == place_height-1:   self.InGame = False
            else: self.Head[1]+=1
        elif self.currentDirection == 3:
            if self.Head[0] == place_width-1:    self.InGame = False
            else: self.Head[0]+=1
        elif self.currentDirection == 4:
            if self.Head[1] == 0:                self.InGame = False
            else: self.Head[1]-=1
        

        #음식을 먹을 시 점수 추가와 보상, 다시 음식 생성
        if self.place[self.Head[0]][self.Head[1]] == 2: 
            self.score += 1
            self.SituationsAndActions[-1][2] += foodDiscount*self.score

            #머리에 생성이 되면 다시 반복
            while True:
                x = random.randint(0, place_height-1)
                y = random.randint(0, place_height-1)
                if self.Head[0] != y or self.Head[1] != x:
                    self.Food = [y, x]
                    self.place[y][x] = 2
                    break

        #꼬리에 닿을 시 아웃
        elif self.place[self.Head[0]][self.Head[1]] == 0: self.InGame = False

        #꼬리 끝 지우기
        else: 
            if self.place[self.tail[0][0]][self.tail[0][1]] != 2: self.place[self.tail[0][0]][self.tail[0][1]] = 1
            del self.tail[0]

        #머리 표시
        self.place[self.Head[0]][self.Head[1]] = 8
        

        #시각화
        #print(f'\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n{self.score}\t{self.Head[2]}\t{count}\t{self.currentDirection}\t{self.SituationsAndActions[-1][0]}')
        #for y in range(self.place_width):
        #    for x in range(self.place_height):
        #        print(self.place[y][x], end = '\t')
        #    print()
        #    print()
        #    print()

    #아웃 시 초기화
    def end(self):
        #벌점
        self.SituationsAndActions[-1][2] = endDiscount

        #가감률에 따라 보상이 가감되며 전도
        for i in reversed(range(len(self.SituationsAndActions)-1)):
            self.SituationsAndActions[i][2] += self.SituationsAndActions[i + 1][2] * DiscountFactor
        self.tail_len = len(self.tail)
        #초기화
        self.place=[[1 for x in range(place_width)] for y in range(place_height)]

        self.Food = [random.randint(0, place_height-1), random.randint(0, place_height-1)]
        self.place[self.Food[0]][self.Food[1]] = 2

        self.Head = [random.randint(1, place_height-2), random.randint(1, place_width-2), [0, 1, 0]]
        self.tail = []
        self.currentDirection = 1
        self.score = 0


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


   
def key(event):
    global AdventureProbability

    if   event.char == 'w' or event.keycode == 38: test.Head[2] = [0.4, 0.7, 0.4]
    elif event.char == 'a' or event.keycode == 37: test.Head[2] = [0.7, 0.4, 0.4]
    elif event.char == 'd' or event.keycode == 39: test.Head[2] = [0.4, 0.4, 0.7]
    elif event.char == 'p':                             print(learning.valueTheorem)
    elif event.char == 'r':                             learning.valueTheorem = learning.valueTheoremFunction()
    elif event.keycode >= 48 or event.keycode <= 57:    AdventureProbability = ((event.keycode-48)*10)+1

def main_():
    global count
    SituationsAndActions_ = []
    while True:
        for x in range(100):
            percentage = [] 

            #백분율 결정
            for i in range(AdventureProbability): percentage.append(True)       #모험
            for i in range(100-AdventureProbability): percentage.append(False)  #경험

            #종료까지 반복
            while learning.InGame:
                if random.choice(percentage):   learning.Adventure()
                else:                           learning.experience()
                learning.SnakeGame()

            #각 상황마다 보상에 따른 역전파 학습
            for i in learning.SituationsAndActions:
                learning.BackPropagation(i[0], i[1], i[2])

            count+=1

            learning.end()

            SituationsAndActions_ = learning.SituationsAndActions[-1]
            learning.SituationsAndActions = []
            #print(learning.valueTheorem)
            learning.InGame = True

        E = 0
        for z in range(3): E += (SituationsAndActions_[1][z] - learning.valueTheorem[-1][0][z])**2/2
        print(f'{np.round(learning.valueTheorem[-1][0], 3)}\t{np.round(SituationsAndActions_[1])}\t{learning.tail_len-1}\t{count}\t{AdventureProbability}')
    
def shown():
    tail_List = []
    for i in range(100): rectangle=canvas.create_rectangle(0, 0, 0, 0, fill='#006699', outline='#006699', tag="tail"+str(i))                                    #꼬리선언
    rectangle=canvas.create_rectangle(test.Food[1]*20, test.Food[0]*20, test.Food[1]*20+18, test.Food[0]*20+18, fill='#FFFF81', outline='#FFFF81', tag="food")  #음식선언
    rectangle=canvas.create_rectangle(test.Head[1]*20, test.Head[0]*20, test.Head[1]*20+18, test.Head[0]*20+18, fill='#0099CC', outline='#0099CC', tag="Head")  #머리선언

    while True:
        percentage = [] 

        #백분율 결정
        for i in range(AdventureProbability): percentage.append(True)       #모험
        for i in range(100-AdventureProbability): percentage.append(False)  #경험

        test.valueTheorem = learning.valueTheorem.copy()                    #카피

        while test.InGame:
            time.sleep(speed)
            if random.choice(percentage):   test.Adventure()
            else:                           test.experience()

            test.SnakeGame()

            #만약 음식을 먹었다면
            if len(tail_List) != test.score:
                #꼬리추가
                tail_List.append(str(test.score))
                canvas.coords("tail"+tail_List[-1], test.tail[-1][1]*20, test.tail[-1][0]*20, test.tail[-1][1]*20+18, test.tail[-1][0]*20+18)

            #꼬리가 있다면
            elif len(tail_List) != 0:
                #마지막 꼬리를 맨 앞으로
                canvas.coords("tail"+tail_List[0], test.tail[-1][1]*20, test.tail[-1][0]*20, test.tail[-1][1]*20+18, test.tail[-1][0]*20+18)
                tail_List = tail_List[1:] + tail_List[:1]


            canvas.coords("food", test.Food[1]*20, test.Food[0]*20, test.Food[1]*20+18, test.Food[0]*20+18)   #음식이동
            canvas.coords("Head", test.Head[1]*20, test.Head[0]*20, test.Head[1]*20+18, test.Head[0]*20+18)   #머리이동
            
        #꼬리초기화
        for i in tail_List: canvas.coords("tail"+i, 0, 0, 0, 0)

        tail_List = []

        test.end()
        test.SituationsAndActions = []

        test.InGame = True

win=tkinter.Tk()
win.title("키 입력")
win.resizable(False, False)

canvas=tkinter.Canvas(win, width = 200, height = 200, bg = '#001133')

for y in range(10):
    for x in range(10):
        rectangle=canvas.create_rectangle(x*20, y*20, x*20+18, y*20+18, fill='#000000', outline='#000000')

learning= Game()
test    = Game()

main_start = threading.Thread(target=main_)
main_start.start()

shown_start = threading.Thread(target=shown)
shown_start.start()

win.bind('<Key>', key)

canvas.pack()
win.mainloop()
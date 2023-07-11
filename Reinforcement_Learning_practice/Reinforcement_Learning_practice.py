import time
import math
import random
import tkinter
import threading
import numpy                as np
import matplotlib.pyplot    as plt
import matplotlib.animation as animation
#from numba import jit

#가로세로
window_width = 10
window_height= 10


Nx_speed = 10               #게임속도

DiscountFactor = 0.3        #보상감가율
AdventureProbability = 50   #모험률
LearningNumber = 1          #한번에 학습하는 정도


class CREATURE:
    def __init__(self):
        self.speed = 1                          #속도

        self.locationX  = random.uniform(0, 10) #위치
        self.locationY  = random.uniform(0, 10)
        self.direction  = 0                     #각도

        self.interval   = 10                    #공격자와 도망자의 거리
        self.angle      = 10                    #공격자와 도망자의 각도

        self.SituationsAndActions   = []        #한 턴에 상황, 행동, 점수리스트
        self.PolicyGraph            = []        #모든 턴에 상황, 행동, 점수리스트

        self.ActionNumber = 0                   #현제액션

    def move(self): #이동
        self.locationX = (self.locationX + math.cos(math.radians(self.direction))*self.speed) % window_width
        self.locationY = (self.locationY + math.sin(math.radians(self.direction))*self.speed) % window_height
        
    def action(self): #행동
        #랜덤 값
        random_ = random.uniform(0, 360)

        
        candidate = []  #현제 상황과 비슷한 상황
        percentage = [] #백분율 함수


        #현제 상황과 비슷한 상황 추출
        for x in self.PolicyGraph:
            if(x[0] > self.angle-10 and x[0] < self.angle+10 and x[1] > self.interval-1 and x[1] < self.interval+1): 
                candidate.append(x)

        #높은 점수 순으로 정렬
        candidate.sort(key=lambda x: -x[3])
        
        #만약 비슷한 상황이 없다면 랜덤, 있다면 모험률에 따라 백분율로 결정
        if candidate == []:
            self.direction = random_
        else: 
            for i in range(AdventureProbability): percentage.append(random_)
            for i in range(100-AdventureProbability): percentage.append(candidate[0][2])
            self.direction = random.choice(percentage)


    def attacker(self, x): #공격자였을 때
        return -x

    def victim(self, x): #도망자였을 때
        return x

    def Role(self, opposing_role, function_): #공격자와 도망자의 거리, 각도, 보상을 리스트에 추가
        #각도
        self.angle = math.degrees(math.atan2(opposing_role.locationX - self.locationX, opposing_role.locationY - self.locationY))

        #거리
        self.interval = math.sqrt(((opposing_role.locationX - self.locationX) ** 2) + ((opposing_role.locationY - self.locationY) ** 2))


        #리스트에 추가
        self.SituationsAndActions.append([self.angle, self.interval, self.direction, function_(self.interval)])

    
    def rewardDistribution(self, function_): #보상측정과 초기화
        #턴이 끝날 때 보상
        self.SituationsAndActions[len(self.SituationsAndActions)-1][3] = function_(-len(self.SituationsAndActions)/20)

        #가감률에 따라 보상이 가감되며 전도
        for i in reversed(range(len(self.SituationsAndActions)-1)):
            self.SituationsAndActions[i][3] += self.SituationsAndActions[i + 1][3] * DiscountFactor

        #현제턴 정보를 전체에 전달
        for i in self.SituationsAndActions: self.PolicyGraph.append(i)

        #초기화
        self.locationX = random.uniform(0, 10)
        self.locationY = random.uniform(0, 10)

        self.SituationsAndActions = []

        #액션 측정
        self.ActionNumber += 1

        
#클래스 시작
attacker = CREATURE()
victim = CREATURE()



def update(args):
    #실행횟수만큼 반복
    for l in range(LearningNumber):
        #도망자가 잡혔을 때
        if(attacker.interval <= 1): 
            #보상측정과 초기화
            attacker.rewardDistribution(attacker.attacker)
            victim.rewardDistribution(victim.victim)

            #액션횟수
            print(attacker.ActionNumber)

        #각각 행동
        attacker.action()
        victim.action()

        #각각 이동
        attacker.move()
        victim.move()

        #각각 리스트 추가
        attacker.Role(victim, attacker.attacker)
        victim.Role(attacker, victim.victim)

    #그래프 화면 재설정
    attackerVisual.set_data(attacker.locationX, attacker.locationY)
    victimVisual.set_data(victim.locationX, victim.locationY)
    return victimVisual, attackerVisual,



def Window():

    window=tkinter.Tk()
    
    window.title("값 조절")
    window.geometry("640x400")
    window.resizable(False, False)

    #모험률
    def DiscountFactorSelect(self):
        global DiscountFactor
        DiscountFactor = DiscountFactorScale.get()


    DiscountFactorVar =tkinter.IntVar()
    DiscountFactorScale=tkinter.Scale(window, variable=DiscountFactorVar, command=DiscountFactorSelect, label = "모험률", orient="horizontal", showvalue=False, tickinterval=10, resolution = 0.1, length=600)
    DiscountFactorScale.pack()


    #학습횟수 글자
    label=tkinter.Label(window, text="학습횟수", width=10, height=1)
    label.pack()
    

    #학습횟수 입력
    def LearningNumberSelect(event):
        global LearningNumber
        print(LearningNumber)
        LearningNumber = int(LearningNumberEntry.get())

    LearningNumberEntry=tkinter.Entry(window)
    LearningNumberEntry.bind("<Return>", LearningNumberSelect)
    LearningNumberEntry.pack()

    window.mainloop()



#그래프 설정
fig, ax = plt.subplots()

#그래프 가로세로
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)

#공격자와 도망자 모습 설정
victimVisual, = plt.plot(victim.locationX, victim.locationY, 'go', markersize = 20)
attackerVisual, = plt.plot(attacker.locationX, attacker.locationY, 'yo', markersize = 20)

#행동반복
ani = animation.FuncAnimation(fig, update, interval=500/Nx_speed, blit=True)

#윈도우 창 실행
WindowProcess = threading.Thread(target=Window)
WindowProcess.start()

#그래프 실행
plt.show()
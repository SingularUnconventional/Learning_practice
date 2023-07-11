import numpy as np
import random

Scale = [50, 50]
D = 0.9994

Q = np.zeros((Scale[0], Scale[1], 4, 2))
S = [[random.randint(0, Scale[0]-1), random.randint(0, Scale[1]-1)]]
A = []
R = []

restaurant = [[random.randint(0, Scale[0]-1), random.randint(0, Scale[1]-1)] for i in range(random.randint(1, 10))]
print(len(restaurant))

while True:
    for N in range(100):
        while not S[-1] in restaurant:
            A.append(random.randint(1, 4))

            if      A[-1] == 1 and S[-1][0] != 0:            S.append([S[-1][0] - 1, S[-1][1]])
            elif    A[-1] == 2 and S[-1][1] != Scale[0]-1:   S.append([S[-1][0], S[-1][1] + 1])
            elif    A[-1] == 3 and S[-1][0] != Scale[1]-1:   S.append([S[-1][0] + 1, S[-1][1]])
            elif    A[-1] == 4 and S[-1][1] != 0:            S.append([S[-1][0], S[-1][1] - 1])
            else: S.append(S[-1])

            R.append(-0.1)

        try:
            R[-1] = 10

            A.append(0)
            R.append(0)

            Len_s_2 = len(S)-2

            R[Len_s_2] += D*R[Len_s_2+1]
        
            Q[S[Len_s_2][0]][S[Len_s_2][1]][A[Len_s_2]-1][0] += 1.0

            e = 1/Q[S[Len_s_2][0]][S[Len_s_2][1]][A[Len_s_2]-1][0]
            Q_ = Q[S[Len_s_2][0]][S[Len_s_2][1]][A[Len_s_2]-1][1]
    
            Q[S[Len_s_2][0]][S[Len_s_2][1]][A[Len_s_2]-1][1] = ((1-e)*Q_) + (e*R[Len_s_2])

            for i in reversed(range(len(S)-2)):
                R[i] += D*R[i+1]
        
                Q[S[i][0]][S[i][1]][A[i]-1][0] += 1

                e = 1/Q[S[i][0]][S[i][1]][A[i]-1][0]
                Q_ = Q[S[i][0]][S[i][1]][A[i]-1][1]
    
                Q[S[i][0]][S[i][1]][A[i]-1][1] = ((1-e)*Q_) + (e*R[i])

                #L = []
                #for l in range(4):
                #    L.append(Q[S[i+1][0]][S[i+1][1]][l][1])

                #L_ = sorted(L)
                #for l in range(4):
                #    if L[l] == L_[-1]:
                #        Q[S[i][0]][S[i][1]][A[i]-1][1] = (1-e)*Q_ + e*(R[i] + D*L[l])

        except :
            print("Error")
        

        S = [[random.randint(0, Scale[0]-1), random.randint(0, Scale[1]-1)]]
        A = []
        R = []

    for y in range(Scale[0]):
        for x in range(Scale[1]):
            if [y, x] in restaurant: print('\033[47m' + "  " + '\033[00m', end='')
            else:
                L = []
                for i in range(4):
                    L.append(Q[y][x][i][1])

                L_ = sorted(L)
                for i in range(4):
                    if L[i] == L_[-1]:
                        if      i == 0: print('\033[94m' + "⬆", end='')
                        elif    i == 1: print('\033[92m' + "➡", end='')
                        elif    i == 2: print('\033[91m' + "⬇", end='')
                        elif    i == 3: print('\033[93m' + "⬅", end='')
                        break
        print('\033[00m')
    print()
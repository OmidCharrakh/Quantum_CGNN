
import numpy as np
import utilities_functions as uFun


############
# 4x4 DAGs
############
A04_00_01 = np.zeros((4,4),int); #A04_00_01
A04_02_01 = np.zeros((4,4),int); A04_02_01[2,0]=A04_02_01[3,1]=1 #A04_02_01
A04_03_01 = np.zeros((4,4),int); A04_03_01[0,1]=A04_03_01[2,0]=A04_03_01[3,1]=1 #A04_03_01
A04_03_02 = np.zeros((4,4),int); A04_03_02[2,0]=A04_03_02[2,1]=A04_03_02[3,1]=1 #A04_03_02
A04_03_03 = np.zeros((4,4),int); A04_03_03[0,3]=A04_03_03[2,0]=A04_03_03[3,1]=1 #A04_03_03
A04_03_04 = np.zeros((4,4),int); A04_03_04[2,3]=A04_03_04[3,0]=A04_03_04[3,1]=1 #A04_03_04
A04_03_05 = np.zeros((4,4),int); A04_03_05[2,0]=A04_03_05[2,3]=A04_03_05[3,1]=1 #A04_03_05
A04_03_06 = np.zeros((4,4),int); A04_03_06[0,1]=A04_03_06[2,0]=A04_03_06[3,0]=1 #A04_03_06
A04_04_01 = np.zeros((4,4),int); A04_04_01[0,1]=A04_04_01[2,0]=A04_04_01[2,1]=A04_04_01[3,1]=1 #A04_04_01
A04_04_02 = np.zeros((4,4),int); A04_04_02[2,0]=A04_04_02[2,1]=A04_04_02[3,0]=A04_04_02[3,1]=1 #A04_04_02
A04_04_03 = np.zeros((4,4),int); A04_04_03[0,1]=A04_04_03[0,3]=A04_04_03[2,0]=A04_04_03[3,1]=1 #A04_04_03
A04_04_04 = np.zeros((4,4),int); A04_04_04[0,3]=A04_04_04[2,0]=A04_04_04[2,1]=A04_04_04[3,1]=1 #A04_04_04
A04_04_05 = np.zeros((4,4),int); A04_04_05[2,0]=A04_04_05[2,1]=A04_04_05[2,3]=A04_04_05[3,1]=1 #A04_04_05
A04_04_06 = np.zeros((4,4),int); A04_04_06[0,1]=A04_04_06[2,0]=A04_04_06[2,3]=A04_04_06[3,1]=1 #A04_04_06
A04_05_01 = np.zeros((4,4),int); A04_05_01[0,1]=A04_05_01[2,0]=A04_05_01[2,1]=A04_05_01[3,0]=A04_05_01[3,1]=1 #A04_05_01
A04_05_02 = np.zeros((4,4),int); A04_05_02[0,1]=A04_05_02[0,3]=A04_05_02[2,0]=A04_05_02[2,1]=A04_05_02[3,1]=1 #A04_05_02
A04_06_01 = np.zeros((4,4),int); A04_06_01[0,1]=A04_06_01[0,3]=A04_06_01[2,0]=A04_06_01[2,1]=A04_06_01[2,3]=A04_06_01[3,1]=1 #A04_06_01

cands_4 = [
    A04_00_01,
    A04_02_01, 
    A04_03_01, A04_03_02, A04_03_03, A04_03_04, A04_03_05, A04_03_06,
    A04_04_01, A04_04_02, A04_04_03, A04_04_04, A04_04_05, A04_04_06,
    A04_05_01, A04_05_02, 
    A04_06_01,
]

############
# 5x5 DAGs
############
A05_04_01 = np.zeros((5,5),int); A05_04_01[2,4]=A05_04_01[3,4]=A05_04_01[4,0]=A05_04_01[4,1]=1 #A05_04_01
A05_04_02 = np.zeros((5,5),int); A05_04_02[0,4]=A05_04_02[2,0]=A05_04_02[3,1]=A05_04_02[4,1]=1 #A05_04_02
A05_04_03 = np.zeros((5,5),int); A05_04_03[2,0]=A05_04_03[3,1]=A05_04_03[3,4]=A05_04_03[4,0]=1 #A05_04_03
A05_04_04 = np.zeros((5,5),int); A05_04_04[2,0]=A05_04_04[3,4]=A05_04_04[4,0]=A05_04_04[4,1]=1 #A05_04_04
A05_04_05 = np.zeros((5,5),int); A05_04_05[2,0]=A05_04_05[3,1]=A05_04_05[4,0]=A05_04_05[4,2]=1 #A05_04_05
A05_04_06 = np.zeros((5,5),int); A05_04_06[4,0]=A05_04_06[4,1]=A05_04_06[4,2]=A05_04_06[4,3]=1 #A05_04_06
A05_04_07 = np.zeros((5,5),int); A05_04_07[2,0]=A05_04_07[3,4]=A05_04_07[4,0]=A05_04_07[4,1]=1 #A05_04_07
A05_04_08 = np.zeros((5,5),int); A05_04_08[2,0]=A05_04_08[3,1]=A05_04_08[4,0]=A05_04_08[4,1]=1 #A05_04_08
A05_04_09 = np.zeros((5,5),int); A05_04_09[2,0]=A05_04_09[3,1]=A05_04_09[4,2]=A05_04_09[4,3]=1 #A05_04_09
A05_05_01 = np.zeros((5,5),int); A05_05_01[2,0]=A05_05_01[2,4]=A05_05_01[3,1]=A05_05_01[4,0]=A05_05_01[4,1]=1 #A05_05_01
A05_05_02 = np.zeros((5,5),int); A05_05_02[2,0]=A05_05_02[3,1]=A05_05_02[4,0]=A05_05_02[4,1]=A05_05_02[4,2]=1 #A05_05_02
A05_05_03 = np.zeros((5,5),int); A05_05_03[2,0]=A05_05_03[2,1]=A05_05_03[3,1]=A05_05_03[4,0]=A05_05_03[4,1]=1 #A05_05_03
A05_05_04 = np.zeros((5,5),int); A05_05_04[0,1]=A05_05_04[2,0]=A05_05_04[3,1]=A05_05_04[4,0]=A05_05_04[4,1]=1 #A05_05_04
A05_05_05 = np.zeros((5,5),int); A05_05_05[0,4]=A05_05_05[2,0]=A05_05_05[2,4]=A05_05_05[3,1]=A05_05_05[4,1]=1 #A05_05_05
A05_05_06 = np.zeros((5,5),int); A05_05_06[2,0]=A05_05_06[3,1]=A05_05_06[3,4]=A05_05_06[4,0]=A05_05_06[4,1]=1 #A05_05_06
A05_05_07 = np.zeros((5,5),int); A05_05_07[1,0]=A05_05_07[2,0]=A05_05_07[3,1]=A05_05_07[4,0]=A05_05_07[4,1]=1 #A05_05_07
A05_05_08 = np.zeros((5,5),int); A05_05_08[2,0]=A05_05_08[3,0]=A05_05_08[3,1]=A05_05_08[4,0]=A05_05_08[4,1]=1 #A05_05_08
A05_05_09 = np.zeros((5,5),int); A05_05_09[1,0]=A05_05_09[2,0]=A05_05_09[3,4]=A05_05_09[4,0]=A05_05_09[4,1]=1 #A05_05_09
A05_06_01 = np.zeros((5,5),int); A05_06_01[2,0]=A05_06_01[2,4]=A05_06_01[3,1]=A05_06_01[3,4]=A05_06_01[4,0]=A05_06_01[4,1]=1 #A05_06_01
A05_06_02 = np.zeros((5,5),int); A05_06_02[2,0]=A05_06_02[3,1]=A05_06_02[4,0]=A05_06_02[4,1]=A05_06_02[4,2]=A05_06_02[4,3]=1 #A05_06_02
A05_06_03 = np.zeros((5,5),int); A05_06_03[0,1]=A05_06_03[2,0]=A05_06_03[2,1]=A05_06_03[3,1]=A05_06_03[4,0]=A05_06_03[4,1]=1 #A05_06_03
A05_06_04 = np.zeros((5,5),int); A05_06_04[1,0]=A05_06_04[2,0]=A05_06_04[3,1]=A05_06_04[3,4]=A05_06_04[4,0]=A05_06_04[4,1]=1 #A05_06_04
A05_06_05 = np.zeros((5,5),int); A05_06_05[2,0]=A05_06_05[3,0]=A05_06_05[3,1]=A05_06_05[3,4]=A05_06_05[4,0]=A05_06_05[4,1]=1 #A05_06_05
A05_06_06 = np.zeros((5,5),int); A05_06_06[2,0]=A05_06_06[2,1]=A05_06_06[3,0]=A05_06_06[3,1]=A05_06_06[4,0]=A05_06_06[4,1]=1 #A05_06_06
A05_07_01 = np.zeros((5,5),int); A05_07_01[0,1]=A05_07_01[0,4]=A05_07_01[2,0]=A05_07_01[2,1]=A05_07_01[2,4]=A05_07_01[3,1]=A05_07_01[4,1]=1 #A05_07_01
A05_07_02 = np.zeros((5,5),int); A05_07_02[0,1]=A05_07_02[2,0]=A05_07_02[2,1]=A05_07_02[3,0]=A05_07_02[3,1]=A05_07_02[4,0]=A05_07_02[4,1]=1 #A05_07_02
A05_08_01 = np.zeros((5,5),int); A05_08_01[2,0]=A05_08_01[2,1]=A05_08_01[2,4]=A05_08_01[3,0]=A05_08_01[3,1]=A05_08_01[3,4]=A05_08_01[4,0]=A05_08_01[4,1]=1 #A05_08_01

cands_5 = [
    A05_04_01, A05_04_02, A05_04_03, A05_04_04, A05_04_05, A05_04_06, A05_04_07, A05_04_08, A05_04_09, 
    A05_05_01, A05_05_02, A05_05_03, A05_05_04, A05_05_05, A05_05_06, A05_05_07, A05_05_08, A05_05_09, 
    A05_06_01, A05_06_02, A05_06_03, A05_06_04, A05_06_05, A05_06_06, 
    A05_07_01, A05_07_02, 
    A05_08_01,
]

############
# 6x6 DAGs
############
A06_06_01 = np.zeros((6,6),int); A06_06_01[2,0]=A06_06_01[2,5]=A06_06_01[3,1]=A06_06_01[4,0]=A06_06_01[4,1]=A06_06_01[5,1]=1 #A06_06_01
A06_06_02 = np.zeros((6,6),int); A06_06_02[2,0]=A06_06_02[3,1]=A06_06_02[4,0]=A06_06_02[4,1]=A06_06_02[5,1]=A06_06_02[5,2]=1 #A06_06_02
A06_06_03 = np.zeros((6,6),int); A06_06_03[1,5]=A06_06_03[2,0]=A06_06_03[3,1]=A06_06_03[4,0]=A06_06_03[4,1]=A06_06_03[5,0]=1 #A06_06_03
A06_06_04 = np.zeros((6,6),int); A06_06_04[2,0]=A06_06_04[3,1]=A06_06_04[3,5]=A06_06_04[4,0]=A06_06_04[4,1]=A06_06_04[5,0]=1 #A06_06_04
A06_06_05 = np.zeros((6,6),int); A06_06_05[2,0]=A06_06_05[3,1]=A06_06_05[3,4]=A06_06_05[4,0]=A06_06_05[5,0]=A06_06_05[5,1]=1 #A06_06_05
A06_06_06 = np.zeros((6,6),int); A06_06_06[2,4]=A06_06_06[3,4]=A06_06_06[4,0]=A06_06_06[4,1]=A06_06_06[5,2]=A06_06_06[5,3]=1 #A06_06_06
A06_06_07 = np.zeros((6,6),int); A06_06_07[2,0]=A06_06_07[3,1]=A06_06_07[4,0]=A06_06_07[4,1]=A06_06_07[5,2]=A06_06_07[5,3]=1 #A06_06_07
A06_07_01 = np.zeros((6,6),int); A06_07_01[2,0]=A06_07_01[3,1]=A06_07_01[4,0]=A06_07_01[4,1]=A06_07_01[5,2]=A06_07_01[5,3]=A06_07_01[5,4]=1 #A06_07_01
A06_07_02 = np.zeros((6,6),int); A06_07_02[2,0]=A06_07_02[2,5]=A06_07_02[3,1]=A06_07_02[4,0]=A06_07_02[4,1]=A06_07_02[4,5]=A06_07_02[5,3]=1 #A06_07_02
A06_08_01 = np.zeros((6,6),int); A06_08_01[2,0]=A06_08_01[2,4]=A06_08_01[3,1]=A06_08_01[3,4]=A06_08_01[4,0]=A06_08_01[4,1]=A06_08_01[5,0]=A06_08_01[5,1]=1 #A06_08_01

cands_6 = [
    A06_06_01, A06_06_02, A06_06_03, A06_06_04, A06_06_05, A06_06_06, A06_06_07, 
    A06_07_01, A06_07_02, 
    A06_08_01,
]
############
# 7x7 DAGs
############
A07_08_01 = np.zeros((7,7),int); A07_08_01[2,0]=A07_08_01[2,5]=A07_08_01[3,1]=A07_08_01[3,6]=A07_08_01[4,0]=A07_08_01[4,1]=A07_08_01[5,1]=A07_08_01[6,0]=1 #A07_08_01
A07_08_02 = np.zeros((7,7),int); A07_08_02[2,0]=A07_08_02[3,1]=A07_08_02[4,0]=A07_08_02[4,1]=A07_08_02[5,1]=A07_08_02[5,2]=A07_08_02[6,0]=A07_08_02[6,3]=1 #A07_08_02

cands_7 = [
    A07_08_01, A07_08_02,
]
############

# reshpae all cands into 8x8 adjacencies and save a txt file
target_nodes = 8
cands_list = cands_4 + cands_5 + cands_6 + cands_7
n_cands = len(cands_list)
container = np.zeros((n_cands, target_nodes*target_nodes), int)

for index, A in enumerate(cands_list):
    A_pad = uFun.pad_adjacency(A, target_nodes)
    container[index] = A_pad.reshape(-1)
np.savetxt('ccm.txt', container, fmt='%s')


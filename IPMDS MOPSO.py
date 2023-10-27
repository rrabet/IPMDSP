
#import random
import pandas as pd
from random import randint
from random import uniform
from random import choices
from random import triangular
from copy import deepcopy
from math import exp
import time
import csv
#import matplotlib.pyplot as plt

#...............................................................................................................
#                                          initial solution and fitness
#...............................................................................................................
global chromosome
chromosome = dict({})

def initial_generation(G, chromosome, Pn):
    Gene = []
    for g in range(G):
        Gene.append([0,[0,0,0], [0,[0,0,0]], 0, 0, [0, 0]])      # [Sol number,[cost, delay time, Likelihood], [Best memory,[0,0]], IsDominated, Grid_index, Grid_sub_index]
        gene = [[],[],[],[], [], []]    # [ [customer priority], [vehicle priority], [customer velocity], [vehicle velocity] ]
        for i in range(customer.n[-1]):
            gene[0].append(uniform(0, 1))
            gene[2].append(uniform(0, 1))
        for i in range(0, vehicle.n[-1]):
            gene[1].append(uniform(0, 1))
            gene[3].append(uniform(0, 1))
        for i3 in range(0, Pn):
            gene[4].append(uniform(0, 1))
            gene[5].append(uniform(0, 1))
        #print("gene[0]====================", gene[0])
        #print("gene[1]====================", gene[1])
        #print("gene[2]====================", gene[2])
        #print("gene[3]====================", gene[3])
        #print("gene[4]====================", gene[4])
        #print("gene[5]====================", gene[5])
        #print("gene[1]====================", gene[1])
        Gene[g][0]= len(chromosome)
        Gene[g][2][0] = len(chromosome)
        chromosome[len(chromosome)] = gene
    #print("len chromosome", len(chromosome))
    #print("chromosome", chromosome)
    #print("Gene", Gene)
    #print("\n")
    return Gene

def Fitness_function(Gene, Pn):
    Results = []                    # [ [first solution index, [cost, delay time, Likelihood]], [second...],...]

    for g in range(len(Gene)):
        #print("Gene[g]", Gene[g])
        Results.append([0,[0,0,0], [0,[0,0,0]], 0, 0, [0, 0, 0]])
        index = Gene[g][0]
        if Gene[g][2][1] == [0, 0, 0]:           # Entering gene is initial solution
            flag = True
        else:
            flag = False
        gene = deepcopy(chromosome[index])
        #print("gene[g]", gene[g])
        customer_list = []
        for m in range(len(gene[0])):
            m = max(gene[0])
            #print("max gene =", m)
            C = gene[0].index(m)
            #print("gene[0].index m", C)
            customer_list.append(C)
            gene[0][C] = -1000000
        #print("customer_list = ", customer_list)

        vehicle_list = []
        for m in range(len(gene[1])):
            m = max(gene[1])
            # print("max gene =", m)
            C = gene[1].index(m)
            # print("gene[0].index m", C)
            vehicle_list.append(C)
            gene[1][C] = -1000000
        #print("vehicle_list = ", vehicle_list)

        main_list = []
        for m in range(len(gene[5])):
            C = gene[5][m]*(main.max_rec-main.min_rec)
            # print("gene[0].index m", C)
            main_list.append(C+main.min_rec)
            gene[5][m] = -1000000
        #print("main_list = ", main_list)

        answer = [[], [], []]
        # print( f'Gene[{index}] ={Gene[index]}')
        for r in range(Rep):
            temp_gene = deepcopy([customer_list, vehicle_list, main_list])
            # print("temp_gene = Gene[index]", Gene[index])
            loading_list = []  # [ [step0[veh0[specified customers]], [veh1[specified customers]], ...],[step1[...],...]
            deliver_timer = []  # [ [deliver time of every customer in first vehicle route],...]
            load_timer = [0 for i in range(len(temp_gene[1]))]  # real time of every vehicle

            for l in range(0, loop):  # .......Loading_list (Dedication phase)
                loading_list.append([])
                deliver_timer.append([])
                for j in range(0, len(temp_gene[1])):
                    temp_veh = []
                    # print("temp_gene[1][j]", temp_gene[1][j])
                    # print("vehicle.cap", vehicle.cap[temp_gene[1][j]])
                    current_cap = deepcopy(vehicle.cap[temp_gene[1][j]])
                    # print("cap =", vehicle.cap[veh])
                    # print("len(temp_gene[0])",len(temp_gene[0]))
                    for c in range(0, len(temp_gene[0])):
                        if temp_gene[0][c] != 0:
                            # print ("work =", customer.wn[c])
                            # print("temp_gene[0][c]",temp_gene[0][c])
                            volume = sum(customer.wn[temp_gene[0][c]])
                            if current_cap >= volume:
                                # print(".........................")
                                # print("customer.wn =", customer.wn[temp_gene[0][c]])
                                # print("current_cap =", current_cap)
                                current_cap -= volume
                                # process_time = customer.process_time[temp_gene[0][c]]
                                temp_veh.append(temp_gene[0][c])
                                temp_gene[0][c] = 0
                                # print("temp_gene[0][c] =", temp_gene[0][c])
                            else:
                                temp_veh.append(0)
                                loading_list[l].append(temp_veh)
                                break

            maint_cost = 0  # maintenance cost
            dist_cost = 0  # distribution cost
            Tardiness = 0
            timer_prod = [0 for i2 in range(0, Pn)]
            timer_main = [0 for i2 in range(0, Pn)]
            # print("timer_prod", timer_prod)
            for step in range(0, len(loading_list)):  # .........Production phase
                for veh in range(0, len(loading_list[step])):
                    timer_temp = [0 for i3 in range(0, Pn)]  # temporary production timer
                    deliver_timer[step].append([0 for o in range(0, len(loading_list[step][veh]))])
                    # print("loading_list[step][veh] =", loading_list[step][veh])
                    for cust in loading_list[step][veh]:
                        if cust != 0:
                            for o1 in range(0, Pn):
                                # products_wn_ordered[o1] += customer.wn[customer_list[cust]][o1]
                                timer_temp[o1] += customer.process_time[cust][o1]
                                timer_main[o1] += customer.process_time[cust][o1]

                    for t2 in range(Pn):  # ...........Maintenance phase
                        fail_prob = [0 for i3 in range(0, Pn)]  # system failure probability
                        if timer_main[t2] > main_list[t2]:  # maintenance frequency
                            # print("maint_cost                       0")
                            maint_cost += main.cost[t2]
                            timer_temp[t2] += main.duration[t2]
                            timer_main[t2] = 0
                        # print("timer_main[t2]", timer_main[t2])
                        fail_prob[t2] = 1 - exp(-(timer_main[t2] * main.fail_landa[t2]))
                        # print("fail_prob[t2]", fail_prob[t2])
                        uni = uniform(0, 1)
                        # print("uni", uni)
                        if fail_prob[t2] > uni:
                            # print("maint_cost                       1")
                            maint_cost += main.fail_cost[t2]
                            timer_temp += main.fail_dur
                        timer_prod[t2] += timer_temp[t2]

                    load_timer[vehicle_list[veh]] = max(timer_prod)

            for step in range(0, len(loading_list)):  # ........Distribution phase
                # print("....................")
                # print(step)
                for veh in range(0, len(loading_list[step])):
                    # print("loading_list[step][veh] =", loading_list[step][veh])
                    for cust in range(0, len(loading_list[step][veh])):
                        # print("customer number =",cust)
                        dest = loading_list[step][veh][cust]  # designated customer for next move
                        # print("loading_list[step][veh][cust] =", loading_list[step][veh][cust])
                        #print("dest =", dest)
                        if cust == 0:  # if current location of vehicle is factory then calculate first customer related costs
                            if dest != 0:
                                dist_cost += vehicle.Fixed_cost[vehicle_list[veh]] \
                                             + (customer.distance[0][dest] / 100) * \
                                             vehicle.alpha[vehicle_list[veh]] * cost.fuel * vehicle.fuel_cons[
                                                 vehicle_list[veh]]
                                load_timer[vehicle_list[veh]] += customer.distance[0][dest] / vehicle.velocity[
                                    vehicle_list[veh]]
                                deliver_timer[step][veh][cust] += load_timer[vehicle_list[veh]]
                                p = customer.presumed_delivery_time[customer_list[dest]]
                                if time_window < deliver_timer[step][veh][cust] - p:
                                    Tardiness += abs(p - deliver_timer[step][veh][cust]) - time_window
                        else:
                            dist_cost += (customer.distance[loading_list[step][veh][cust - 1]][dest] / 100) * \
                                         vehicle.alpha[vehicle_list[veh]] * cost.fuel * vehicle.fuel_cons[
                                             vehicle_list[veh]]
                            load_timer[vehicle_list[veh]] += customer.distance[loading_list[step][veh][cust - 1]][dest] / vehicle.velocity[vehicle_list[veh]]
                            deliver_timer[step][veh][cust] += load_timer[vehicle_list[veh]]
                            p = customer.presumed_delivery_time[customer_list[dest]]
                            if time_window < deliver_timer[step][veh][cust] - p:
                                Tardiness += abs(p - deliver_timer[step][veh][cust]) - time_window
            # fitness[i].append(i)
            answer[0].append(dist_cost)
            answer[1].append(Tardiness)
            answer[2].append(maint_cost)

        #print("index", index)
        Results[g][0] = index                   # [ [first solution index, [cost, delay time, Likelihood]], [second...],...]
        Results[g][1] = [sum(answer[0])/len(answer[0]), sum(answer[1])/len(answer[1]), sum(answer[2])/len(answer[2])]
        #print(" Results[g]",  Results[g])
        if flag == True:                        # gene [g] is from initial solutions
            Results[g][2][0] = index
            Results[g][2][1] = [sum(answer[0])/len(answer[0]), sum(answer[1])/len(answer[1]), sum(answer[2])/len(answer[2])]
        else:                                   # adding best personal memory of the particle
            if Gene[g][2][1][0] < sum(answer[0])/len(answer[0]) and Gene[g][2][1][1] < \
                    sum(answer[1])/len(answer[1]) and  Gene[g][2][1][2] < sum(answer[2])/len(answer[2]):
                Results[g][2] = Gene[g][2]
            elif Gene[g][2][1][0] > sum(answer[0])/len(answer[0]) and Gene[g][2][1][1] > \
                    sum(answer[1])/len(answer[1]) and Gene[g][2][1][2] > sum(answer[2])/len(answer[2]):
                Results[g][2][0] = index
                Results[g][2][1] = [sum(answer[0])/len(answer[0]), sum(answer[1])/len(answer[1]), sum(answer[2])/len(answer[2])]
            else:
                x = uniform(0, 1)
                if x > 0.5:
                    Results[g][2] = Gene[g][2]
                else:
                    Results[g][2][0] = index
                    Results[g][2][1] = [sum(answer[0])/len(answer[0]), sum(answer[1])/len(answer[1]), sum(answer[2])/len(answer[2])]

            #print(f'Result[{g}] = {Results[g]}')
        # print("...............................................................................................")
    #print("\n")
    #print("Results =", Results)
    return Results

def Domination(Res, Rep =[]):
    rep = []
    temp = []
    for sol in Rep:
        if sol not in temp:
            temp.append(sol)
            #print("Rep sol", sol)
    for sol in Res:
        if sol not in temp:
            temp.append(sol)
            #print("Res sol", sol)
    for i in range(len(temp)):
        for j in range(i+1, len(temp)):
            if temp[i][1][0] > temp[j][1][0] and temp[i][1][1] >= temp[j][1][1] and temp[i][1][2] >= temp[j][1][2]:
                temp[i][3] += 1
            elif temp[i][1][0] < temp[j][1][0] and temp[i][1][1] <= temp[j][1][1] and temp[i][1][2] <= temp[j][1][2]:
                temp[j][3] += 1

    for i in range(0, len(temp)):
        if temp[i][3] == 0 and temp[i][1][0]!= 0:
            rep.append(temp[i])
        #print(Res[i])
    #if len(rep) == 0:

    return rep

def createGrid(Rep, nGrid, alpha):
    cmin = 1000000000
    cmax = 0

    tmin = 10000000
    tmax = 0

    Likemin = 10000000
    Likemax = 0
    for i in range(len(Rep)):
        if Rep[i][1][0] > cmax:
            cmax = Rep[i][1][0]
        elif Rep[i][1][0] < cmin:
            cmin = Rep[i][1][0]
        if Rep[i][1][1] > tmax:
            tmax = Rep[i][1][1]
        elif Rep[i][1][1] < tmin:
            tmin = Rep[i][1][1]
        if Rep[i][1][2] > Likemax:
            Likemax = Rep[i][1][2]
        elif Rep[i][1][2] < Likemin:
            Likemin = Rep[i][1][2]
    #print(cmin, cmax, tmin, tmax)
    dc =  (1 + 2 *alpha) * (cmax - cmin)
    #cmax = int(cmax + (cmax-cmin)*0.1)
    cmin = int(cmin - (cmax-cmin)*0.1)
    C_bounds = [int(cmin+(dc*j/nGrid)) for j in range(nGrid + 1)]
    #print(C_bounds)

    dt = (1 + 2 * alpha) * (tmax - tmin)
    #tmax = int(tmax + (tmax - tmin) * 0.1)
    tmin = int(tmin - (tmax - tmin) * 0.1)
    #print("tmin =", tmin)
    T_bounds = [int(tmin + (dt * j / nGrid)) for j in range(nGrid + 1)]

    dL =  (1 + 2 * alpha) * (Likemax - Likemin)
    #tmax = int(tmax + (tmax - tmin) * 0.1)
    Likemin = int(Likemin - (Likemax - Likemin) * 0.1)
    #print("tmin =", tmin)
    L_bounds = [int(Likemin + (dL * j / nGrid)) for j in range(nGrid + 1)]
    #print(L_bounds)
    #cmax = cmax + dc * alpha
    #cmin = cmin - dc * alpha

    for j in range(len(Rep)):
        #print("Rep", Rep[j])
        if Rep[j][3] == 0:
            for k in range(nGrid ):
                if C_bounds[k] <= Rep[j][1][0] < C_bounds[k+1]:
                    Rep[j][5][0] = k
                if T_bounds[k] <= Rep[j][1][1] < T_bounds[k+1]:
                    Rep[j][5][1] = k
                if L_bounds[k] <= Rep[j][1][2] < L_bounds[k+1]:
                    Rep[j][5][2] = k

            Rep[j][4] = (Rep[j][5][0] + Rep[j][5][1] + Rep[j][5][2])* nGrid
        #print(Rep[j])
    return Rep

def SelectLeader(Rep, Beta):
    GI = []         # Grid Index
    GS = []         # Grid counter
    GP = []         # Grid selection probability
    for i in range(len(Rep)):
        if Rep[i][4] not in GI:
            GI.append(Rep[i][4])
            GS.append(0)
        else:
            GS[GI.index(Rep[i][4])] += 1
    for g in range(len(GS)):
        GP.append(exp(-Beta * GS[g]))
    #print("GI =", GI)
    #print("GP =", GP)
    SC = choices(GI, GP)

    if GS[GI.index(SC[0])] > 0:
        u = randint(0, GS[GI.index(SC[0])])
    else:
        u = 0

    for leader in Rep:
        if leader[4] == SC[0]:
            if u == 0:
                Sleader = leader
                break
            else:
                u -= 1

    #print("GI = ", GI)
    #print("GS = ", GS)
    #print("GP = ", GP)
    #print("SC[0] = ", SC[0])
    #print("Sleader = ", Sleader)
    return Sleader

def newposition(Res, Sleader, chromosome, w, c1, c2):
    #print("Sleader =", Sleader)
    #print("len Res.......................", len(Res))
    #print("\n")
    for res in Res:
        #print("res 0", res)
        #print("chromosome [res[0]] =", chromosome [res[0]] )
        #print("len chromosome [res[0]][0] =", len(chromosome[res[0]][0]) )
        #print("len chromosome [res[0]][1] =", len(chromosome[res[0]][1]) )
        #print("len chromosome [res[0]][2] =", len(chromosome[res[0]][2]) )
        #print("len chromosome [res[0]][3] =", len(chromosome[res[0]][3]) )

        #print("Best memory =", chromosome [res[2][0]] )

        gene = [[],[],[],[],[],[]]
        for i in range(len(chromosome[res[0]][0])):
            pc = chromosome[res[0]][0][i]       # position of current situation
            vc = chromosome[res[0]][2][i]       # velocity of current position
            pb = chromosome[res[2][0]][0][i]    # position of best memory situation
            pl = chromosome[Sleader[0]][0][i]   # position of leader situation
            nv = w*vc + c1*uniform(0,1)*(pb - pc) + c2*uniform(0,1)*(pl - pc)   # new velocity
            np =  pc + nv                        # new position
            gene[0].append(np)
            gene[2].append(nv)

        for j in range(len(chromosome[res[0]][1])):
            pc = chromosome[res[0]][1][j]       # position of current situation
            vc = chromosome[res[0]][3][j]       # velocity of current position
            pb = chromosome[res[2][0]][1][j]    # position of best memory situation
            pl = chromosome[Sleader[0]][1][j]   # position of leader situation
            nv = w*vc + c1*uniform(0,1)*(pb - pc) + c2*uniform(0,1)*(pl - pc)   # new velocity
            np =  pc + nv                        # new position
            gene[1].append(np)
            gene[3].append(nv)

        for j in range(0, len(chromosome[res[0]][4])):
            #print("\n")
            #print("chromosome[res[0]][4]", chromosome[res[0]][4])
            #print("chromosome[res[0]][5]", chromosome[res[0]][5])
            pc = chromosome[res[0]][4][j]       # position of current situation
            #print("pc", pc)
            vc = chromosome[res[0]][5][j]       # velocity of current position
            #print("vc", vc)
            pb = chromosome[res[2][0]][4][j]    # position of best memory situation
            #print("pb", pb)
            pl = chromosome[Sleader[0]][2][j]   # position of leader situation
            #print("pl", pl)
            nv = w*vc + c1*uniform(0,1)*(pb - pc) + c2*uniform(0,1)*(pl - pc)   # new velocity
            #print("nv", nv)
            np =  pc + nv                        # new position
            #print("np", np)
            if np < 0:
                np = 0
            elif np > 1:
                np = 1

            gene[4].append(np)
            gene[5].append(nv)



        res[0] = len(chromosome)
        chromosome[len(chromosome)] = gene

        #print("res 1=", res)
    #print("\n")

    return Res

def Brain(MaxIt, Pn, G, chromosome, Beta, w, c1, c2, a3, b3, c3):
    Results = []

    Res = Fitness_function(initial_generation(G, chromosome, Pn), Pn)
    Rep = createGrid(Domination(Res), nGrid, alpha)
    for it in range(MaxIt):
        Sleader = SelectLeader(Rep, Beta)
        #print("Selected leader [dist_Cost, Tardiness, main_cost]=", Sleader[1])
        n_pos = newposition(Res, Sleader, chromosome, w, c1, c2)
        Res = Fitness_function(n_pos, Pn)
        Rep = Domination(Res, Rep)
        Rep = createGrid(Rep, nGrid, alpha)
        #if it in (1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 99):
        if it == 99:
            #print("\n")
            #print(" it    ", it)
            #print("len Repository ", len(Rep))
            #print("[delivery cost, delay time, maintenance costs ]")
            for i in range(len(Rep)):
                print("Front ---> Rep[i][1] ",  Rep[i][1])
                Results.append(Rep[i][1])
                #Results2[0].append(Rep[i][1][0])
                #Results2[1].append(Rep[i][1][1])
                #Results2[2].append(Rep[i][1][2])



    #print(Results2)
    #Results = pd.DataFrame(Results, columns=['delivery cost', 'delay time', 'maintenance costs'])
    #excel_file = pd.ExcelWriter("MOPSO results.xlsx")
    #Results.to_excel(excel_file, sheet_name='Pareto front', index=True)

    #excel_file.save()
    print("\n")
    print("Repository elements", len(Rep))
    #print("Repository elements", Rep)
    #print("solution num. [cost, delay time], [best memory,[cost, delay time], numbers dominated, grid index, grid sub index]")
    if True:
        dsp = []
        f0 = []
        f1 = []
        f2 = []
        for i in range(len(Rep)):
            f0.append(Rep[i][1][0])
            f1.append(Rep[i][1][1])
            f2.append(Rep[i][1][2])
            temp_dsp = []
            for j in range(len(Rep)):
                if i != j:
                    temp_dsp.append(abs(Rep[i][1][1]-Rep[j][1][1]) + abs(Rep[i][1][0]- Rep[j][1][0]) + abs(Rep[i][1][2]- Rep[j][1][2]))
            dsp.append(min(temp_dsp))
            #print("solution =",Rep[i][0], "        dist_cost, Tardiness, main_cost =",Rep[i][1])
            #print("maintenance periods =            ", chromosome[Rep[i][0]][4] )
        #print("solution =",Rep[0][0], "        dist_cost, Tardiness, main_cost =",Rep[0][1])

        #print("dsp = ", dsp)


        F_Star = [10000000000, 10000000000, 10000000000]  # Results of best solution
        for star in Rep:
            #print("star", star)
            if star[1][0] < F_Star[0]:
                F_Star[0] = star[1][0]
            if star[1][1] < F_Star[1]:
                F_Star[1] = star[1][1]
            if star[1][2] < F_Star[2]:
                F_Star[2] = star[1][2]
        f1min = F_Star[0]
        f2min = F_Star[1]
        f3min = F_Star[2]

        RNI = len(Rep) / len(chromosome)
        print("RNI =", RNI)

        MID = (sum(f0) - (len(f0)) * min(f0) + sum(f1) - (len(f1)) * min(f1) - (len(f2)) * min(f2)) ** (1.0 / 2.0) / len(Rep)
        print("MID =", MID)

        SP = 0
        for d in dsp:
            SP += ((1 / len(dsp)) * (d - (sum(dsp) / len(dsp))) ** 2) ** (1.0 / 2.0)
        print("SP =", SP)

        Duration = (time.time()) - Start_time
        print("Duration = %s seconds" % Duration)

        dGD = 0
        for i in range(len(Rep)):
            dGD += (Rep[i][1][0] - f1min) ** 2
            dGD += (Rep[i][1][1] - f2min) ** 2
            dGD += (Rep[i][1][2] - f3min) ** 2
        GD = (1 / len(Rep)) * ((dGD) ** (1 / 2))
        print("GD =", GD)

        dRV = []
        for i in range(len(Rep)):
            dRV.append(min(abs(Rep[i][1][0] - f1min), abs(Rep[i][1][1] - f2min), abs(Rep[i][1][2] - f3min)))
        dRV_bar = sum(dRV) / len(dRV)
        RV_sigma = sum([(dRV_bar - drv) ** 2 for drv in dRV])
        RV = ((1 / (len(dRV) - 1)) * RV_sigma) ** (1 / 2)
        print("RV =", RV)

        MD = ((1 / 2) * (sum(f0) / (max(f0) - min(f0)) + (sum(f1) / (max(f1) - min(f1))) + (sum(f2) / (max(f2) - min(f2))) )) ** (1.0 / 2.0)
        print("MD =", MD)

        print("WM =", (RNI + 2 * MID + 2 * SP + 2 * MD) / 7)


#......................................................................................................................
#                                              Data input and order
#......................................................................................................................

#for customer_number in (10, 13, 15, 17, 21, 25, 27, 29, 30, 35, 40, 43, 47, 52, 58, 63, 66, 69, 70):
for scenario in range(9):
    print(scenario)


    PIs = [
           [20, [1, 1, 1]],
           [25, [2, 1, 2]],
           [30, [2, 1, 2]],
           [35, [2, 2, 1]],
           [40, [2, 2, 1]],
           [40, [2, 2, 3]],
           [45, [2, 3, 2]],
           [50, [3, 2, 2]],
           [55, [2, 3, 2]],
           [55, [4, 4, 2]],
           [60, [4, 4, 2]],
           [65, [4, 4, 2]],
           [70, [4, 2, 4]],
           [75, [4, 2, 4]],
           [80, [4, 2, 4]],
           [85, [4, 2, 4]],
           [90, [4, 4, 4]],
           [95, [4, 4, 4]],
           [100, [4, 4, 4]],
           [100, [5, 5, 5]],
    ]

    customer_number = PIs[scenario][0]
    print("customer_number", customer_number)
    Pn = 3                                            # number of products
    Prd_r = [72, 75, 50]                              # production rate

    MaxIt = 100
    G = 100                                           # population number
    nRep = 20
    nGrid = 15
    w = 0.1                                             #Inertia Weight
    c1 = 0.1                                            # personal learning coefficient
    c2 = 0.02                                            # global learning coefficient
    alpha = 0.2
    beta = 0.05                                          # Leader selection pressure

    a3 = 0
    b3 = 0
    c3 = 0
    gamma = 1                                            # Deletion selection pressure
    Rep = 7                                          # number of simulation repeats

    Start_time = time.time()

    class customer:
        n = [0]
        wn = [0]                                      # required product for every customer
        presumed_delivery_time = [0]
        process_time = [0]
        distance = []                                                                               # distance between customers

    total_wn = 0
    total_p = 0
    total_t = 0
    for c in range (customer_number):
        customer.n.append(c + 1)
        customer.wn.append([int(triangular(1, 8, 3)), int(triangular(0, 5, 3)), int(triangular(0, 3, 2))])
        customer.process_time.append([(customer.wn[c+1][q]*24)/Prd_r[q] for q in range(Pn)])
        total_wn += sum(customer.wn[c + 1])                      # Total ordered products
        total_p += max(customer.process_time[c + 1])             # Total process times


    time_window = 24
    a = 24
    b = 120
    for c in range (customer_number):
        customer.presumed_delivery_time.append(uniform(a, b))

    with open('C:/Users/rabet/OneDrive/Desktop/Projects/My papers/IPMDS/old/JKMC.csv') as file:
        reader = csv.reader(file)
        cities = []
        counti = 0
        for row in reader:
            counti += 1
            countj = 0
            if counti > 1:
                customer.distance.append([])
                for item in row:
                    countj += 1
                    if countj == 1:
                        cities.append(item)
                    if countj > 1:
                        customer.distance[counti-2].append(int(item))

    class info:
        #type_Number = [10, 7, 5]
        type_Number = PIs[scenario][1]
        cap = [18, 8.5, 7]
        fuel_cons = [17, 15, 12]
        velocity = [70, 80, 90]
        alpha = [(1.45, 1.35), (1.35, 1.3),(1.3, 1.2)]  # Alpha vehicle specific constant
        Fixed_cost = [5, 4, 3]
        #cost = [17, 15, 12]

    total_veh = sum(info.type_Number)
    total_cap = sum([info.type_Number[i]*info.cap[i] for i in range(len(info.type_Number))])

    class vehicle:
        n = [i for i in range(total_veh)]
        cap = []
        cost = []
        fuel_cons = []
        velocity = []
        alpha = []
        Fixed_cost = []

    for i in range(len(info.type_Number)):
        for j in range(info.type_Number[i]):
            vehicle.cap.append(info.cap[i])
            #vehicle.cost.append(info.cost[i])
            vehicle.fuel_cons.append(info.fuel_cons[i])
            vehicle.velocity.append(info.velocity[i])
            vehicle.alpha.append(uniform(info.alpha[i][0], info.alpha[i][1] ))   # alpha
            vehicle.Fixed_cost.append(info.Fixed_cost[i])

    class main:
        duration = [int(triangular(1, 3, 5)), int(triangular(1, 3, 5)), int(triangular(1, 3, 5))]
        cost = [10, 8, 9]
        min_rec = 5            # minimum period for maintenance recommended from providers of production line
        max_rec = 35            # maximum period for maintenance recommended from providers of production line
        fail_landa = [0.0007, 0.0005, 0.0003]
        fail_dur = [int(triangular(15, 20, 24)), int(triangular(15, 20, 24)), int(triangular(15, 20, 24))]
        fail_cost = [int(triangular(52, 80, 160)), int(triangular(58, 100, 170)), int(triangular(65, 100, 180))]

    class cost:
        fuel = 0.003

    loop = int(total_wn / total_cap) + 3

    Brain(MaxIt, Pn, G, chromosome, beta, w, c1, c2, a3, b3, c3)

    #...............................model verification.......................................

    #Res = Fitness_function(initial_generation(G, chromosome, Pn), Pn)
    #print("Res", Res)








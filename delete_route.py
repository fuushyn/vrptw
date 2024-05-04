import random 
from dataclasses import dataclass
import copy
from lex import lex_next, lex_next_prune, is_last_subseq
from load_instance import Node, Edge, load_data
import logging

##load from loader
nodes, edges,Q = load_data("input.txt")

N = len(nodes)-1 #no of customers
maxTime = 100
k_max = 7
N_near= 100

sig = []

for i in range(1, len(nodes)):
    sig.append([nodes[0], nodes[i], nodes[0]])

print(len(sig))
# print(sig[0])
print(len(sig[0]))
print(len(sig[1]))
print("okokok")

def get_all_insertions(v, sig):
    print("getting all insertions")
    print("len sig", len(sig))
    print("len sig[0]", len(sig[0]))
    print("len sig[-1]", len(sig[-1]))
    all_insertions = []
    for i in range(len(sig)):
        for j in range(1, len(sig[i])):
            sig_new = copy.deepcopy(sig)
            sig_new[i].insert(j, v)
            all_insertions.append(sig_new)
    return all_insertions

def check_route_feasibility(route):
    #capacity constraint
    q = 0
    for v in route[1:-1]:
        q+= v.q_v
    
    if(q> Q):
        return False
    
    #time constraint
    a = [0 for i in range(N+2)]
    a[0] = nodes[0].e_v

    for i in range(1,N+1):
        a[i]= max(a[i-1]+ nodes[i-1].s_v+ edges[i-1][i].c_vw, nodes[i].e_v)

    a[N+1]= max(a[N]+ nodes[N].s_v+ edges[N][0].c_vw, nodes[0].e_v)

    for i in range(1, N+1):
        if(a[i]> nodes[i].l_v):
            return False
    if(a[N+1]> nodes[0].l_v):
        return False
    return True

def get_all_feasible_insertions(v, sig):
    all_f_insertions = []
    for i in range(len(sig)):
        for j in range(1, len(sig[i])):
            sig_new = copy.deepcopy(sig)
            sig_new[i].insert(j, v)
            if(check_route_feasibility(sig_new[i])):
                all_f_insertions.append(sig_new)
    return all_f_insertions

def get_min_penalty_sig(N_insert, alpha=1):
    min_p = get_penalty(N_insert[0])
    min_sig = N_insert[0]
    for s in N_insert:
        p = get_penalty(s, alpha)
        if(p<min_p):
            min_sig = s
            min_p = p
    return min_sig

def get_penalty(sig, alpha=1):
    #capacity violation
    q = 0
    for route in sig:
        for node in route[1:-1]:
            q+= node.q_v
    P_c = 0
    if(q>Q):
        P_c = q-Q

    #time violation
    P_tw = 0
    a = [nodes[0].e_v for i in range(N+2)]
    # print("len edges", len(edges))
    # print("len edges[0]", len(edges[0]))
    # print("N", N)
    for i in range(1,N+1):
        a[i]= max(a[i-1]+ nodes[i-1].s_v+ edges[i-1][i].d_e, nodes[i].e_v)
        if(a[i]>nodes[i].l_v):
            P_tw+= a[i]- nodes[i].l_v
            a[i]= nodes[i].l_v
    a[N+1] = max(a[N]+ nodes[N].s_v+edges[N][0].d_e, nodes[0].e_v)
    if(a[N+1]>nodes[0].l_v):
        P_tw+= a[N+1]- nodes[0].l_v
        a[N+1] = nodes[0].l_v

    F_p = P_c + alpha*P_tw
    return F_p

def select_if_route(sig):
    random.shuffle(sig)
    for index, value in enumerate(sig):
        if(check_route_feasibility(value)):
            return value
    return []


def get_composite_neighbourhood(v_in, sig): #v_in should already be present in sig
    # assert check_present(v_in, sig)
    N = []
    N += two_opt_star(v_in,sig)
    N += out_relocate(v_in, sig)
    N += in_relocate(v_in, sig)
    N += exchange(v_in, sig)
    return N


def two_opt_star(v_in, sig):
    ### v and w must belong to different routes in 2 opt star
    N = []
    v_sorted = get_sorted_customers(v_in, N_near)
    for i in range(len(v_sorted)):
        sig_prime1 = copy.deepcopy(sig)

        sig_prime2 = copy.deepcopy(sig)

        w = v_sorted[i]
        found = False
        for j in range(len(sig)):
            for i in range(len(sig)):
                if(v_in in sig[j] and w in sig[i] and i != j):
                    ind_v = sig[j].index(v_in)
                    ind_w = sig[i].index(w)
                    sig_prime1[j]= sig[j][:ind_v]+ sig[i][ind_w+1:]
                    sig_prime1[i]= sig[i][:ind_w+1]+sig[j][ind_v:]
                    found  = True
                    break
            if(found):
                N.append(sig_prime1)
                break

        

        found = False
        for j in range(len(sig)):
            for i in range(len(sig)):
                if(v_in in sig[j] and w in sig[i] and i!=j):
                    ind_v = sig[j].index(v_in)
                    ind_w = sig[i].index(w)
                    sig_prime2[j]= sig[j][:ind_v +1]+ sig[i][ind_w:]
                    sig_prime2[i]= sig[i][:ind_w]+ sig[j][ind_v+1:]
                    found  = True
                    break
            if(found):
                N.append(sig_prime2)
                break


    return N

def out_relocate(v_in, sig):
    N = []
    v_sorted = get_sorted_customers(v_in, N_near)
    for i in range(len(v_sorted)):
        sig_prime1 = copy.deepcopy(sig)

        sig_prime2 = copy.deepcopy(sig)

        w = v_sorted[i]
        for j in range(len(sig)):
            if(v_in in sig[j]):
                sig_prime1[j].remove(v_in)
                sig_prime2[j].remove(v_in)
                break
        for j in range(len(sig)):
            if(w in sig[j]):
                index1 = sig_prime1[j].index(w)
                index2 = sig_prime2[j].index(w)
                sig_prime1[j].insert(index1, v_in)
                sig_prime2[j].insert(index2+1, v_in)
                break
        N.append(sig_prime1)
        N.append(sig_prime2)
    return N

def in_relocate(v_in, sig):
    N = []
    v_sorted = get_sorted_customers(v_in, N_near)
    for i in range(len(v_sorted)):
        sig_prime1 = copy.deepcopy(sig)

        sig_prime2 = copy.deepcopy(sig)

        w = v_sorted[i]
        for j in range(len(sig)):
            if(w in sig[j]):
                sig_prime1[j].remove(w)
                sig_prime2[j].remove(w)
                break
        for j in range(len(sig)):
            if(v_in in sig[j]):
                index1 = sig_prime1[j].index(v_in)
                index2 = sig_prime2[j].index(v_in)
                sig_prime1[j].insert(index1, w)
                sig_prime2[j].insert(index2+1, w)
                break
        N.append(sig_prime1)
        N.append(sig_prime2)
    return N

def exchange(v_in, sig):
    N = []
    v_sorted = get_sorted_customers(v_in, N_near)
    for i in range(len(v_sorted)):
        sig_prime1 = copy.deepcopy(sig)

        sig_prime2 = copy.deepcopy(sig)

        w = v_sorted[i]
        found = False
        for j in range(len(sig)):
            for i in range(len(sig)):
                if(v_in in sig[j] and w in sig[i]):
                    ind_v = sig[j].index(v_in)
                    ind_w = sig[i].index(w)
                    sig_prime1[i].insert(ind_w-1, v_in)
                    sig_prime1[i].remove(sig[i][ind_w-1])

                    sig_prime1[j].insert(ind_v, sig[i][ind_w-1])
                    sig_prime1[j].remove(v_in)
                    found  = True
                    break
            if(found):
                N.append(sig_prime1)
                break

        found = False
        for j in range(len(sig)):
            for i in range(len(sig)):
                if(v_in in sig[j] and w in sig[i]):
                    ind_v = sig[j].index(v_in)
                    ind_w = sig[i].index(w)
                    sig_prime2[i].insert(ind_w+1, v_in)
                    sig_prime2[i].remove(sig[i][ind_w+1])

                    sig_prime2[j].insert(ind_v, sig[i][ind_w+1])
                    sig_prime2[j].remove(v_in)
                    found  = True
                    break
            if(found):
                N.append(sig_prime2)
                break

    return N 

def get_sorted_customers(v_in, N_near):
    ##not including nodes
    all_cus= edges[int(v_in.id)][1:] ##excluding depot 
    all_cus_sorted  = sorted(all_cus, key=lambda edge: edge.d_e)
    all_cus_nodes = list(map(lambda e: nodes[e.w],all_cus_sorted))
    return all_cus_nodes[1:N_near+1]


def get_route_neighbourhood(v_in, sig, r):
    N_sig = get_composite_neighbourhood(v_in, sig)
    N_r = []
    for sig_prime in N_sig:
        if(r not in sig_prime):
            N_r.append(sig_prime)
    return N_r

def squeeze(v, sig, alpha= 1):
    sig_copy = copy.deepcopy(sig)

    N_insert = get_all_insertions(v, sig)
    print("len N_insert", len(N_insert))
    sig = get_min_penalty_sig(N_insert)
    
    while(get_penalty(sig, alpha)!=0):
        print("while, penalty: ",get_penalty(sig, alpha) )
        r = select_if_route(sig)
        N_r = get_route_neighbourhood(v,sig, r)
        sig_prime = get_min_penalty_sig(N_r, alpha)
        if(get_penalty(sig_prime, alpha)<get_penalty(sig, alpha)):
            sig = copy.deepcopy(sig_prime)
        else:
            break
    if(get_penalty(sig, alpha)!=0):
        sig = copy.deepcopy(sig_copy)
    return sig



def delete_route(sig):
    print("delete route")
    print("len sig", len(sig))
    print("len sig[0]", len(sig[0]))
    print("len sig[-1]", len(sig[-1]))

    sig_copy = copy.deepcopy(sig)
 #save
    rand_index = random.randint(0, len(sig)-1)
    rem_route = sig.pop(rand_index)

    ep = rem_route[1:-1]
    p = [1 for _ in range(N+1)]
    time = 0

    print("before while")
    print("len sig[-1]", len(sig[-1]))
    print("len sig[-2]", len(sig[-2]))

    while(len(ep)!=0 and time<maxTime):
        print("time", time)
        time +=1
        v_in = ep.pop()
        print("v_in.id", v_in.id)
        print("before N-sinrt")
        print("len sig[-1]", len(sig[-1]))
        print("len sig[-2]", len(sig[-2]))

        N_insert = get_all_feasible_insertions(v_in, sig)
        print("after N_insert")
        print("len sig[-1]", len(sig[-1]))
        print("len sig[-2]", len(sig[-2]))

        if(len(N_insert)!=0):
            print("Ninsert non zero")
            rand_index = random.randint(0, len(N_insert)-1)
            sig_prime = N_insert.pop(rand_index)
            sig = sig_prime
        
        else:
            ### squeeze mechanism
            print("squeezing", v_in, "........")
            sig = squeeze(v_in, sig)
            print("after squeeze")
            print("len sig[-1]", len(sig[-1]))
            print("len sig[-2]", len(sig[-2]))
        

        present = False
        for route in sig:
            if(v_in in route):
                present = True
                break
        
        if(not present):
            print("entering insertion ejections")
            # print(v_in.id)
            p[v_in.id] += 1
            P_best = float('inf')
            best_ejections = [1]
            best_ejections_route = 0
            best_insertion_position = 0

            for route_index, route in enumerate(sig):
                for position in range(1, len(route)):
                    
                    original_route = copy.deepcopy(route) ## contains nodes
                    # print("og route", original_route)
                    original_route.insert(position, v_in)
                    n_c = len(original_route)-2 ## no of customers
                    # print("n_c", n_c)

                    ##compute z_i ->latest possible arrival times
                    # z_i  = min(l_i, z_i+1 - c_i_i+1 - s_i)
                    # z_depot = l_depot
                    z= [0 for i in range(n_c+2)]
                    z[-1] = original_route[-1].l_v
                    for i in range(1, n_c+1)[::-1]:
                        curr_node = original_route[i]
                        next_node = original_route[i+1]

                        z[i] = min(curr_node.l_v, z[i+1]- edges[curr_node.id][next_node.id].d_e- curr_node.s_v)
                    
                    
                    ejections = [1] ### contains indices of the original route
                    a_j_prev  = 0
                    ### don't eject depot- made sure in lex
                    while(ejections !=[]):
                        print("checking ejections: ", ejections)
                        # compute P_sum
                        P_sum = 0
                        for e in ejections:
                            P_sum += p[original_route[e].id]

                        # compute a_j
                        j = ejections[-1]+1


                        ejected_route = copy.deepcopy(original_route)

                        for e in ejections[::-1]:
                            ejected_route.pop(e)
                        
                        ##j in e.r. is j-len(ejections)
                        j_ej = j-len(ejections)

                        a_0 = ejected_route[0].e_v 
                        a_prev = a_0 
                        for i in range(1, j_ej+1):
                            curr_node = ejected_route[i]
                            prev_node = ejected_route[i-1]

                            a_curr = max(a_prev+prev_node.s_v+ edges[prev_node.id][curr_node.id].d_e,curr_node.e_v )
                            
                            a_prev = a_curr

                        a_j = a_prev 

                        # compute Q_prime, aka total demand of original route
                        Q_prime = 0

                        for node in original_route:
                            Q_prime+= node.q_v
                        
                        # compute sum(q_e)
                        q_e_sum = 0
                        for e in ejections:
                            q_e_sum += original_route[e].q_v

                        ##### pruning
                            #### invalid route and subroutes
                        cond1 = (P_sum>=P_best)
                        cond2 = (original_route[j].l_v < a_j)
                        cond3 = (a_j == a_j_prev and Q_prime<=Q)

                        if(cond1 or cond2 or cond3):
                            ejections = lex_next_prune(ejections, n_c , k_max)
                            continue 
                        
                        # check feasible
                        cond4 = (a_j <=z[j])
                        cond5 = (Q_prime- q_e_sum<=Q)

                        if(cond4 and cond5):
                            a_j_prev = a_j

                            if(P_sum< P_best):
                                P_best = P_sum 
                                best_ejections = ejections
                                best_ejections_route = route_index
                                best_insertion_position = position 


                        ejections = lex_next(ejections, n_c, k_max)

            sig_new = copy.deepcopy(sig) 
            sig_new[best_ejections_route].insert( best_insertion_position, v_in)

            for e in best_ejections[::-1]:
                ep.append(sig_new[best_ejections_route][e])
                sig_new[best_ejections_route].pop(e)

            sig = copy.deepcopy(sig_new)
            # sig = perturb(sig)###?

    if(ep!=[]):
        sig = sig_copy   
        print("no change")
    return sig 



sig = delete_route(sig)

print(len(sig))
print(sig[-1])
print(sig[-2])

# for route in sig:
#     print(len(route))

import random 
from dataclasses import dataclass

from lex import lex_next, lex_next_prune, is_last_subseq
from load_instance import Node, Edge, load_data
import logging

##load from loader
nodes, edges,Q = load_data("input.txt")

N = len(nodes)-1 #no of customers
maxTime = 1000
k_max = 10
N_near=100

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
            sig_new = sig
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
            sig_new = sig
            sig_new[i].insert(j, v)
            if(check_route_feasibility(sig_new[i])):
                all_f_insertions.append(sig_new)
    return all_f_insertions

def get_min_penalty_sig(N_insert, alpha=0.99):
    min_p = get_penalty(N_insert[0])
    min_sig = N_insert[0]
    for s in N_insert:
        p = get_penalty(s, alpha)
        if(p<min_p):
            min_sig = s
            min_p = p
    return min_sig

def get_penalty(sig, alpha=0.99):
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
    # N += in_relocate(v_in, sig)
    # N += exchange(v_in, sig)
    return N


def two_opt_star(v_in, sig):
    N = []
    v_sorted = get_sorted_customers(v_in, N_near)
    for i in range(len(v_sorted)):
        sig_prime1 = sig
        sig_prime2 = sig
        w = v_sorted[i]
        found = False
        for j in range(len(sig)):
            if(found):
                break
            for i in range(len(sig)):
                if(v_in in sig[j] and w in sig[i]):
                    ind_v = sig[j].index(v_in)
                    ind_w = sig[i].index(w)
                    sig_prime1[j]= sig[j][:ind_v]+ sig[i][ind_w+1:]
                    sig_prime1[i]= sig[i][:ind_w+1]+sig[j][ind_v:]
                    found  = True
                    break
        N.append(sig_prime1)

        found = False
        for j in range(len(sig)):
            if(found):
                break
            for i in range(len(sig)):
                if(v_in in sig[j] and w in sig[i]):
                    ind_v = sig[j].index(v_in)
                    ind_w = sig[i].index(w)
                    sig_prime2[j]= sig[j][:ind_v +1]+ sig[i][ind_w:]
                    sig_prime2[i]= sig[i][:ind_w]+ sig[j][ind_v+1:]
                    found  = True
                    break
        N.append(sig_prime2)

    return N

def out_relocate(v_in, sig):
    N = []
    v_sorted = get_sorted_customers(v_in, N_near)
    for i in range(len(v_sorted)):
        sig_prime1 = sig
        sig_prime2 = sig
        w = v_sorted[i]
        for j in range(len(sig)):
            if(v_in in sig[j]):
                sig_prime1[j].remove(v_in)
                sig_prime2[j].remove(v_in)
                break
        for j in range(len(sig)):
            if(w in sig[j]):
                index = sig.index(w)
                sig_prime1.insert(index, v_in)
                sig_prime2.insert(index+1, v_in)
                break
        N.append(sig_prime1)
        N.append(sig_prime2)
    return N

def in_relocate(v_in, sig):
    N = []
    v_sorted = get_sorted_customers(v_in, N_near)
    for i in range(len(v_sorted)):
        sig_prime1 = sig
        sig_prime2 = sig
        w = v_sorted[i]
        for j in range(len(sig)):
            if(w in sig[j]):
                sig_prime1[j].remove(w)
                sig_prime2[j].remove(w)
                break
        for j in range(len(sig)):
            if(v_in in sig[j]):
                index = sig.index(v_in)
                sig_prime1.insert(index, w)
                sig_prime2.insert(index+1, w)
                break
        N.append(sig_prime1)
        N.append(sig_prime2)
    return N

def exchange(v_in, sig):
    N = []
    v_sorted = get_sorted_customers(v_in, N_near)
    for i in range(len(v_sorted)):
        sig_prime1 = sig
        sig_prime2 = sig
        w = v_sorted[i]
        found = False
        for j in range(len(sig)):
            if(found):
                break
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
        N.append(sig_prime1)


def get_sorted_customers(v_in, N_near):
    ##not including nodes
    all_cus= edges[int(v_in.id)][1:] ##excluding depot 
    all_cus_sorted  = sorted(all_cus, key=lambda edge: edge.d_e)
    return all_cus_sorted[1:N_near+1]


def get_route_neighbourhood(v_in, sig, r):
    N_sig = get_composite_neighbourhood(v_in, sig)
    N_r = []
    for sig_prime in N_sig:
        if(r not in sig_prime):
            N_r.append(sig_prime)
    return N_r

def squeeze(v, sig, alpha= 0.99):
    sig_copy = sig
    N_insert = get_all_insertions(v, sig)
    print("len N_insert", len(N_insert))
    sig = get_min_penalty_sig(N_insert)
    while(get_penalty(sig, alpha)!=0):
        r = select_if_route(sig)
        N_r = get_route_neighbourhood(v,sig, r)
        sig_prime = get_min_penalty_sig(N_r, alpha)
        if(get_penalty(sig_prime, alpha)<get_penalty(sig, alpha)):
            sig = sig_prime
        else:
            break
    if(get_penalty(sig, alpha)!=0):
        sig = sig_copy
    return sig



def delete_route(sig):
    print("delete route")
    print("len sig", len(sig))
    print("len sig[0]", len(sig[0]))
    print("len sig[-1]", len(sig[-1]))

    sig_copy = sig #save
    rand_index = random.randint(0, len(sig)-1)
    rem_route = sig.pop(rand_index)

    ep = rem_route[1:-1]
    p = [1 for _ in range(N)]
    time = 0

    while(len(ep)!=0 and time<maxTime):
        print("time", time)
        time +=1
        v_in = ep.pop()
        print("v_in.id", v_in.id)
        N_insert = get_all_feasible_insertions(v_in, sig)

        if(len(N_insert)!=0):
            rand_index = random.randint(0, len(N_insert)-1)
            sig_prime = N_insert.pop(rand_index)
            sig = sig_prime
        
        else:
            ### squeeze mechanism

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
            p[v_in.id] += 1
            P_best = float('inf')
            best_ejections = [1]
            best_ejections_route = 0
            best_insertion_position = 0

            for route in sig:
                for position in range(1, len(route)):
                    original_route = route ## contains nodes
                    original_route.insert(position, v_in)
                    n_c = len(original_route)-2 ## no of customers


                    ##compute z_i ->latest possible arrival times
                    # z_i  = min(l_i, z_i+1 - c_i_i+1 - s_i)
                    # z_depot = l_depot
                    z= [0 for i in range(n_c+2)]
                    z[-1] = original_route[-1].l_v
                    for i in range(1, n_c+1)[::-1]:
                        curr_node = original_route[i]
                        next_node = original_route[i+1]

                        z[i] = min(curr_node.l_v, z[i+1]- edges[curr_node.id][next_node.id]- curr_node.s_v)
                    
                    
                    ejections = [1] ### contains indices of the original route
                    a_j_prev  = 0
                    while(ejections !=[n_c]):
                        # compute P_sum
                        P_sum = 0
                        for e in ejections:
                            P_sum += p[original_route[e].id]

                        # compute a_j
                        j = ejections[-1]+1


                        ejected_route = original_route

                        for e in ejections[::-1]:
                            ejected_route.pop(e)
                        
                        ##j in o.r. is j-len(ejections)
                        j_ej = j-len(ejections)

                        a_0 = ejected_route[0].e_v 
                        a_prev = a_0 
                        for i in range(1, j_ej+1):
                            curr_node = ejected_route[i]
                            prev_node = ejected_route[i-1]

                            a_curr = max(a_prev+prev_node.s_v+ edges[prev_node.id][curr_node.id],curr_node.e_v )
                            
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
                                best_ejections_route = route
                                best_insertion_position = position 

                        ejections = lex_next(ejections, n_c, k_max)


            sig_new = sig 
            sig_new[best_ejections_route].insert( best_insertion_position, v_in)

            for e in best_ejections[::-1]:
                ep.append(sig_new[best_ejections_route][e])
                sig_new[best_ejections_route].pop(e)

            sig = sig_new 
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

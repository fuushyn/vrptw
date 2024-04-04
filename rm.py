 import random 
from dataclasses import dataclass
from typing import List

@dataclass
class Node:
    id: float
    s_v: float  # Service time
    q_v: float  # Demand 
    e_v: float  # Window start
    l_v: float  # Window end
    x: float # x coord
    y: float # y coord

@dataclass
class Edge:
    v: Node
    w: Node
    c_vw: float  # Travel time
    d_e: float  # Distance

N_near = 50
N = 100 # no of nodes
Q= 200 #load from loader



# array Nodes
nodes  =[] # load from loader
edges = [] # load from loader
# array Edges


def check_route_feasibility(route):
    #capacity constraint
    q = 0
    for v in route[1:-1]:
        q+= nodes[v].q_v
    
    if(q>Q):
        return False
    
    #time constraint
    a = [nodes[0].e_v for i in range(N+2)]
    for i in range(1,N+2):
        a[i]= max(a[i-1]+ nodes[i-1].s_v+ edges[i-1][i], nodes[i].e_v)
    
    for i in range(1, N+2):
        if(a[i]> nodes[i].l_v):
            return False
    
    return True

# sig = [
#     [v0, v1, v4],
#     [v0, v2, v4], 
#     [v0, v3, v4]
# ]

def check_present(v, sig):
    present = False
    for route in sig:
        if(v in route):
            present = True
    return present

def get_all_insertions(v, sig):
    all_insertions = []
    for i in range(len(sig)):
        for j in range(1, len(sig[i])):
            sig_new = sig
            sig_new[i].insert(j, v)
            all_insertions.append(sig_new)
    return all_insertions


def get_all_feasible_insertions(v, sig):
    all_f_insertions = []
    for i in range(len(sig)):
        for j in range(1, len(sig[i])):
            sig_new = sig
            sig_new[i].insert(j, v)
            if(check_route_feasibility(sig_new[i])):
                all_f_insertions.append(sig_new)
    return all_f_insertions

def get_composite_neighbourhood(sig, v): #v_in should already be present in sig
    assert check_present(v_in, sig)
    N = []
    N += two_opt_star(v_in,sig)
    N += out_relocate(v_in, sig)
    N += in_relocate(v_in, sig)
    N += exchange(v_in, sig)
    return N

def get_sorted_customers():
    pass


def two_opt_star(sig, v):
    N = []
    v_sorted = get_sorted_customers(v_in)
    for i in range(N_near):
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

def out_relocate(sig, v):
    N = []
    v_sorted = get_sorted_customers(v_in)
    for i in range(N_near):``
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

def in_relocate(sig, v):
    N = []
    v_sorted = get_sorted_customers(v_in)
    for i in range(N_near):
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

def exchange(sig, v):
    N = []
    v_sorted = get_sorted_customers(v_in)
    for i in range(N_near):
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

        found = False
        for j in range(len(sig)):
            if(found):
                break
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
        N.append(sig_prime2)

    return N

def get_route_neighbourhood(sig, r, v):
    N_sig = get_composite_neighbourhood(sig, v)
    N_r = []
    for sig_prime in N_sig:
        if(r not in sig_prime):
            N_r.append(sig_prime)
    return N_r

def get_min_penalty_sig(N_insert, alpha):
    min_p = get_penalty(N_insert[0])
    min_sig = N_insert[0]
    for s in N_insert:
        p = get_penalty(s, alpha)
        if(p<min_p):
            min_sig = s
            min_p = p
    return min_sig


def get_penalty(sig, alpha):
    #capacity violation
    q = 0
    for route in sig:
        for node in route[1:-1]:
            q+= nodes[node].q_v
    P_c = 0
    if(q>Q):
        P_c = q-Q

    #time violation
    P_tw = 0
    a = [nodes[0].e_v for i in range(N+2)]
    for i in range(1,N+2):
        a[i]= max(a[i-1]+ nodes[i-1].s_v+ edges[i-1][i], nodes[i].e_v)
        if(a[i]>nodes[i].l_v):
            P_tw+= a[i]- nodes[i].l_v
            a[i]= nodes[i].l_v
    
    F_p = P_c + alpha*P_tw
    return F_p


def select_if_route(sig):
    random.shuffle(sig)
    for index, value in enumerate(sig):
        if(check_route_feasibility(value)):
            return value
    return []

def get_insertion_ejections(sig):
    pass

def get_min_penalty_insertion_ejection(N_eject):
    pass 

def perturb(sig):
    pass


def get_earliest_arrival_times(sig):
    a_v = []
    for route in sig:
        a = []
        for index, value in enumerate(route):
            if(index ==0):
                a_i = nodes[value].e_v
                a.append(a_i)
                continue
            
            a_i = max(a[-1]+ nodes[route[index-1]].s_v + edges[route[index-1]][value], nodes[value].e_v)
            a.append(a_i)
        
        a_v.append(a)
    return a_v

def get_latest_possible_arrival_times(sig):
    ## compute z_v
    z_v = []
    for route in sig:
        z = []
        route.reverse()
        for index, value in enumerate(route):
            curr = nodes[value]
            prev = nodes[route[index-1]]

            if(index ==0):
                z.append(curr.l_v)
                continue

            z_new = min(curr.l_v, z[-1]-curr.s_v- edges[value][route[index-1]])
            z.append(z_new)
        
        route.reverse()
        z_v.append(z)
    return z_v

def squeeze(v, sig):
    sig_copy = sig
    N_insert = get_all_insertions(v, sig)
    sig = get_min_penalty_sig(N_insert)
    while(get_penalty(sig, alpha)!=0):
        r = select_if_route(sig)
        N_r = get_route_neighbourhood(sig, r, v)
        sig_prime = get_min_penalty_sig(N_r, alpha)
        if(get_penalty(sig_prime, alpha)<get_penalty(sig, alpha)):
            sig = sig_prime
        else:
            break
    if(get_penalty(sig, alpha)!=0):
        sig = sig_copy
    return sig

def delete_route(sig):
    sig_copy = sig #save
    rand_index = random.randint(0, len(sig)-1)
    rem_route = sig.pop(rand_index)

    p = [1 for i in range(N+1)]
    ep = rem_route[1:-1]
    while(ep!=[]):
        v_in = ep.pop(-1)
        N_insert = get_all_feasible_insertions(v_in, sig)
        if(N_insert!=[]):
            rand_index = random.randint(0, len(N_insert)-1)
            sig_prime = N_insert.pop(rand_index)
            sig = sig_prime

        else:
            sig = squeeze(v_in, sig)

        present = False
        for route in sig:
            if(v_in in route):
                present = True
                break
        if(not present):
            p[v_in]+=1 
            ### eject

            ## get a_i for sig-> earliest starting times
            ## a_0  = e_0
            ## a_i = max(a_i-1+ s_i-1 + c_i-1_i, e_i)

            ## get z_i for sig-> latest arrival time such that path remains feasible
            ## min_i {l_i - T(1,..,i)}  where T(1,..,i)= total travel time on path


            # a_v = []
            # z_v = []

            ## compute a_v
            # for route in sig:
            #     a = []
            #     for index, value in enumerate(route):
            #         if(index ==0):
            #             a_i = nodes[value].e_v
            #             a.append(a_i)
            #             continue
                    
            #         a_i = max(a[-1]+ nodes[route[index-1]].s_v + edges[route[index-1]][value], nodes[value].e_v)
            #         a.append(a_i)
                
            #     a_v.append(a)
            
            ## compute z_v
            # for route in sig:
            #     z = []
            #     route.reverse()
            #     for index, value in enumerate(route):
            #         curr = nodes[value]
            #         prev = nodes[route[index-1]]

            #         if(index ==0):
            #             z.append(curr.l_v)
            #             continue

            #         z_new = min(curr.l_v, z[-1]-curr.s_v- edges[value][route[index-1]])
            #         z.append(z_new)
                
            #     route.reverse()
            #     z_v.append(z)

            a_v = get_earliest_arrival_times(sig)
            z_v = get_latest_possible_arrival_times(sig)

            ## insert v and update a_v and z_v
            for route in sig:
                insertions = get_all_insertions(v_in, sig)
                for sig_prime in sig:
                    a_v = get_earliest_arrival_times(sig_prime)
                    z_v = get_latest_possible_arrival_times(sig_prime)
                    ejections = []

            ## iterate over all possible ejections lexicographically
            ### compute a_j_new in O(1)
            ### check pruning conditions
            ### return best sequence     

            def generate_subsequences(arr, k):
                n = len(arr)
                result = []
                min_p = float('inf')  # Initialize max_sum to negative infinity
                prev_subs
                prev_a

                def generate_subsequences_helper(start, current_subsequence):
                    nonlocal max_sum

                    if breaking_condition(current_subsequence):
                        return  # Skip subsequences if condition is not satisfied

                    current_sum = sum(current_subsequence)

                    if current_sum > max_sum:
                        max_sum = current_sum

                    if len(current_subsequence) > 0:
                        result.append(current_subsequence[:])

                    if len(current_subsequence) == k:
                        return  # Skip further recursion if the length is already k

                    for i in range(start, n):
                        generate_subsequences_helper(i + 1, current_subsequence + [arr[i]])

                def breaking_condition(subsequence):
                    curr_p = 0
                    for s in subsequence:
                        curr_p+= p[s]

                    cond_p = curr_p >= min_p
                    

                    if(len(subsequence) > len(old_subsequence)): ## new node ejected
                        ## update a and other conditions
                        return cond_p

                    else if(len(subsequence) == len(old_subsequence)): ## incremented last node
                        j = route.index(subsequence[-1])-1
                        j_start = j-1
                        while(route[j_start] in subsequence):
                            j_start -=1

                        ## update a_j
                        a[j] = a_prev[j]- edges[route[j+1]][route[j_start]] + edges[route[j]][route[j_start]]
                
                    # Example condition: Check if the sum is not equal to 6
                    return sum(subsequence) == 6

                generate_subsequences_helper(0, [])

                return result, max_sum


            for route in sig:








            N_eject = get_insertion_ejections(sig)
            sig, ejections = get_min_penalty_insertion_ejection(N_eject)
            ep += ejections
            sig = perturb(sig)

    if(ep!=[]):
        sig = sig_copy
    return sig



def eax(p_a, p_b):
    #p_a and p_b are parent solutions
    ## step 1 - generate graph from parents
    ## each solutions is complete => covers all nodes in nodes array

    edge_list_a = [[] for node in nodes]
    edge_list_rev_a = [[] for node in nodes]
    
    edge_list_b = [[] for node in nodes]
    edge_list_rev_b = [[] for node in nodes]

    
    vis = [0 for node in nodes]

    for route in p_a:
        for index, value in enumerate(route[:-1]):
            edge_list_a[value].append(route[index+1])
            edge_list_rev_a[route[index+1]].append(value)
    
    for route in p_b:
        for index, value in enumerate(route[:-1]):
            edge_list_b[value].append(route[index+1])
            edge_list_rev_b[route[index+1]].append(value)


    ###########!!!!! create copies of edge lists cuz being modified after this


    ## step 2 - divide edges into AB-cycles
    ab_cycles = [] ## list of edge lists
    vis_count = 1
    while(vis_count <n):
        ab_cycle = [[] for node in nodes]
        ab_cycle_rev = [[] for node in nodes]

        start = random.randint(0, n)
        if(start ==0):
            if(len(edge_list_a[start])==0):
                continue
            random.shuffle(edge_list_a[start])
        else if (start !=0 and vis[start]==1):
            continue

        first= edge_list_a[start].pop()
        ab_cycle[start].append(first)
        ab_cycle_rev[first].append(start)
        last = first
        size = 1 ## num edges in AB cycle
        while(True):
            end = 0
            if(size%2==0):
                next_node = edge_list_a[last].pop() ## what if empty, what if depot?
                if(len(ab_cycle[next_node])!=0):
                    end = 1
                vis[last] = 1
                vis_count+=1
                ab_cycle[last].append(next_node)
                ab_cycle_rev[next_node].append(last)
                last = next_node
                size+=1
                if(end==1):
                    break
            else:
                next_node = edge_list_b_rev[last].pop()
                if(len(ab_cycle[next_node])!=0):
                    end = 1

                # vis[last]= 1
                # vis_count +=1
                ## visits are counted only when an a-edge leaves the node
                ab_cycle[next_node].append(last)
                ab_cycle_rev[last].append(next_node)
                last= next_node
                size+=1
                if(end==1):
                    break
                
        ab_cycles.append(ab_cycle)
    
    ### step 3 - select E-set
    ## simple E-set
    ## block strategy

    ### simple
    eset = random.choice(ab_cycles)


    ### step 4
    intermediate = [[] for node in nodes]
    for node in nodes:
        intermediate[node] = list(set(edge_list_a_copy[node])- (set(edge_list_a_copy[node])&set(eset[node])) + (set(edge_list_b_copy[node])&set(eset[node])))
    
    ### step 5 - connect subtours to main route
    ### separate subtours from routes containing depot
    


    def separate_cycles(edge_list):
        def dfs(node, visited, current_path):
            if visited[node]:
                if node in current_path:
                    cycle = current_path[current_path.index(node):]
                    if 0 in cycle:
                        depot_cycles.append(cycle)
                    else:
                        non_depot_cycles.append(cycle)
                return
            visited[node] = True
            current_path.append(node)
            for neighbor in edge_list[node]:
                if neighbor not in current_path:  # Avoid immediate backtracking
                    dfs(neighbor, visited, current_path)
            current_path.pop()
            visited[node] = False

        n = len(edge_list)
        visited = [False] * n
        depot_cycles = []
        non_depot_cycles = []

        for i in range(n):
            if not visited[i]:
                dfs(i, visited, [])

        return depot_cycles, non_depot_cycles

    # Assuming edge_list is defined and populated with your graph data
    depot_cycles, non_depot_cycles = separate_cycles(intermediate)

    ### while all aren't merged
    
    
    # Randomly remove an element
    random_index = random.randint(0, len(non_depot_cycles) - 1)
    ndc = non_depot_cycles.pop(random_index)

    ## get list of combinations
    combinations = [] ### {depot cycles, non depot cycles}

        





def eama(n_pop, n_ch):
    m = determine_m()
    sigs = [generate_init(m) for i in range(n_pop)]
    ## choose 2 random sigs



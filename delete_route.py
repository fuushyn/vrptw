import random 

##load from loader
nodes  = []
edges = []
Q = 200 #read input

N = len(nodes)-1 #no of customers
maxTime = 100


sig = []

for i in range(1, len(nodes)):
    sig.append([nodes[0].id, nodes[i].id, nodes[0].id])



def check_route_feasibility(route):
    #capacity constraint
    q = 0
    for v in route[1:-1]:
        q+= nodes[v].q_v
    
    if(q> Q):
        return False
    
    #time constraint
    a = [nodes[0].e_v for i in range(N+2)]
    for i in range(1,N+2):
        a[i]= max(a[i-1]+ nodes[i-1].s_v+ edges[i-1][i], nodes[i].e_v)
    
    for i in range(1, N+2):
        if(a[i]> nodes[i].l_v):
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


def get_composite_neighbourhood(v_in, sig): #v_in should already be present in sig
    assert check_present(v_in, sig)
    N = []
    N += two_opt_star(v_in,sig)
    N += out_relocate(v_in, sig)
    N += in_relocate(v_in, sig)
    N += exchange(v_in, sig)
    return N


def two_opt_star(v_in, sig):
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

def out_relocate(v_in, sig):
    N = []
    v_sorted = get_sorted_customers(v_in)
    for i in range(N_near):
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

def exchange(v_in, sig):
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


def get_sorted_customers():
    pass


def get_route_neighbourhood(sig, r, v):
    N_sig = get_composite_neighbourhood(v_in, sig)
    N_r = []
    for sig_prime in N_sig:
        if(r not in sig_prime):
            N_r.append(sig_prime)
    return N_r

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


def get_insertion_ejection_neighbourhood(v_in, sig, k_max= 5):
    pass

def delete_route(sig):
    sig_copy = sig #save
    rand_index = random.randint(0, len(sig)-1)
    rem_route = sig.pop(rand_index)

    ep = rem_route[1:-1]
    p = [1 for _ in range(N)]
    time = 0

    while(len(ep)!=0 and time<maxTime):
        time +=1
        v_in = ep.pop()
        N_insert = get_all_feasible_insertions(v_in, sig)

        if(len(N_insert)!=0):
            rand_index = random.randint(0, len(N_insert)-1)
            sig_prime = N_insert.pop(rand_index)
            sig = sig_prime
        
        else:
            ### squeeze mechanism
            # pass
            sig = squeeze(v_in, sig)
        

        present = False
        for route in sig:
            if(v_in in route):
                present = True
                break
        
        if(not present):
            ### insertion ejection
            ### random ejection
            N_insert = get_all_insertions(v_in, sig)








## generate subsequences of 1..n of a given max lenght in lexicographic order


def lex_next(subseq, n, k_max):
    if(subseq[-1]<n):
        if((len(subseq)<k_max)):
            subseq.append(subseq[-1]+1)
        else:
            subseq[-1]+=1
    else:
        subseq = subseq[:-1]
        subseq[-1]+=1 
    return subseq

def lex_next_prune(subseq, n , k_max):
    ## prune and hop to higher lex
    ### eg 1...k. prune 1..k k+1, 1...k+2 and go to 1..k-1 k+1

    if(subseq[-1]<n):
        subseq[-1]+=1
    else:
        subseq = subseq[:-1]
        subseq[-1]+=1
    return subseq

def is_last_subseq(subseq, n, k_max):
    if(subseq == [n]):
        return True
    else:
        return False
    

subseq = [1,3,4]
n = 5
k_max = 3
while(not is_last_subseq(subseq, n, k_max)):
    # print(subseq)
    subseq = lex_next_prune(subseq,n, k_max)
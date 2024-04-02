from dataclasses import dataclass
import math 

@dataclass
class Node:
    id: float
    s_v: float  # Service time
    q_v: float  # Demand 
    e_v: float  # Window start
    l_v: float  # Window end
    x: float #x coord
    y: float #y coord

@dataclass
class Edge:
    v: Node
    w: Node
    c_vw: float  # Travel time
    d_e: float  # Distance


#### travel time is same as distance (https://www.sintef.no/projectweb/top/vrptw/documentation2/ )


def load_data(input_file):
    with open(input_file, 'r') as f:
        content = f.readlines()

    # print("instance: ", content[0])
    # print("vehicle and capacity: ", content[4])
    
    capacity = int(content[4].split()[1])
    nodes = []

    for i in range(9, len(content)):
        values = content[i].split()
        values = list(map(lambda x: float(x), values))
        ## values = [node_id, x, y, demand, start_time, end_time, service_time]

        new_node = Node(id=values[0], x= values[1], y = values[2], q_v= values[3], e_v = values[4], l_v = values[5], s_v= values[6])

        nodes.append(new_node)

    edges = [[] for i in range(len(nodes))]

    n = len(nodes)
    for i in range(n):
        for j in range(n):
            n1 = nodes[i]
            n2 = nodes[j]
            # print(n1.x, n2.x)
            distance = math.sqrt((n1.x-n2.x)**2 + (n1.y- n2.y)**2)

            edge_1 = Edge(v = n1.id, w = n2.id,c_vw = distance, d_e=distance)
            edges[i].append(edge_1)

    return nodes, edges, capacity

nodes, edges, capacity = load_data("input.txt")
# print(len(edges))
# print(len(nodes))
# print(capacity)        
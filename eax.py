import random

!! todo
- add graph to sig function
- add graph from sig constructor
- change cost function 



class Graph:
    def __init__(self, vertices):
        self.vertices = vertices
        self.edges = []

    def add_edge(self, u, v):
        self.edges.append((u, v))

    def remove_edge(self, u, v):
        self.edges.remove((u, v))

    def has_edge(self, u, v):
        return (u, v) in self.edges

    def get_edges(self):
        return self.edges

def create_gab(pa, pb):
    gab = Graph(pa.vertices)
    ea = set(pa.get_edges())
    eb = set(pb.get_edges())
    gab_edges = (ea.union(eb)) - (ea.intersection(eb))
    for u, v in gab_edges:
        gab.add_edge(u, v)
    return gab

def find_ab_cycle(gab, depot, pa, pb):
    start_node = random.choice(gab.vertices)
    cycle = [start_node]
    current_node = start_node
    while True:
        if current_node == depot:
            next_node = random.choice([v for u, v in gab.get_edges() if u == current_node])
        else:
            next_nodes =  [v if u==current_node else u if v==current_node for u, v in gab.get_edges()]

        if (next_nodes[0] in cycle):
            if(next_nodes[1] in cycle):
                break
            else:
                cycle.append(next_nodes[1])
                current_node = next_nodes[1]
        elif (next_nodes[1] in cycle):
            cycle.append(next_nodes[0])
            current_node = next_nodes[0]

    return cycle

def remove_ab_cycle_edges(gab, ab_cycle):
    for i in range(len(ab_cycle)):
        u, v = ab_cycle[i], ab_cycle[(i + 1) % len(ab_cycle)]
        gab.remove_edge(u, v)
        gab.remove_edge(v, u)

def generate_ab_cycles(gab, depot, pa, pb):
    ab_cycles = []
    while gab.get_edges():
        ab_cycle = find_ab_cycle(gab, depot, pa, pb)
        ab_cycles.append(ab_cycle)
        remove_ab_cycle_edges(gab, ab_cycle)
    return ab_cycles


def generate_e_sets(ab_cycles, strategy, depot):
    e_sets = []
    if strategy == "single":
        for ab_cycle in ab_cycles:
            e_sets.append([ab_cycle])
    elif strategy == "block":
        while ab_cycles:
            center_cycle = random.choice(ab_cycles)
            e_set = [center_cycle]
            for ab_cycle in ab_cycles:
                if any(node in center_cycle for node in ab_cycle if node != depot) and len(ab_cycle) < len(center_cycle):
                    e_set.append(ab_cycle)
            for ab_cycle in e_set:
                ab_cycles.remove(ab_cycle)
            e_sets.append(e_set)
    return e_sets

def create_intermediate_solution(base_solution, e_set, ea, eb):
    intermediate_solution = Graph(base_solution.vertices)
    for u, v in base_solution.get_edges():
        intermediate_solution.add_edge(u, v)
    for ab_cycle in e_set:
        for i in range(len(ab_cycle)):
            u, v = ab_cycle[i], ab_cycle[(i + 1) % len(ab_cycle)]
            if (u, v) in ea:
                intermediate_solution.remove_edge(u, v)
            elif (u, v) in eb:
                intermediate_solution.add_edge(u, v)
    return intermediate_solution

def connect_subtours(intermediate_solution, depot):
    subtours = []
    visited = set()
    for u, v in intermediate_solution.get_edges():
        if u not in visited:
            subtour = [u]
            visited.add(u)
            current_node = v
            while current_node != u:
                subtour.append(current_node)
                visited.add(current_node)
                current_node = [v for u, v in intermediate_solution.get_edges() if u == current_node][0]
            subtours.append(subtour)

    while len(subtours) > 1:
        subtour = subtours.pop(0)
        if depot not in subtour:
            best_cost = float('inf')
            best_route = None
            best_subtour_edge = None
            best_route_edge = None
            for route in subtours:
                if depot in route:
                    for i in range(len(subtour)):
                        for j in range(len(route)):
                            subtour_edge = (subtour[i], subtour[(i + 1) % len(subtour)])
                            route_edge = (route[j], route[(j + 1) % len(route)])
                            new_route = route[:j + 1] + subtour[i:] + subtour[:i + 1] + route[j + 1:]
                            cost = sum(intermediate_solution.get_cost(u, v) for u, v in zip(new_route, new_route[1:]))
                            if cost < best_cost:
                                best_cost = cost
                                best_route = route
                                best_subtour_edge = subtour_edge
                                best_route_edge = route_edge
            subtours.remove(best_route)
            best_route.remove(best_route_edge[1])
            best_route.insert(best_route.index(best_route_edge[0]) + 1, best_subtour_edge[0])
            best_route.extend(subtour[subtour.index(best_subtour_edge[1]):])
            best_route.extend(subtour[:subtour.index(best_subtour_edge[0]) + 1])
            subtours.append(best_route)

    return subtours[0]
import networkx as nx
import matplotlib
import matplotlib.pyplot as plt
import copy as cp
from operator import itemgetter
import itertools

################################################################################
# BASIC OPERTIONS ON NODES
################################################################################
def setX(G, node, value):
    G.node[node]['coords'] = (value, G.node[node]['coords'][1])

def setY(G, node, value):
    G.node[node]['coords'] = (G.node[node]['coords'][0], value)

def setCoords(G, node, x, y):
    G.node[node]['coords'] = (x,y)

def getX(G, node):
    return G.node[node]['coords'][0]

def getY(G, node):
    return G.node[node]['coords'][1]

def getCoords(G, node):
    return G.node[node]['coords']


################################################################################
# HEURISTIC GRAPH LAYERING
# return a list of list of nodes at each layer
# keep 1 node of layer 0 and move the rest to layer 1
################################################################################

# layer according to rank = eccentricity - radius
# use center as node, keep 1 node of 0 as center
def makeLayersFromRanks(G):

    layer_nodes = [[]] * (nx.diameter(G) - nx.radius(G) + 1)

    # add rank attribute as eccentricity
    for i in G.nodes:
        layer = nx.eccentricity(G, i) - nx.radius(G)
        layer_nodes[layer] = layer_nodes[layer] + [i]

    # correct layers if there are more than 1 rank-0 nodes
    if len(layer_nodes[0]) > 1:

        # add extra root nodes to an extra layer #1
        layer_nodes.insert(1, layer_nodes[0][1:])

        # delete extra root nodes
        layer_nodes[0] = [layer_nodes[0][0]]

    return layer_nodes

# layer according to rank = eccentricity - radius
# move nodes from layer 0 of higher degree to lower layer
def makeLayersFromRanks2(G):

    layer_nodes = [[]] * (nx.diameter(G) - nx.radius(G) + 1)

    # add rank attribute as eccentricity
    for i in G.nodes:
        layer = nx.eccentricity(G, i) - nx.radius(G)
        layer_nodes[layer] = layer_nodes[layer] + [i]

    # correct layers if there are more than 1 rank-0 nodes
    if len(layer_nodes[0]) > 1:
        # get node with minimum degree
        deg = min(list(G.degree(layer_nodes[0])), key=itemgetter(1))[0]
        print(deg)

        # add extra root nodes to an extra layer #1
        layer_nodes.insert(1, [x for x in layer_nodes[0] if x is not deg])

        # delete extra root nodes
        layer_nodes[0] = [deg]

    return layer_nodes


# layer according to distance to center node
def makeLayersFromDistanceToCenter(G):
    center_node = nx.center(G)[0]

    layer_nodes = [[]]*(nx.radius(G)+1)
    # calculate shortest path of all other nodes to center node
    for i in G.nodes:
        layer = nx.shortest_path_length(G, center_node, i)
        layer_nodes[layer] = layer_nodes[layer] + [i]

    return layer_nodes


################################################################################
# PLOTTING
################################################################################

# ASSIGN COORDS TO NODES FOR PLOTTING
def assignCoords(G, layer_nodes, delta_x, delta_y):
    # assign coords of each nodes
    for i in range(len(layer_nodes)):
        # find middle node
        length = len(layer_nodes[i])
        mid_node = int(length/2) + (length % 2 > 0) - 1
        for j in range(length):
            x = (j - mid_node)*delta_x
            y = -delta_y * i
            setCoords(G, layer_nodes[i][j], x, y)
            #G.node[layer_nodes[i][j]]['coords'] = (x, y)


# RECALCULATE COORDS TO VERTICALIZE EDGES WHERE POSSIBLE

def verticalizeCoords(G, layer_nodes):
    anchor = max(enumerate([len(x) for x in layer_nodes]), key=itemgetter(1))[0]

    # iterate every upper layer
    for i in range(anchor - 1,0,-1):
        # Calculate new x_coords from adjacent nodes
        prev = -10000
        for j in layer_nodes[i]:
            adj_nodes = [getX(G, x) for x in G.neighbors(j) if x in layer_nodes[i+1]]

            newX = sum(adj_nodes)/len(adj_nodes)

            if newX - prev < 1:
                newX = prev + 0.5

            setX(G, j, newX)

            prev = newX



    for i in range(anchor + 1, len(layer_nodes)):
        prev = -10000
        for j in layer_nodes[i]:
            adj_nodes = [getX(G, x) for x in G.neighbors(j) if x in layer_nodes[i-1]]

            newX = sum(adj_nodes)/len(adj_nodes)

            if newX - prev < 1:
                newX = prev + 0.5

            setX(G, j, newX)

            prev = newX


# PLOT LAYERED GRAPH
def plotLayers(G):
    pos = dict()
    color = dict()
    node_size = dict()
    #     pos = [()]*nx.number_of_nodes(G)
    #     color = ['']*nx.number_of_nodes(G)
    #     node_size = [0]*nx.number_of_nodes(G)
    for i in G:
        # pos[i] = G.node[i]['coords']
        pos[i] = getCoords(G, i)

        if G.node[i]['original'] == 'yes':
            color[i] = 'r'
            node_size[i] = 300
        else:
            color[i] = '#b3cbf2'
            node_size[i] = 200

    plt.figure(figsize=(20, 12))
    nx.draw_networkx(G, pos=pos, node_color=list(
        color.values()), node_size=list(node_size.values()))
    plt.show()


def plotFinal(G, x_width=20, y_width=14):
    pos = dict()

    # find all original nodes
    original_nodes = [x for x in list(G.nodes) if G.node[x]['original']=='yes']

    for i in G:
        pos[i] = getCoords(G, i)

    plt.figure(figsize=(x_width, y_width))
    # plot all edges
    nx.draw_networkx_edges(G, pos=pos)
    # plot original nodes
    # nx.draw_networkx_nodes(G, pos=pos, nodelist=original_nodes, node_color=list(
    #     color.values()), node_size=list(node_size.values()))

    Gx = G.subgraph(original_nodes)
    posx = dict()
    colorx = dict()
    node_sizex = dict()
    for i in Gx:
        posx[i] = getCoords(Gx, i)
        if Gx.node[i]['original'] == 'yes':
            colorx[i] = 'r'
            node_sizex[i] = 300
        else:
            colorx[i] = '#b3cbf2'
            node_sizex[i] = 200

    nx.draw_networkx(Gx, pos=posx, node_color=list(
        colorx.values()), node_size=list(node_sizex.values()))
    plt.show()



################################################################################
# CALCULATE CROSSING OF LAYERD GRAPH
################################################################################

# CALCULATE CROSSING BETWEEN 2 ADJACENT LAYERS
def countLayerCrossing(G, layer1_nodes, layer2_nodes):

    crossing = 0
    # create a subgraph containing nodes from these 2 layers
    layergraph = G.subgraph(layer1_nodes + layer2_nodes)

    # get all edges of nodes at layer 2
    layer2_adj = [layergraph.edges(i) for i in layer2_nodes]

    # get adjacent nodes of each node at layer 2
    layer2_adj = [[x[1] for x in y] for y in layer2_adj]

    # represent adjacent nodes by their indices (orders) on layer 1
    layer2_adj = [[layer1_nodes.index(x) for x in y] for y in layer2_adj]

    #??? To improve: sort each sublist

    # scan through second layer, check every pair left to right
    for i in range(len(layer2_nodes) - 1):
        # this is the left node
        # get coords of its adjacent nodes on layer 1
        left = layer2_adj[i]

        for j in range(i + 1, len(layer2_nodes)):
            # this is the right node
            # get coords of its adjacent nodes on layer 1
            right = layer2_adj[j]

            # count crossing for each edge on the right node
            crossing += sum(sum(x > y for x in left) for y in right)

    return crossing


# CALCULATE CROSSING OF THE WHOLE GRAPH
def countAllCrossing(G, layer_nodes):
    return [countLayerCrossing(G, layer_nodes[i], layer_nodes[i + 1]) for i in range(len(layer_nodes) - 1)]




################################################################################
# REMOVE CONNECTED NODES OF SAME LAYERS
################################################################################

# by retaining the maximum independent set and move others to newly
# created layer next
def removeFlatEdges(G, layer_nodes, layer, trial_count):

    # construct a subgraph of this layer
    layer_graph = nx.subgraph(G, layer_nodes[layer])

    # find maximal independent set
    # since the algorithm is not optimal, repeat trial_count times
    ind_set = []

    for i in range(trial_count):
        ind_set_new = nx.maximal_independent_set(layer_graph)
        if len(ind_set) < len(ind_set_new):
            ind_set = ind_set_new

    # return if there is no flat edges
    if len(ind_set) == len(layer_nodes[layer]):
        return

    # find nodes to move to another layer
    new_layer_nodes = [x for x in layer_nodes[layer] if x not in ind_set]

    # update the original layer with independent set
    layer_nodes[layer] = ind_set

    # add remaining nodes to a new layer
    layer_nodes.insert(layer + 1, new_layer_nodes)



# Retain max independent set from nodes connected to direct upper layer
def removeFlatEdgesConnected(G, layer_nodes, layer, trial_count):
    if len(layer_nodes[layer]) == 1:
        return
    # get neighbors of all nodes of layer k - 1
    neighbors_upper = [i for j in layer_nodes[layer - 1]
                       for i in G.neighbors(j)]
    # remove nodes from this layer that is not connected to any nodes from
    # upper
    new_layer = [i for i in layer_nodes[layer] if i not in neighbors_upper]
    # get remaining nodes
    nodes_k = []
    if len(new_layer) != len(layer_nodes[layer]):
        nodes_k = [i for i in layer_nodes[layer] if i not in new_layer]
    else:
        nodes_k = new_layer
        new_layer = []
    # if nodes_k has only 1 node
    if len(nodes_k) > 1:
        # construct a subgraph of this layer
        layerk_graph = nx.subgraph(G, nodes_k)

        # find maximal independent set
        # since the algorithm is not optimal, repeat trial_count times
        ind_set = []

        for i in range(trial_count):
            ind_set_new = nx.maximal_independent_set(layerk_graph)
            if len(ind_set) < len(ind_set_new):
                ind_set = ind_set_new
    else:
        ind_set = nodes_k

    # # return if there is no flat edges
    # if len(ind_set) == len(nodes_k):
    #     return

    # find nodes to move to next layer
    new_layer += [x for x in nodes_k if x not in ind_set]

    if new_layer == []:
        return
    # update the original layer with independent set
    layer_nodes[layer] = ind_set

    # add remaining nodes to a new layer
    layer_nodes.insert(layer + 1, new_layer)


# Retain max independent set from nodes connected to upper layers
# Move to lower node without creating new layers
def removeFlatEdgesConnected2(G, layer_nodes, layer, trial_count):

    # get neighbors of all nodes of layer k - 1
    upper_nodes = list(itertools.chain.from_iterable(layer_nodes[:layer]))
    neighbors_upper = [i for j in upper_nodes
                       for i in G.neighbors(j)]
    # remove nodes from this layer that is not connected to any nodes from
    # upper
    new_layer = [i for i in layer_nodes[layer] if i not in neighbors_upper]
    # get remaining nodes
    nodes_k = []
    if len(new_layer) != len(layer_nodes[layer]):
        nodes_k = [i for i in layer_nodes[layer] if i not in new_layer]
    else:
        nodes_k = new_layer
        new_layer = []

    # construct a subgraph of this layer
    layerk_graph = nx.subgraph(G, nodes_k)

    # find maximal independent set
    # since the algorithm is not optimal, repeat trial_count times
    ind_set = []

    for i in range(trial_count):
        ind_set_new = nx.maximal_independent_set(layerk_graph)
        if len(ind_set) < len(ind_set_new):
            ind_set = ind_set_new

    # # return if there is no flat edges
    # if len(ind_set) == len(nodes_k):
    #     return

    # find nodes to move to next layer
    new_layer += [x for x in nodes_k if x not in ind_set]

    if new_layer == []:
        return
    # update the original layer with independent set
    layer_nodes[layer] = ind_set

    # add remaining nodes to a new layer
    layer_nodes.insert(layer + 1, new_layer)



################################################################################
# NODE PROMOTION
# Move a node to upper/lower layers when it does not have any connnection with
# lower/upper layer and if it does not create a new flat edges
# Sweep from bottom to top, scan each node
################################################################################
def promoteNodes(G, layer_nodes):
    num_layer = len(layer_nodes)
    # get neighbors of all nodes for each layer
    neighbors =  [[x for y in layer_nodes[z] for x in G.neighbors(y)] for z in range(num_layer)]

    for i in range(num_layer-1, 0, -1): # layer indice
        # Get all neighbors of all lower nodes
        neighbors_lower = []
        if i != num_layer - 1:
            neighbors_lower = [x for y in neighbors[i+1:] for x in y]
            neighbors_lower = list(set(neighbors_lower))
        move_list = []
        for j in layer_nodes[i]: # each node in layer i
            # skip lower connection check for bottom layer
            if j not in neighbors_lower:
                if j not in neighbors[i-1]:
                    # Add node j to list to move
                    move_list.append(j)
        # Add move list to upper layer (i-1)
        layer_nodes[i-1] += move_list
        # Remove move_list from current layer (i)
        layer_nodes[i] = [x for x in layer_nodes[i] if x not in move_list]

    return None

################################################################################
# REMOVE LONG EDGES BY ADDING DUMMY NODES
# dummy nodes has feature 'original' = 'no'
################################################################################

def addDummies(Gx, layersx, layer):
    max_node = max(Gx.nodes) + 1

    for i in layersx[layer]:
        adj_nodes = list(Gx.neighbors(i))

        for j in range(layer + 2, len(layersx)):
            # get connected nodes in this layer
            adj = [x for x in adj_nodes if x in layersx[j]]

            if not adj:
                continue

            # add dummies to each adjacent node
            for k in adj:
                # remove original edges
                Gx.remove_edge(i, k)

                # add 1 dummy node to each layer crossing
                for l in range(layer + 1, j):

                    # add node to graph and layer_nodes
                    Gx.add_node(max_node)
                    Gx.nodes[max_node]['original'] = 'no'
                    layersx[l].append(max_node)

                    # add left edge
                    if l == layer + 1:
                        Gx.add_edge(i, max_node)
                    else:
                        Gx.add_edge(max_node - 1, max_node)

                    # add right edge
                    if l == j - 1:
                        Gx.add_edge(max_node, k)
                    max_node += 1


################################################################################
# LAYER-BY-LAYER SWEEPING
# fix the adjacent layer(s) and rearranging the layer using barycenter method
# Method: calculate average of coords of nodes (on adj layer(s)) that are adjacent to a node on the layer
# then sort the layer's node list by this value
################################################################################


# 1 sided sweeping of 1 layer
# centroid is calculated from only upper layer
def sweepLayer1Sided(G, layer_nodes, layer, godown):

    if godown == True:
        prev = layer - 1
    else:
        prev = layer + 1

    center_list = []

    for i in layer_nodes[layer]:
        # get all adjacent nodes from upper layer
        adj_nodes = [x for x in layer_nodes[prev] if x in G.neighbors(i)]

        # accquire indices of adjacent nodes in upper layer
        ordering = [layer_nodes[prev].index(x) for x in adj_nodes]

        # calculate barycenter and save to center_list
        if not ordering:
            center_list = center_list + [0]
        else:
            center_list = center_list + [sum(ordering) / len(ordering)]
    layer_nodes[layer] = [x for a, x in sorted(
        zip(center_list, layer_nodes[layer]))]

    return None

# 2-sided sweeping of 1 layer
# centroid is calculated from upper and lower layers
def sweepLayer2Sided(G, layer_nodes, layer):

    center_list = []

    for i in layer_nodes[layer]:
        # get all adjacent nodes from upper, lower layer
        adj_nodes_upper = [x for x in layer_nodes[layer-1] if x in G.neighbors(i)]
        adj_nodes_lower = [x for x in layer_nodes[layer+1] if x in G.neighbors(i)]

        # accquire indices of adjacent nodes in upper, lower layers
        ordering_upper = [layer_nodes[layer-1].index(x) for x in adj_nodes_upper]
        ordering_lower = [layer_nodes[layer+1].index(x) for x in adj_nodes_lower]

        # calculate barycenter and save to center_list
        # assume that every node has at least 1 connection

        center_list += [(sum(ordering_upper) + sum(ordering_lower)) / (len(ordering_upper) + len(ordering_lower))]
    layer_nodes[layer] = [x for a, x in sorted(zip(center_list, layer_nodes[layer]))]

    return None


# 1-sided sweeping of the whole graph
def sweepGraph1Sided(G, layer_nodes, crossing, fromlayer, tolayer, godown, condition):
    if godown:
        myrange = range(fromlayer, tolayer+1)
    else:
        myrange = range(fromlayer, tolayer-1, -1)
    for i in myrange:
        layer_copy = layer_nodes[i][:]
        sweepLayer1Sided(G, layer_nodes, i, godown)
        if condition:
            if i + 1 >= len(layer_nodes):
                lower_crossing = 0
                lower_crossing_old = 0
            else:
                lower_crossing = countLayerCrossing(G, layer_nodes[i], layer_nodes[i+1])
                lower_crossing_old = crossing[i]
            if i - 1 < 0:
                upper_crossing = 0
                upper_crossing_old = 0
            else:
                upper_crossing = countLayerCrossing(G, layer_nodes[i-1], layer_nodes[i])
                upper_crossing_old = crossing[i-1]

            # revert the sweeping if crossing not reduced
            if lower_crossing + upper_crossing > upper_crossing_old + lower_crossing_old:
                layer_nodes[i] = layer_copy[:]
                layer_copy = []
            else: # update CROSSING if reduced
                if i - 1 >= 0:
                    crossing[i-1] = upper_crossing
                if i + 1 < len(layer_nodes):
                    crossing[i] = lower_crossing

    return None


# 2-sided sweeping of the whole graph
def sweepGraph2Sided(G, layer_nodes, crossing, godown):
    if godown:
        myrange = range(2, len(layer_nodes)-1)
    else:
        myrange = range(len(layer_nodes)-2, 0, -1)
    for i in myrange:
        layer_copy = layer_nodes[i][:]
        sweepLayer2Sided(G, layer_nodes, i)

        lower_crossing = countLayerCrossing(G, layer_nodes[i], layer_nodes[i+1])
        upper_crossing = countLayerCrossing(G, layer_nodes[i-1], layer_nodes[i])
        # revert the sweeping if crossing not reduced
        if lower_crossing + upper_crossing > crossing[i-1] + crossing[i]:
            layer_nodes[i] = layer_copy[:]
            layer_copy = []
        else: # update CROSSING if reduced
            crossing[i-1] = upper_crossing
            crossing[i] = lower_crossing

    return None


################################################################################
# KERNELS FOR EXPERIMENT
################################################################################

def reduceKernel1Sided(G, layerFunc, flatFunc, sweepcycle):

    layer_nodes = makeLayersFromRanks(G)
    if layerFunc == 2:
        layer_nodes = makeLayersFromRanks2(G)
    elif layerFunc == 3:
        layer_nodes = makeLayersFromDistanceToCenter(G)
    i = 1
    while i < len(layer_nodes):
        # assignCoords(G, layer_nodes, 1, 1, 20)
        # plotLayers(G)
        if flatFunc == 1:
            removeFlatEdgesConnected(G, layer_nodes, i, 100)
        else:
            removeFlatEdgesConnected2(G, layer_nodes, i, 100)
        i += 1
    assignCoords(G, layer_nodes, 1, 1, 20)
    plotLayers(G)

    num_layer = len(layer_nodes)

    for i in range(0, num_layer):
        addDummies(G, layer_nodes, i)
    assignCoords(G, layer_nodes, 1, 1, 20)
    plotLayers(G)


    # calculate total crossing before any sweeping
    crossing = countAllCrossing(G, layer_nodes)
    print("Crossing before sweep: ", sum(crossing))

    # 1st top down unconditional sweeping
    sweepGraph1Sided(G, layer_nodes, [], 2, num_layer-2, godown=True, condition=False)
    crossing = countAllCrossing(G, layer_nodes)
    print("Crossing after top down sweep #1: ", sum(crossing))

    # 1st bottom up conditional sweeping
    sweepGraph1Sided(G, layer_nodes, crossing, num_layer-2, 1, godown=False, condition=True)
    crossing = countAllCrossing(G, layer_nodes)
    print("Crossing after bottom up sweep #1: ", sum(crossing))
    pre_crossing = sum(crossing)
    for i in range(sweepcycle-1):
        sweepGraph1Sided(G, layer_nodes, crossing, 2, num_layer-2, godown=True, condition=True)
        print("Crossing after top down sweep #", i+2, " : ", sum(crossing))
        sweepGraph1Sided(G, layer_nodes, crossing, num_layer-2, 1, godown=False, condition=True)
        crossing = countAllCrossing(G, layer_nodes)
        print("Crossing after bottom up sweep #", i+2, " : ", sum(crossing))

        if sum(crossing) < pre_crossing:
            pre_crossing = sum(crossing)
        else:
            break

    assignCoords(G, layer_nodes, 1, 1, 20)
    plotLayers(G)

    return None


def reduceKernel2Sided(G, layerFunc, flatFunc, sweepcycle, toPromote, toPlot):
    # Layer graph
    layer_nodes = makeLayersFromRanks(G)
    if layerFunc == 2:
        layer_nodes = makeLayersFromRanks2(G)
    elif layerFunc == 3:
        layer_nodes = makeLayersFromDistanceToCenter(G)

    # plot layered graph
    if toPlot:
        assignCoords(G, layer_nodes, 1, 1, 20)
        plotLayers(G)

    # Remove flat edges
    i = 1
    while i < len(layer_nodes):
        # assignCoords(G, layer_nodes, 1, 1, 20)
        # plotLayers(G)
        if flatFunc == 1:
            removeFlatEdgesConnected(G, layer_nodes, i, 100)
        else:
            removeFlatEdgesConnected2(G, layer_nodes, i, 100)
        i += 1

    # plot new graph
    if toPlot:
        assignCoords(G, layer_nodes, 1, 1, 20)
        plotLayers(G)

    num_layer = len(layer_nodes)

    if toPromote:
        # Promote nodes
        promoteNodes(G, layer_nodes)

        # plot new graph
        if toPlot:
            assignCoords(G, layer_nodes, 1, 1, 20)
            plotLayers(G)

    # Remove edges crossing over a layer by adding a dummy node
    for i in range(0, num_layer):
        addDummies(G, layer_nodes, i)

    if toPlot:
        assignCoords(G, layer_nodes, 1, 1, 20)
        plotLayers(G)

    # calculate total crossing before any sweeping
    crossing = countAllCrossing(G, layer_nodes)
    print("Crossing before sweep: ", sum(crossing))

    # 1st top down unconditional sweeping
    sweepGraph1Sided(G, layer_nodes, crossing, 2, num_layer-1, godown=True, condition=False)
    crossing = countAllCrossing(G, layer_nodes)
    print("Crossing after top down sweep #1: ", sum(crossing))

    # 1st bottom up conditional sweeping
    sweepGraph1Sided(G, layer_nodes, crossing, num_layer-2, 1, godown=False, condition=True)
    crossing = countAllCrossing(G, layer_nodes)
    print("Crossing after bottom up sweep #1: ", sum(crossing))

    for i in range(sweepcycle):
        sweepGraph2Sided(G, layer_nodes, crossing, godown=True)
        print("Crossing after top down 2-sided sweep #", i+1, " : ", sum(crossing))
        sweepGraph2Sided(G, layer_nodes, crossing,godown=False)
        print("Crossing after bottom up 2-sided sweep #", i+1, " : ", sum(crossing))

    # 2nd top down unconditional sweeping
    sweepGraph1Sided(G, layer_nodes, crossing, 2, num_layer-1, godown=True, condition=True)
    crossing = countAllCrossing(G, layer_nodes)
    print("Crossing after top down sweep 2nd: ", sum(crossing))

    # 2nd bottom up conditional sweeping
    sweepGraph1Sided(G, layer_nodes, crossing, num_layer-2, 1, godown=False, condition=True)
    crossing = countAllCrossing(G, layer_nodes)
    print("Crossing after bottom up sweep 2nd: ", sum(crossing))

    if toPlot:
        assignCoords(G, layer_nodes, 1, 1, 20)
        plotLayers(G)

    return None

def reduceCrossing(G, layerFunc, flatFunc, toPrint=False, toPlot=False):
    # Layer graph
    layer_nodes = makeLayersFromRanks(G)
    if layerFunc == 2:
        layer_nodes = makeLayersFromRanks2(G)
    elif layerFunc == 3:
        layer_nodes = makeLayersFromDistanceToCenter(G)

    # plot layered graph
    if toPlot:
        assignCoords(G, layer_nodes, 1, 1)
        plotLayers(G)

    # Remove flat edges
    i = 1
    while i < len(layer_nodes):
        # assignCoords(G, layer_nodes, 1, 1, 20)
        # plotLayers(G)
        if flatFunc == 1:
            removeFlatEdgesConnected(G, layer_nodes, i, 100)
        else:
            removeFlatEdgesConnected2(G, layer_nodes, i, 100)
        i += 1

    # plot new graph
    if toPlot:
       assignCoords(G, layer_nodes, 1, 1)
       plotLayers(G)

    num_layer = len(layer_nodes)

    # Remove edges crossing over a layer by adding a dummy node
    for i in range(0, num_layer):
        addDummies(G, layer_nodes, i)

    if toPlot:
        assignCoords(G, layer_nodes, 1, 1)
        plotLayers(G)


    # calculate total crossing before any sweeping
    crossing = countAllCrossing(G, layer_nodes)
    if toPrint:
        print("Crossing before sweep: ", sum(crossing))

    # Store current crossing
    prev_cross_outer = sum(crossing)


    # 1st top down unconditional sweeping
    sweepGraph1Sided(G, layer_nodes, crossing, 2, num_layer-1, godown=True, condition=False)
    crossing = countAllCrossing(G, layer_nodes)
    if toPrint:
        print("Top down 1-sided sweep: ", sum(crossing))

    # 1st bottom up conditional sweeping
    sweepGraph1Sided(G, layer_nodes, crossing, num_layer-2, 1, godown=False, condition=True)
    crossing = countAllCrossing(G, layer_nodes)
    if toPrint:
        print("Bottom up 1-sided sweep: ", sum(crossing))

    # Sweep 2 ways until no further crossing reduction
    while True:
        prev_cross_inner = sum(crossing)
        # Do 1 layer sweep up and down until no further crossing reduction
        while True:
            # 1st top down unconditional sweeping
            sweepGraph1Sided(G, layer_nodes, crossing, 2, num_layer-1, godown=True, condition=True)
            crossing = countAllCrossing(G, layer_nodes)
            if toPrint:
                print("Top down 1-sided sweep: ", sum(crossing))

            # 1st bottom up conditional sweeping
            sweepGraph1Sided(G, layer_nodes, crossing, num_layer-2, 1, godown=False, condition=True)
            crossing = countAllCrossing(G, layer_nodes)
            if toPrint:
                print("Bottom up 1-sided sweep: ", sum(crossing))

            # Break if no crossing reduction
            new_crossing = sum(crossing)
            if new_crossing >= prev_cross_inner:
                break

            prev_cross_inner = new_crossing

        # Do 2-sided sweep until no improvement
        while True:
            sweepGraph2Sided(G, layer_nodes, crossing, godown=True)
            crossing = countAllCrossing(G, layer_nodes)
            if toPrint:
                print("Top down 2-sided sweep: ", sum(crossing))

            sweepGraph2Sided(G, layer_nodes, crossing, godown=False)
            crossing = countAllCrossing(G, layer_nodes)
            if toPrint:
                print("Bottom up 2-sided sweep: ", sum(crossing))

            # Break if no crossing reduction
            new_crossing = sum(crossing)
            if new_crossing >= prev_cross_inner:
                break

            prev_cross_inner = new_crossing

        # Break if no crossing reduction
        if prev_cross_outer <= prev_cross_inner:
            break
        if toPrint:
            print("Prev cross outer: ", prev_cross_outer)
            print("Prev cross inner: ", prev_cross_inner)

        prev_cross_outer = prev_cross_inner
        # break

    if toPlot:
        assignCoords(G, layer_nodes, 1, 1)
        plotLayers(G)

    assignCoords(G, layer_nodes, 1, 1)
    verticalizeCoords(G, layer_nodes)
    #plotFinal(G, x_width=x_width, y_width=y_width)

    return sum(crossing)


# create a graph for testing
def makeGraph(a, b, c, d, e, f, myseed, toPlot):
    G = nx.random_partition_graph([a, b, c, d], e, f, myseed)
    if toPlot:
        plt.figure(figsize=(20, 12))
        nx.draw(G, with_labels=True)
        plt.show()

    # add originality feature to every node
    # 'original' = 'yes' for all original node
    for i in G.nodes:
        G.nodes[i]['original'] = 'yes'
    return G
    # find all original nodes
    original_nodes = [x for x in list(G.nodes) if G.node[x]['original']=='yes']

    for i in G:
        pos[i] = G.node[i]['coords']

    plt.figure(figsize=(20, 12))
    # plot all edges
    nx.draw_networkx_edges(G, pos=pos)
    # plot original nodes
    # nx.draw_networkx_nodes(G, pos=pos, nodelist=original_nodes, node_color=list(
    #     color.values()), node_size=list(node_size.values()))

    Gx = G.subgraph(original_nodes)
    posx = dict()
    colorx = dict()
    node_sizex = dict()
    for i in Gx:
        posx[i] = Gx.node[i]['coords']
        if Gx.node[i]['original'] == 'yes':
            colorx[i] = 'r'
            node_sizex[i] = 300
        else:
            colorx[i] = '#b3cbf2'
            node_sizex[i] = 200

    nx.draw_networkx(Gx, pos=posx, node_color=list(
        colorx.values()), node_size=list(node_sizex.values()))
    plt.show()


#############################################################################
# HOW TO USE - SINGLE RUN
# RUN:
#       for i in G.nodes:
#           G.nodes[i]['original'] = 'yes'
#       total_crossing = reduceCrossing(G, layerFunc=3, flatFunc=1, toPrint=False, toPlot=False)
#       plotFinal(mygraph)
# G: networkx graph object
# toPrint=True to print crossing after each iteration, default: False
# toPlot=True to plot a graph after each step, default: False
# other inputs: use default
#
#
# MULTIPLE RUN FOR THE BEST RESULT
#       for i in G.nodes:
#           G.nodes[i]['original'] = 'yes'
#
#       min_crossing = 10000000
#       minG = 0
#
#       for i in range(iter):
#           Gx = cp.deepcopy(G)
#           crossing = reduceCrossing(Gx, 3, 1, toPrint=False, toPlot=False)
#           if crossing < min_crossing:
#               min_crossing = crossing
#               minG = cp.deepcopy(Gx)
#
#       print("Final crossing: ", min_crossing)
#       plotFinal(minG)
#
# iter: number of tests
#############################################################################



# [5,10,5,5] seed = 1
# G1 = makeGraph(5, 10, 5, 5, 0.5, 0.05, 1, toPlot=False)
# reduceKernel2Sided(G1, layerFunc=1, flatFunc=1, sweepcycle=2, toPromote=False, toPlot=False)def assignCoords(G, layer_nodes, delta_x, delta_y, x_width):

# G2 = makeGraph(10, 5, 5, 5, 0.5, 0.05, 1, toPlot=False)
# reduceKernel2Sided(G2, 1, 1, 3, toPromote=True, toPlot=False)


#
# G3 = makeGraph(5, 10, 5, 5, 0.5, 0.05, 1, toPlot=True)

# reduceCrossing(G3, 3, 1, toPrint=False, toPlot=False)
# plotFinal(G3)

# TEST WITH REAL DATA FROM CICELY

G = nx.Graph()

G.add_nodes_from([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30])

G.add_edges_from([(1, 2), (1, 3), (1, 4), (1, 5), (1, 6), (1, 7), (1, 8), (1, 9), (1, 10), (1, 11), (1, 12), (1, 13), (1, 14), (1, 15), (1, 16), (1, 17), (1, 18), (1, 19), (2, 18), (2, 3), (2, 27), (2, 28), (3, 18), (3, 23), (3, 26), (3, 27), (3, 28), (3, 4), (3, 9), (3, 30), (4, 18), (4, 27), (4, 26), (5, 6), (6, 18), (6, 16), (7, 18), (7, 8), (8, 18), (8, 15), (8, 29), (9, 18), (9, 27), (9, 30), (10, 18), (10, 11), (11, 18), (12, 18), (12, 13), (13, 18), (13, 14), (14, 18), (15, 18), (15, 29), (17, 18), (18, 20), (18, 21), (18, 22), (18, 23), (18, 19),  (18, 24), (18, 25), (20, 21), (23, 26)])


plt.figure(figsize=(20, 12))
nx.draw(G, with_labels=True)
plt.show()

# add originality feature to every node
# 'original' = 'yes' for all original node
for i in G.nodes:
    G.nodes[i]['original'] = 'yes'

min_crossing = 10000000
minG = 0

for i in range(10):
    Gx = cp.deepcopy(G)
    crossing = reduceCrossing(Gx, 3, 1, toPrint=False, toPlot=False)
    if crossing < min_crossing:
        min_crossing = crossing
        minG = cp.deepcopy(Gx)

print("Final crossing: ", min_crossing)
plotFinal(minG)

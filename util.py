'''
This class provides some general utily functions
'''
import numpy as np
import osmnx as ox
import networkx as nx
from math import sin, cos, sqrt, atan2, radians
from haversine import haversine, Unit
import geopy
import geopy.distance


def get_node_id(G, n):
    '''
    Returns the id of a node
    :param G: The graph
    :param n: The node of which you want to know the id
    :return: The id
    '''
    index = list(G.nodes()).index(n)
    node_id = list(G.nodes)[index]
    return node_id


def get_straight_line_distance(G, n1, n2):
    '''
    Returns the straight line distance between two GPS points using the haversine formular
    :param G: The graph
    :param n1: Point 1
    :param n2: Point 2
    :return: Distance between point 1 and point 2 in meters
    '''
    lat1 = G.nodes[n1]['y']
    lon1 = G.nodes[n1]['x']
    lat2 = G.nodes[n2]['y']
    lon2 = G.nodes[n2]['x']

    origin = (lat1, lon1)
    destination = (lat2, lon2)

    return haversine(origin, destination, unit=Unit.METERS)


def is_deadend(g, node):
    '''
    Returns true if the node is a dead-end
    :param g: the Graph
    :param node: The node of interest
    :return: True if dead end, False if not
    '''
    _neighbors = list(g.neighbors(node))
    if node in _neighbors:
        _neighbors.remove(node) #remove self-loop
    return len(_neighbors) <= 1


def choose_edge(G, edges, current, next, destination, method):
    '''
    This function emulates the choosing behaviour of the agents for the retracing of the path
    For every path there are nodes in the path. Those nodes can be connected by more than one edge
    Depending on the agent/method a different edge might be chosen: e.g the LA agent chooses the edge
    according to the angle. So whenever the path is recreated and there are multiple edges between the
    nodes. Thins function check which one is correct.

    :param G: The graph
    :param edges: The edges between which should be decided
    :param current: The current node/position
    :param next: The next node/position
    :param destination: The overall destination of the path
    :param method: The method depending on the agent
    :return: the edge to choose
    '''
    best_edge = None

    if method == "angle":
        best_angle = float("inf")

        # Get the GPS point of the destination
        dest_lat = G.nodes[destination]['y']
        dest_lon = G.nodes[destination]['x']
        destination_point = (dest_lat, dest_lon)

        # Get the GPS point of the current location
        curr_lat = G.nodes[current]['y']
        curr_lon = G.nodes[current]['x']
        current_point = (curr_lat, curr_lon)

        # Bearing from our current position to the destination
        bearing_curr_dest = ox.bearing.calculate_bearing(curr_lat, curr_lon, dest_lat, dest_lon)

        for i in range(len(edges)):

            try:
                edge_geometry = edges[i]['geometry']
                line_string = str(edge_geometry).replace(",", "").replace("(", "").replace(")", "").replace(
                    "LINESTRING", "")
                coord_list = line_string.lstrip().split(" ")
                lat = float(coord_list[3])
                lon = float(coord_list[2])
            except:
                lat = G.nodes[next]['y']
                lon = G.nodes[next]['x']

            # Get bearing to the next
            neighbour_point = (lat, lon)
            bearing1 = ox.bearing.calculate_bearing(curr_lat, curr_lon, lat, lon)

            # Calculate the angle to the neighbour
            angle = abs((bearing_curr_dest - bearing1 + 180) % 360 - 180)
            if angle < best_angle:
                best_angle = angle
                best_edge = edges[i]

    elif method == "distance":
        best_length = float("inf")
        for i in range(len(edges)):

            edge_length = edges[i]["length"]

            if edge_length < best_length:
                best_length = edge_length
                best_edge = edges[i]

    else:
        print("Major error, there was no edge length whatsoever")

    return best_edge


# Returns the length of a route in meters
def get_length_of_path(G, route, method):
    '''
    Calculates the length of a path. A path is just a list of nodes
    Each edge between two nodes has a length attribute which stores the length between
    them in meters. This function sums up all those lengths.
    :param G: The graph
    :param route: The route of interest, usually created by an agent
    :param method: The method the agent uses
    :return: The total length of the path
    '''
    start = route[0]
    destination = route[-1]
    total_length = 0.0

    # Loop through all nodes and get the appropriate edges
    for i in range(len(route)-1):
        source = route[i]
        target = route[i+1]

        # Check if there is an edge, there might not be one in rare cases
        if G.has_edge(source, target):
            # Check if there is more than one edge connecting the two nodes
            edges = G.get_edge_data(source, target)
            if len(edges) > 1:
                edge = choose_edge(G, edges, source, target, destination, method)
            else:
                edge = G[source][target][0]

        # If there is no edge there might be one backwards
        else:
            # Switch source and target
            if G.has_edge(target, source):
                edge = G[target][source][0]
            else:
                # No idea what would happen here
                print("Weird problem, there seems to be no edge whatsoever")
        length = edge["length"]
        total_length += length
    return total_length


def get_graph_properties(G):
    '''
    Get some random graph porpteries like numer of nodes etc.
    :param G: The graph
    :return: the properties
    '''
    return ox.stats.extended_stats(G, connectivity=False, anc=False, ecc=False, bc=False, cc=False)
    # print(f"Average Shortest Path Length: {nx.average_shortest_path_length(G, weight='length')}")

def get_surrounding_coords(center, direction):

    lat = center[0]
    lon = center[1]

    # Define starting point.
    start = geopy.Point(lat, lon)

    # Define a general distance object, initialized with a distance of 1 km.
    d = geopy.distance.VincentyDistance(kilometers=10)

    # Use the `destination` method with a bearing of 0 degrees (which is north)
    # in order to go from point `start` 1 km to north.
    dest = d.destination(point=start, bearing=direction)
    dest_lat = dest.latitude
    dest_lon = dest.longitude

    return(dest_lat, dest_lon)


# Exports route(s) on a leaflet map in a html file
def visualize_routes(G, routes, opacity=1, name="route", color="#CC0000", weight=10):
    '''
    Creates a heatmap stlyed leaflet map with given routes based on the given Graph.
    Start and destination of each route must be the same.
    Output of the function is a HTML-file with given name.
    :param G: The graph
    :param routes: The routes to be printed on the leaflet map as list of routes
    :param opacity: Percentage of the opacity of the printed route
    :param name: The name of tge output file (HTML)
    :param color: Color of the route on the leaflet map given in hex
    :param weight: Weight of the route line
    '''

    route_list = []

    for i in range(len(routes)):
        if (i % 2) == 0:
            if i == 0:
                route_list.extend(routes[i])
            else:
                faulty_route = routes[i]
                faulty_route.pop(0)
                route_list.extend(faulty_route)
        else:
            faulty_route = routes[i]
            faulty_route.reverse()
            faulty_route.pop(0)
            route_list.extend(faulty_route)

    route_map = ox.plot_route_folium(G, route_list, color=color, opacity=opacity, weight=weight)
    route_map.save(name+'.html')

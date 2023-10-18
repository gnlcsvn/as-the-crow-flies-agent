import util
import osmnx as ox
import matplotlib.pyplot as plt


class LeastAngleAgent:
    """
    The LeastAngle agent implements an agent that applies the LA strategy
    by Hochmair to reach a destination. This implementation behaves like so:
    As long as the current position does not match the target position, the
    agent will scan its neighbours and choose the best unvisited neighbour in
    terms of deviation to the target angle. Once all neighbours have been visited
    and the agent reaches the node again, it continues with the next best already
    visited node. If at any point the agent visits any two nodes in the same sequence
    twice and the target node is already fully visited, the agents interrupts the search
    """

    # TODO This version of the agent can not handle deep dead ends but is very similar to HH version

    def __init__(self, G, color_map, size_map):
        """
        Initializes the agent
        :param G: The Graph from the main file
        :param color_map: Color map to coherently color the nodes
        :param size_map: Size map to coherently change the size of nodes
        """
        self.G = G
        self.color_map = color_map
        self.size_map = size_map

    def find_path(self, start, destination):
        """
        Kick off method for the depth first search
        :param start: The start node
        :param destination: The end node
        :return: The route that was found. Returns empty list of no route was found.
        """
        route = self.bfs(start, destination)
        if route:
            return route
        else:
            return []

    def sort_neighbours(self, current, neighbours, destination):
        """
        Sorts the neighbours based on their angle to the destination
        :param current: The node the agent is currently at
        :param neighbours: All the nodes neighbours
        :param destination: The destination node
        :return: Sorted list of neighbours
        """

        # Get the GPS point of the destination
        dest_lat = self.G.nodes[destination]['y']
        dest_lon = self.G.nodes[destination]['x']
        destination_point = (dest_lat, dest_lon)

        # Get the GPS point of the current location
        curr_lat = self.G.nodes[current]['y']
        curr_lon = self.G.nodes[current]['x']
        current_point = (curr_lat, curr_lon)

        # Bearing from our current position to the destination
        bearing_curr_dest = ox.bearing.calculate_bearing(curr_lat, curr_lon, dest_lat, dest_lon)

        # Bubble Sort to sort the nodes. Lists are usually < 5 elements. No need for efficiency
        performed_swap = True
        while performed_swap:
            performed_swap = False
            for i in range(len(neighbours) - 1):

                # Try to get first street segment for more accurate angle.
                # Throws exception when there are no segments
                try:
                    # Best_angle for comparison
                    best_angle = float("inf")
                    # Check if there is more than one edge
                    edges = self.G[current][neighbours[i]]
                    out_edges = list(self.G.out_edges(current, data=True))
                    if len(edges) > 1:
                        for j in range(len(edges)):
                            try:
                                edge_geometry = edges[j]['geometry']
                                line_string = str(edge_geometry).replace(",", "").replace("(", "").replace(")",
                                                                                                           "").replace(
                                    "LINESTRING", "")
                                coord_list = line_string.lstrip().split(" ")
                                edge_lat = float(coord_list[3])
                                edge_lon = float(coord_list[2])
                            except:
                                edge_lat = self.G.nodes[neighbours[i]]['y']
                                edge_lon = self.G.nodes[neighbours[i]]['x']

                            # Get bearing to the next
                            neighbour_point = (edge_lat, edge_lon)
                            edge_bearing = ox.bearing.calculate_bearing(curr_lat, curr_lon, edge_lat, edge_lon)

                            # Calculate the angle to the neighbour
                            angle = abs((bearing_curr_dest - edge_bearing + 180) % 360 - 180)
                            if angle < best_angle:
                                best_angle = angle
                                lat = edge_lat
                                lon = edge_lon
                    else:
                        edge = self.G[current][neighbours[i]][0]
                        edge_geometry = edge['geometry']
                        line_string = str(edge_geometry).replace(",", "").replace("(", "").replace(")", "").replace(
                            "LINESTRING", "")
                        coord_list = line_string.lstrip().split(" ")
                        lat = float(coord_list[3])
                        lon = float(coord_list[2])

                        # Display the first street segment that is used for angle calculation
                        # temp_node = ox.get_nearest_node(self.G, (lat, lon))
                        # self.color_map[list(self.G.nodes).index(temp_node)] = 'green'
                        # self.size_map[list(self.G.nodes).index(temp_node)] = 10
                        # ox.plot_graph(self.G, node_color=self.color_map, node_size=self.size_map)
                        #
                        # fig, ax = ox.plot_graph(self.G, node_color=self.color_map, node_size=self.size_map, show=False, close=False)
                        # ax.scatter(lon, lat, c='red', s=30)
                        # plt.show()

                except:
                    lat = self.G.nodes[neighbours[i]]['y']
                    lon = self.G.nodes[neighbours[i]]['x']

                # Get bearing to the first neighbour
                neighbour_point = (lat, lon)
                bearing1 = ox.bearing.calculate_bearing(curr_lat, curr_lon, lat, lon)

                # Print the current node
                # self.color_map[list(self.G.nodes).index(neighbours[i])] = 'blue'
                # self.size_map[list(self.G.nodes).index(neighbours[i])] = 10
                # ox.plot_graph(self.G, node_color=self.color_map, node_size=self.size_map)

                # Next neighbour (i+1)
                # Try to get first street segment for more accurate angle.
                # Throws exception when there are no segments
                try:
                    # Best_angle for comparison
                    best_angle = float("inf")
                    # Check if there is more than one edge
                    edges = self.G[current][neighbours[i + 1]]
                    out_edges = list(self.G.out_edges(current, data=True))
                    if len(edges) > 1:
                        for k in range(len(edges)):
                            try:
                                edge_geometry = edges[k]['geometry']
                                line_string = str(edge_geometry).replace(",", "").replace("(", "").replace(")",
                                                                                                           "").replace(
                                    "LINESTRING", "")
                                coord_list = line_string.lstrip().split(" ")
                                edge_lat = float(coord_list[3])
                                edge_lon = float(coord_list[2])
                            except:
                                edge_lat = self.G.nodes[neighbours[i + 1]]['y']
                                edge_lon = self.G.nodes[neighbours[i + 1]]['x']

                            # Get bearing to the next
                            neighbour_point = (edge_lat, edge_lon)
                            edge_bearing = ox.bearing.calculate_bearing(curr_lat, curr_lon, edge_lat, edge_lon)

                            # Calculate the angle to the neighbour
                            angle = abs((bearing_curr_dest - edge_bearing + 180) % 360 - 180)
                            if angle < best_angle:
                                best_angle = angle
                                lat = edge_lat
                                lon = edge_lon
                    else:
                        edge = self.G[current][neighbours[i + 1]][0]
                        edge_geometry = edge['geometry']
                        line_string = str(edge_geometry).replace(",", "").replace("(", "").replace(")", "").replace(
                            "LINESTRING", "")
                        coord_list = line_string.lstrip().split(" ")
                        lat = float(coord_list[3])
                        lon = float(coord_list[2])

                        # Display the first street segment that is used for angle calculation
                        # temp_node = ox.get_nearest_node(self.G, (lat, lon))
                        # self.color_map[list(self.G.nodes).index(temp_node)] = 'green'
                        # self.size_map[list(self.G.nodes).index(temp_node)] = 10
                        # ox.plot_graph(self.G, node_color=self.color_map, node_size=self.size_map)
                        #
                        # fig, ax = ox.plot_graph(self.G, node_color=self.color_map, node_size=self.size_map, show=False, close=False)
                        # ax.scatter(lon, lat, c='red', s=30)
                        # plt.show()

                except:
                    lat = self.G.nodes[neighbours[i + 1]]['y']
                    lon = self.G.nodes[neighbours[i + 1]]['x']

                # Get bearing to the first neighbour
                neighbour_point = (lat, lon)
                bearing2 = ox.bearing.calculate_bearing(curr_lat, curr_lon, lat, lon)

                # Draw the current neighbour node
                # self.color_map[list(self.G.nodes).index(neighbours[i + 1])] = 'blue'
                # self.size_map[list(self.G.nodes).index(neighbours[i + 1])] = 10
                # ox.plot_graph(self.G, node_color=self.color_map, node_size=self.size_map)

                # Calculate the angles to the two neighbours
                angle1 = abs((bearing_curr_dest - bearing1 + 180) % 360 - 180)
                angle2 = abs((bearing_curr_dest - bearing2 + 180) % 360 - 180)

                # Bubble Sort switch them
                if angle1 > angle2:
                    neighbours[i], neighbours[i + 1] = neighbours[i + 1], neighbours[i]
                    performed_swap = True

        return neighbours

    def is_fully_visited(self, node, visited_dict):
        """
        Helper function to check if a node has been fully visited
        :param node: Current node, agents position
        :param visited_dict: Dictionary of all nodes and their visited status
        :return: True if all neighbours of node have been visited before
        """
        neighbours = list(self.G.neighbors(node))
        fully_visited = True
        for neighbour in neighbours:
            if neighbour not in visited_dict:
                fully_visited = False
        return fully_visited

    def visit(self, node, visited_dict):
        if node not in visited_dict:
            visited_dict[node] = True

    def get_previous(self, node, prev_dict):
        if node in prev_dict:
            return prev_dict[node]
        else:
            return None

    def add_prev(self, node_to_append, node_to_append_to, prev_dict):
        if self.get_previous(node_to_append_to, prev_dict):
            prev_dict[node_to_append_to].append(node_to_append)
        else:
            prev_dict[node_to_append_to] = []
            prev_dict[node_to_append_to].append(node_to_append)

    def bfs(self, at, destination):
        """
        Best First Search to find a valid path to the goal.
        :param at: The current node (starting node of the path)
        :param destination: The destination, where the agents wants to go
        :return: The route, if path was found, emtpy list if path not available
        """
        # Dictionary that stores a list of previous nodes for every node
        prev = {}

        # Dictionary that stores the visited state of each node
        visited = {}

        # List that holds all node that have been visited on the way to the destination
        route = []

        # Create a stack for DFS and push the current source node
        stack = [at]

        # Frustration level, Sets the number that the agents is able to do a complete traceback
        frustration_level = 0

        while len(stack):
            # Pop a node from stack
            at = stack[-1]
            stack.pop()

            # Add the node to the route and mark as visited
            route.append(at)
            self.visit(at, visited)

            # Plot the current node
            # self.color_map[list(self.G.nodes).index(at)] = 'red'
            # self.size_map[list(self.G.nodes).index(at)] = 20
            # ox.plot_graph(self.G, node_color=self.color_map, node_size=self.size_map)

            # If we have reached the destination we can return the route
            if at == destination:
                return route, frustration_level

            # Get all neighbours
            neighbours = list(self.G.neighbors(at))

            # Get previous node if there is one
            if self.get_previous(at, prev):
                # The last visited node is the last one in the list of previous nodes [-1]
                previous_node = self.get_previous(at, prev)[-1]

                # Check if we are stuck in a loop
                # If the list of previous nodes from this node contains previous_node more than once
                # And is fully visited
                if self.get_previous(at, prev).count(previous_node) > 1 and self.is_fully_visited(at, visited):
                    # print("Stuck in Loop")
                    return [], frustration_level
                # If not stuck
                # Filter out previous node from neighbours
                # Node has previous (should be the case for all but start)
                elif previous_node:
                    # Plot the previous node grey as we have traversed over it
                    # self.color_map[list(self.G.nodes).index(previous_node)] = 'grey'
                    # self.size_map[list(self.G.nodes).index(previous_node)] = 15
                    # ox.plot_graph(self.G, node_color=self.color_map, node_size=self.size_map)

                    # Previous is not in neighbours if there is a one way street
                    # If there is a previous in neighbours, remove it
                    if previous_node in neighbours:
                        neighbours.remove(previous_node)

            # There is no previous which means this is the start node
            else:
                # If we are at the start node and we have only ourself as neighbour, stop
                if len(neighbours) == 1 and at in neighbours:
                    return [], frustration_level

            # If neighbours is now empty we have reached a dead end -> go back to previous
            # If agent starts in a cut off one way or disconnected dead end there is no neighbour
            # In that case just cancel the run
            if not neighbours:
                # self.color_map[list(self.G.nodes).index(at)] = 'black'
                # self.size_map[list(self.G.nodes).index(at)] = 15
                # ox.plot_graph(self.G, node_color=self.color_map, node_size=self.size_map)

                if self.get_previous(at, prev):
                    stack.append(previous_node)
                    self.add_prev(at, previous_node, prev)
                else:
                    return [], frustration_level
            else:
                # sort neighbours
                neighbours = self.sort_neighbours(at, neighbours, destination)

                # Choose the best unvisited neighbour
                for neighbour in neighbours:
                    if neighbour not in visited:
                        # Take the next neighbour and put it on the stack
                        self.add_prev(at, neighbour, prev)
                        stack.append(neighbour)
                        break
                # If all neighbours have been visited before choose the next best again.
                # This will probably result in stuck in loop in the next step
                if self.is_fully_visited(at, visited):
                    if frustration_level > 0:
                        # Solution with going back until node with unvisited neighbours
                        frustration_level -= 1
                        next_node = self.get_previous(at, prev)[0]
                        # print("all visited")
                        self.add_prev(at, next_node, prev)
                        stack.append(next_node)
                    else:
                        # Solution with canceling
                        # print("all visited")
                        self.add_prev(at, neighbours[0], prev)
                        stack.append(neighbours[0])

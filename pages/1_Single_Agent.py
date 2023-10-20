# Importing necessary libraries
import streamlit as st
import pandas as pd
import numpy as np
import osmnx as ox
import matplotlib.pyplot as plt
from geopy.geocoders import Nominatim
from la_agent import LeastAngleAgent
from PIL import Image
import networkx as nx

def interpolate_offset(dist, dist_min=500, dist_max=2000, offset_min=0.0005, offset_max=0.002):
    """
    Linearly interpolate the offset value based on the given distance.
    
    Parameters:
    - dist (int): The current distance value.
    - dist_min (int): The minimum value of the distance range.
    - dist_max (int): The maximum value of the distance range.
    - offset_min (float): The desired offset when dist is at its minimum.
    - offset_max (float): The desired offset when dist is at its maximum.
    
    Returns:
    - float: The interpolated offset value.
    """
    return offset_min + (dist - dist_min) * (offset_max - offset_min) / (dist_max - dist_min)

def get_graph_stats(G):
    """
    Fetches basic statistics of the graph and stores them in the session state.

    Parameters:
    - G (networkx graph): The graph for which the statistics are to be fetched.
    """
    st.session_state['stats'] = ox.stats.basic_stats(G)

def print_graph_stats():
    """
    Displays the statistics of the city graph using Streamlit's UI components.
    """
    # General Information
    st.subheader("General Information")
    st.write(f"Number of Nodes (Intersections): {st.session_state['stats']['n']}")
    st.write(f"Number of Edges (Streets): {st.session_state['stats']['m']}")
    st.write(f"Average Node Degree: {round(st.session_state['stats']['k_avg'], 2)}")

    # Street Information
    st.subheader("Street Information")
    st.write(f"Total Street Length: {round(st.session_state['stats']['street_length_total'], 2)} meters")
    st.write(f"Average Street Length: {round(st.session_state['stats']['street_length_avg'], 2)} meters")
    st.write(f"Number of Street Segments: {st.session_state['stats']['street_segment_count']}")
    st.write(f"Average Circuity: {round(st.session_state['stats']['circuity_avg'], 2)}")

    # Visualizing Streets per Node Proportions
    st.subheader("Streets per Node Proportions")
    chart_data = pd.DataFrame.from_dict(st.session_state['stats']['streets_per_node_proportions'], orient='index', columns=['Proportion'])
    st.bar_chart(chart_data)

    # Edge Information
    st.subheader("Edge Information")
    st.write(f"Total Edge Length: {round(st.session_state['stats']['edge_length_total'], 2)} meters")
    st.write(f"Average Edge Length: {round(st.session_state['stats']['edge_length_avg'], 2)} meters")

    # Node Information
    st.subheader("Node Information")
    st.write(f"Average Streets per Node: {round(st.session_state['stats']['streets_per_node_avg'], 2)}")
    st.write(f"Number of Intersections: {st.session_state['stats']['intersection_count']}")

    # Visualizing Streets per Node Counts
    st.subheader("Streets per Node Counts")
    chart_data_counts = pd.DataFrame.from_dict(st.session_state['stats']['streets_per_node_counts'], orient='index', columns=['Counts'])
    st.bar_chart(chart_data_counts)

    # Miscellaneous Information
    st.subheader("Miscellaneous Information")
    st.write(f"Proportion of Self Loops: {round(st.session_state['stats']['self_loop_proportion'], 4)}")

def get_graph(city, dist):
    """
    Fetches the graph representation of the city using OSMnx.

    Parameters:
    - city (str): Name of the city.
    - dist (int): Distance in meters from the city center.
    
    Returns:
    - G (networkx graph): Graph representation of the city.
    - ax (matplotlib axis): The axis on which the graph is plotted.
    """
    geolocator = Nominatim(user_agent="geoapi")
    location = geolocator.geocode(city)
    
    if not location:
        st.error("Couldn't find city. Try a different search term.")
        return None, None

    point = (location.latitude, location.longitude)
    G = ox.graph_from_point(point, dist=dist, dist_type='bbox', clean_periphery=True, simplify=True, network_type="bike")
    
    if G:
        st.write(f"Displaying graph for {city} around center point {point}")
        fig, ax = ox.plot_graph(G, 
                                show=False, 
                                close=False, 
                                bgcolor='white',
                                node_color='grey',
                                node_size=10,
                                edge_color='black',
                                edge_linewidth=0.5)
        with plot_spot:
            st.pyplot(fig)

        color_map = ['white' for _ in G.nodes]
        size_map = [5 for _ in G.nodes]
        st.session_state.la_agent = LeastAngleAgent(G, color_map, size_map)
        get_graph_stats(G)
        return G, ax

import matplotlib.pyplot as plt

def create_random_route(plot_spot, G):
    """
    Creates a random route in the city graph and visualizes it.

    Parameters:
    - plot_spot (streamlit delta generator): The place in the Streamlit UI where the plot should appear.
    - G (networkx graph): Graph representation of the city.
    """
    fig, ax = ox.plot_graph(G, 
                            show=False, 
                            close=False, 
                            bgcolor='white',
                            node_color='grey',
                            node_size=10,
                            edge_color='black',
                            edge_linewidth=0.5)

    start_node, target_node = np.random.choice(G.nodes, 2)
    route1, _ = st.session_state.la_agent.find_path(start_node, target_node)
    
    # Get the x and y coordinates of the start and end nodes
    x_start, y_start = G.nodes[start_node]['x'], G.nodes[start_node]['y']
    x_end, y_end = G.nodes[target_node]['x'], G.nodes[target_node]['y']

    # Load PNG icons for start and end nodes
    end_icon = plt.imread('marker.png')

    # In your create_random_route function, calculate the offset using:
    offset = interpolate_offset(st.session_state['dist'])

    # Overlay the icons on the plot with a fixed size
    ax.imshow(end_icon, extent=(x_end-offset, x_end+offset, y_end-offset, y_end+offset), zorder=3)

    
    if nx.has_path(G, start_node, target_node):
        route2 = nx.shortest_path(G, source=start_node, target=target_node, weight='length')
    else:
        route2 = []

    if route1 and route2:
        try:
            ox.plot_graph_route(G, route1, ax=ax, show=False, close=False, route_color='red', route_linewidth=2)
            ox.plot_graph_route(G, route2, ax=ax, show=False, close=False, route_color='blue', route_linewidth=2)
            with plot_spot:
                st.pyplot(fig)
        except:
            create_random_route(plot_spot, G)
    else:
        create_random_route(plot_spot, G)


    

# Streamlit UI Initialization
st.set_page_config(page_title="Single Agent", page_icon="🧭")
if 'G' not in st.session_state:
    st.session_state['G'] = None
if 'ax' not in st.session_state:
    st.session_state['ax'] = None
if 'route_visible' not in st.session_state:
    st.session_state.route_visible = False
if 'la_agent' not in st.session_state:
    st.session_state.la_agent = None
if 'stats' not in st.session_state:
    st.session_state['stats'] = None
if 'stats' not in st.session_state:
    st.session_state['dist'] = None

image = Image.open('header_img.png')
st.title("Visualising As-The-Crow-Flies Navigation")
st.image(image, caption='Created with DALLE 3')

with st.form("my_form"):
    city = st.text_input("Enter a city:", value='Zurich')
    st.session_state['dist'] = st.slider('Graph size (meters from city center):', min_value=500, max_value=2000, value=1000, step=100)
    submitted = st.form_submit_button("Show City")

button1 = st.empty()
plot_spot = st.empty()

if submitted:
    with st.spinner('Loading Street Network'):
        st.session_state['G'], st.session_state['ax'] = get_graph(city, st.session_state['dist'])
        st.session_state.route_visible = True
        if st.session_state['stats']:
            print_graph_stats()

if st.session_state.route_visible:
    if button1.button('Create Random Route'):
        with st.spinner('Calculating Route'):
            create_random_route(plot_spot, st.session_state.G)
            st.write("Red: As-The-Crow-Flies, Blue: Shortest Path")
            if st.session_state['stats']:
                print_graph_stats()

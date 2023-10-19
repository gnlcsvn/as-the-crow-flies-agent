import streamlit as st
import pandas as pd
import numpy as np
import osmnx as ox
import matplotlib.pyplot as plt
from geopy.geocoders import Nominatim
from la_agent import LeastAngleAgent
from PIL import Image
import networkx as nx

st.set_page_config(
    page_title="Single Agent",
    page_icon="🧭",
)


def get_graph_stats(G):
    st.session_state['stats'] = ox.stats.basic_stats(G)

def print_graph_stats():
    st.title("City Graph Statistics")

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
    # Get latitude and longitude of the city
    geolocator = Nominatim(user_agent="geoapi")
    location = geolocator.geocode(city)
    
    if location is None:
        st.error("Couldn't find city. Try a different search term.")
        return None, None

    # Retrieve the graph from OSMnx
    point = (location.latitude, location.longitude)
    G = ox.graph_from_point(point, dist=dist, dist_type='bbox', clean_periphery=True, simplify=True, network_type="bike")
    if G is not None and point is not None:
        st.write(f"Displaying graph for {city} around center point {point}")
        fig, ax = ox.plot_graph(G, 
                                show=False, 
                                close=False, 
                                bgcolor='white',
                                node_color='grey',
                                node_size=10,
                                edge_color='black',
                                edge_linewidth=0.5)
        
        #send the plot to it's spot "in the line" 
        with plot_spot:
            st.pyplot(fig)

        color_map = ['white' for i in range(G.number_of_nodes())]
        size_map = [5 for i in range(G.number_of_nodes())]

        # Create an instance of WayfindingAgent
        st.session_state.la_agent = LeastAngleAgent(G, color_map, size_map)
        get_graph_stats(G)
        return(G, ax)

def create_random_route(plot_spot, G):
    # Reproduce the base graph
    fig, ax = ox.plot_graph(G, 
                                show=False, 
                                close=False, 
                                bgcolor='white',
                                node_color='grey',
                                node_size=10,
                                edge_color='black',
                                edge_linewidth=0.5)

    start_node = np.random.choice(G.nodes)
    target_node = np.random.choice(G.nodes)
            
    # After getting the route
    route1 = st.session_state.la_agent.find_path(start_node, target_node)
    route1 = route1[0]
            
    # Calculate the shortest path using Dijkstra's algorithm
    try:
        route2 = nx.shortest_path(G, source=start_node, target=target_node, weight='length')
    except nx.NetworkXNoPath:
        route2 = []

    if route1 and route2:  # ensure both routes are not None or empty
    # Draw the least-angle route
        fig, ax = ox.plot_graph_route(G, route1,
                                      ax=ax, 
                                      show=False, 
                                      close=False,
                                      bgcolor='white',
                                      node_color='grey',
                                      node_size=10,
                                      edge_color='black',
                                      edge_linewidth=0.5,
                                      route_color='red',
                                      route_linewidth=2)

        # Draw the shortest path on the same plot
        ox.plot_graph_route(G, route2,
                            ax=ax, 
                            show=False, 
                            close=False,
                            route_color='blue',  # Different color for clarity
                            route_linewidth=2)
        with plot_spot:
            st.pyplot(fig)
    else:
        # Repeat until route is found
        create_random_route(plot_spot, G)
    
    st.write("Red: ATCF, Blue: Shortest Path")

# Streamlit UI

# Initialize session_state variables if they don't exist
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


image = Image.open('header_img.png')
st.image(image, caption='')

st.title("Visualising As-The-Crow-Flies Navigation")

with st.form("my_form"):
    city = st.text_input("Please enter a city name:")
    dist = st.slider('Choose the size of the graph', 0, 2000, 500)

    
    # Every form must have a submit button.
    submitted = st.form_submit_button("Show City")


button1 = st.empty()
plot_spot = st.empty()

if submitted:
    with st.spinner('Loading Street Network'):
        st.session_state['G'], st.session_state['ax'] = get_graph(city, dist)
        st.session_state.route_visible = True
        if st.session_state['stats']:
            print_graph_stats()

if st.session_state.route_visible:
    if button1.button('Create Random Route'):
        with st.spinner('Calculating Route'):
            create_random_route(plot_spot, st.session_state.G)
            if st.session_state['stats']:
                print_graph_stats()


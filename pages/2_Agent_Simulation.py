# Importing necessary libraries
import streamlit as st
from la_agent import LeastAngleAgent
from util import get_straight_line_distance, get_length_of_path
import networkx as nx
import osmnx as ox
import numpy as np
import pandas as pd
from PIL import Image
from geopy.geocoders import Nominatim
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


def get_graph(city, dist):
    """
    Retrieve a graph representation of the city using OSMnx.
    
    Parameters:
    - city (str): Name of the city.
    - dist (int): Distance in meters from the city center.
    
    Returns:
    - G (networkx graph): Graph representation of the city.
    """
    geolocator = Nominatim(user_agent="geoapi")
    location = geolocator.geocode(city)

    # Check if the city was found
    if not location:
        st.error("Couldn't find city. Try a different search term.")
        return

    # Get graph from the city's point
    point = (location.latitude, location.longitude)
    G = ox.graph_from_point(point, dist=dist, dist_type='bbox', clean_periphery=True, simplify=True, network_type="bike")
    
    if G:
        st.write(f"Graph for {city} centered at {point}")
        fig, ax = ox.plot_graph(G, show=False, close=False, bgcolor='white',
                                node_color='grey', node_size=10, edge_color='black', edge_linewidth=0.5)
        with plot_spot:
            st.pyplot(fig)

        color_map = ['white' for _ in G.nodes]
        size_map = [5 for _ in G.nodes]
        st.session_state.la_agent = LeastAngleAgent(G, color_map, size_map)

    return G

def la_simulation(G, iterations):
    """
    Simulate paths using the Least Angle Agent and compare it to the shortest path.
    
    Parameters:
    - G (networkx graph): Graph representation of the city.
    - iterations (int): Number of simulations.
    """
    iterations_counter = 0

    # Lists to store results
    la_success = []
    shortest_path_success = []
    la_length = [np.nan] * iterations
    shortest_path_length = [np.nan] * iterations
    dist_start_target_list = [np.nan] * iterations
    every_agent_found_route = 0

    progress_bar = st.progress(0)

    # Main simulation loop
    while iterations_counter < iterations:
        # Randomly select start and end nodes
        start_node, target_node = np.random.choice(G.nodes, 2)
        dist_start_target = get_straight_line_distance(G, start_node, target_node)

        # Find path using the Least Angle Agent
        route1, _ = st.session_state.la_agent.find_path(start_node, target_node)
        
        # Find the shortest path
        route4 = nx.shortest_path(G, start_node, target_node, weight='length', method='dijkstra') if nx.has_path(G, start_node, target_node) else []

        # Store results
        if route1 and route4:
            every_agent_found_route += 1
            la_length[iterations_counter] = get_length_of_path(G, route1, "angle")
            shortest_path_length[iterations_counter] = nx.shortest_path_length(G, start_node, target_node, weight="length")
            dist_start_target_list[iterations_counter] = dist_start_target

        # Update success lists based on whether paths were found
        la_success.append(1 if route1 else 0)
        shortest_path_success.append(1 if route4 else 0)
        iterations_counter += 1
        progress_bar.progress(iterations_counter / iterations)

    progress_bar.empty()

    # Display results
    st.write("### Simulation Results")
    la_success_rate = sum(la_success) / iterations
    la_pf = np.nanmean(la_length) / np.nanmean(shortest_path_length)
    st.write(f"Least Angle Agent Success Rate: {la_success_rate:.2f}")
    st.write(f"LA PF: {la_pf:.2f}")

    # Display average path lengths for both agents
    st.write("### Average Path Lengths")
    avg_lengths = {
        'Agent': ['LA Agent', 'Shortest Path'],
        'Average Length': [np.nanmean(la_length), np.nanmean(shortest_path_length)]
    }
    st.bar_chart(pd.DataFrame(avg_lengths).set_index('Agent'))

    # Scatter plot of path lengths against straight-line distance between start and target
    st.write("### Path Length vs Straight-Line Distance")
    scatter_data = pd.DataFrame({
        'Straight-Line Distance': dist_start_target_list,
        'LA Path Length': la_length,
        'Shortest Path Length': shortest_path_length
    }).dropna()

    x = scatter_data['Straight-Line Distance'].values.reshape(-1, 1)
    fig, ax = plt.subplots()

    # Plot for both agents
    for agent, color in [('LA Path Length', 'blue'), ('Shortest Path Length', 'green')]:
        y = scatter_data[agent].values
        reg = LinearRegression().fit(x, y)
        ax.scatter(x, y, color=color, alpha=0.5, label=agent)
        ax.plot(x, reg.predict(x), color=color, linestyle='--')

    ax.set_xlabel('Straight-Line Distance')
    ax.set_ylabel('Path Length')
    ax.legend()
    st.pyplot(fig)

    st.write(f"Routes found for all agents: {every_agent_found_route}/{iterations}")

# Streamlit UI Setup
st.set_page_config(page_title="Agent Simulation", page_icon="🧭")
if 'G' not in st.session_state:
    st.session_state['G'] = None

image = Image.open('header_img.png')
st.title("As-The-Crow-Flies Agent Simulation in Urban Environments")
st.image(image, caption='Created with DALLE 3')
st.write("""
In this simulation, we demonstrate the capability of a Least Angle Agent to navigate an urban environment. The agent tries to find a path from a random start point to a random end point in the city graph. We compare the path taken by the agent with the shortest path. The success rate of the agent in finding a path and the path factor (PF), which is the ratio of the path length taken by the agent to the shortest path length, are displayed.
""")

with st.form("input_form"):
    city = st.text_input("Enter a city:", value='Zurich')
    dist = st.slider('Graph size (meters from city center):', min_value=500, max_value=2000, value=1000, step=100)
    iterations = st.number_input("Iterations:", min_value=10, max_value=1000, value=100, step=10)
    submit = st.form_submit_button("Run Simulation")

plot_spot = st.empty()
if submit:
    with st.spinner("Running simulation..."):
        G = get_graph(city, dist)
        if G:
            la_simulation(G, iterations)

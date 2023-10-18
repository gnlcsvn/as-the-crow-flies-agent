import streamlit as st
import pandas as pd
import numpy as np
import osmnx as ox
import matplotlib.pyplot as plt
from geopy.geocoders import Nominatim
from la_agent import LeastAngleAgent
from PIL import Image


def get_node_color_map(G, route):
	"""
	Get a color map for nodes in graph G, highlighting the nodes in 'route'.
	"""
	color_map = []
	for node in G.nodes:
		if node in route:
			color_map.append('red')  # color for route nodes
		else:
			color_map.append('grey')  # color for all other nodes
			return color_map

def get_graph(city, dist):
	# Get latitude and longitude of the city
	geolocator = Nominatim(user_agent="geoapi")
	location = geolocator.geocode(city)
	
	if location is None:
		st.error("Couldn't find city. Try a different search term.")
		return None, None

	# Retrieve the graph from OSMnx
	point = (location.latitude, location.longitude)
	G = ox.graph_from_point(point, dist=dist, dist_type='bbox', network_type='all')
	if G is not None and point is not None:
		st.write(f"Displaying graph for {city} around point {point}")
		fig, ax = ox.plot_graph(G, show=False, close=False)
		#send the plotly chart to it's spot "in the line" 
		with plot_spot:
			st.pyplot(fig)

		color_map = ['white' for i in range(G.number_of_nodes())]
		size_map = [5 for i in range(G.number_of_nodes())]

		# Create an instance of WayfindingAgent
		st.session_state.la_agent = LeastAngleAgent(G, color_map, size_map)
		return(G, ax)

def create_random_route(plot_spot, G):
    # Reproduce the base graph
    fig, ax = ox.plot_graph(G, show=False, close=False)

    start_node = np.random.choice(G.nodes)
    target_node = np.random.choice(G.nodes)
            
    # After getting the route
    route1 = st.session_state.la_agent.find_path(start_node, target_node)
    route1 = route1[0]
            
    if route1:  # ensure route1 is not None or empty
        fig, ax = ox.plot_graph_route(G, route1, ax=ax, edge_color='red', edge_linewidth=3, show=False, close=False)
        with plot_spot:
            st.pyplot(fig)




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


image = Image.open('sunrise.png')
st.image(image, caption='')

st.title("Visualising As-The-Crow-Flies Navigation")
city = st.text_input("Please enter a city name:")
dist = st.slider('Choose the size of the graph', 0, 2000, 500)

col1, col2 = st.columns(2)

with col1:
	button1 = st.empty()

with col2:
	button2 = st.empty()

plot_spot = st.empty()

if button1.button('Show City'):
	with st.spinner('Loading Street Network'):
	    st.session_state['G'], st.session_state['ax'] = get_graph(city, dist)
	    st.session_state.route_visible = True

if st.session_state.route_visible:
    if button2.button('Create Random Route'):
    	with st.spinner('Calculating Route'):
    		create_random_route(plot_spot, st.session_state.G)






	

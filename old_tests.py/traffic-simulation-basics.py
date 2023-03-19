import networkx as nx
import numpy as np

# Define the environment: a graph representing a city grid
G = nx.Graph()

# Add nodes to the graph, one for each cell in the grid
for i in range(num_rows):
    for j in range(num_cols):
        G.add_node((i, j), pos=(i, j))

# Add edges to the graph to represent the connections between cells
for i in range(num_rows):
    for j in range(num_cols):
        if i > 0: # only for cases on the edge
            G.add_edge((i, j), (i - 1, j), weight="street", speed=50)
            G.add_edge((i, j), (i - 1, j), weight="sidewalk", speed=10)
        if i < num_rows - 1: # only for cases on the edge
            G.add_edge((i, j), (i + 1, j), weight="street", speed=50)
            G.add_edge((i, j), (i + 1, j), weight="sidewalk", speed=10)
        if j > 0: # only for cases on the edge
            G.add_edge((i, j), (i, j - 1), weight="street", speed=50)
            G.add_edge((i, j), (i, j - 1), weight="sidewalk", speed=10)
        if j < num_cols - 1: # only for cases on the edge
            G.add_edge((i, j), (i, j + 1), weight="street", speed=50)
            G.add_edge((i, j), (i, j + 1), weight="sidewalk", speed=10)

# Define the reward signal: the time it takes the agent to reach the destination
def compute_reward(current_state, destination_state):
    # Compute the shortest path between the current state and the destination state
    path = nx.shortest_path(G, source=current_state, target=destination_state, weight="speed")
    # Compute the total time it takes to traverse the path
    time = sum([G[path[i]][path[i+1]]["speed"] for i in range(len(path) - 1)])
    return time

# Define the agent: a simple Q-learning agent
class QLearningAgent:
    def __init__(self, num_actions):
        self.num_actions = num_actions
        self.q_values = np.zeros((num_rows, num_cols, num_actions))
    
    def get_action(self, state):
        # Select the action with the highest Q-value for the given state
        return np.argmax(self.q_values[state])
    
    def update_q_values(self, state, action, reward, next_state):
        # Update the Q-values using the Q-learning update rule
        alpha = 0.1  # learning rate
        gamma = 0.9  # discount factor
        self.q_values[state][action] = (1 - alpha) * self.q_values[state][action] + alpha * (reward + gamma * np.max(self.q_values[next_state]))

# Create the Q-learning agent
agent = QLearningAgent(num_actions=8)

# Set up the destination state
destination_state = (num_rows - 1, num_cols - 1)

# Train the agent
num_episodes = 1000
for episode in range(num_episodes):
    # Select a random starting state
    current_state = (np.random.randint(num_rows), np.random.randint(num_cols))
    while current_state != destination_state:
        # Select an action
        action = agent.get_action(current_state)
        
        # Take the action and observe the reward and next state
        next_state = list(G.neighbors(current_state))[action]
        reward = compute_reward(current_state, destination_state)
        
        # Update the Q-values
        agent.update_q_values(current_state, action, reward, next_state)
        
        # Set the current state to the next state
        current_state = next_state

# Test the agent
current_state = (0, 0)
while current_state != destination_state:
    # Select the action with the highest Q-value for the current state
    action = agent.get_action(current_state)
    
    # Take the action and update the current state
    next_state = list(G.neighbors(current_state))[action]
    current_state = next_state
    
    print(f"Moved from {current_state} to {next_state}")


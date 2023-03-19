def get_action(self, state, transport):
    # Select the action with the highest Q-value for the given state and transport
    if transport == "bike":
        # Only consider edges that are marked as "bike lane"
        valid_actions = [i for i, (n, _) in enumerate(G[state]) if "bike lane" in G[state][n]]
    elif transport == "car":
        # Only consider edges that are marked as "street"
        valid_actions = [i for i, (n, _) in enumerate(G[state]) if "street" in G[state][n]]
    else:
        # Consider all actions
        valid_actions = range(self.num_actions)
    q_values = self.q_values[state][valid_actions]
    return valid_actions[np.argmax(q_values)]


# Select the action with the highest Q-value for a bike
action = agent.get_action(current_state, transport="bike")

# Select the action with the highest Q-value for a car
action = agent.get_action(current_state, transport="car")

# Select the action with the highest Q-value for any mode of transport
action = agent.get_action(current_state, transport=None)

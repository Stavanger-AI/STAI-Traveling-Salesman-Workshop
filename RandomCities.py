import numpy as np
import tkinter as tk
from scipy.spatial.distance import cdist
from tqdm import tqdm
import threading
import time
import random


def generate_random_city_locations(num_cities):
    city_locations = [
        (random.randint(50, 450), random.randint(50, 450)) for _ in range(num_cities)
    ]
    return city_locations


# Example of generating 10 random city locations
# random_city_locations = generate_random_city_locations(10)

# You can change the argument (10 in this example) to generate a different number of random city locations.
# You can also change the range of the random numbers to generate cities in a different area.

fixed_city_locations = [
    (100, 100),
    (200, 150),
    (320, 200),
    (450, 350),
    (500, 500),
    (100, 300),
    (250, 400),
    (300, 100),
    (400, 250),
    (500, 350),
    (100, 500),
    (200, 100),
    (350, 400),
    (400, 300),
    (500, 100),
    (150, 250),
    (250, 150),
    (350, 450),
    (450, 350),
    (50, 450),
]


class Agent:
    def __init__(self):
        pass

    def expand_state_vector(self, state):
        if len(state.shape) == 1 or len(state.shape) == 3:
            return np.expand_dims(state, axis=0)
        else:
            return state

    def remember(self, *args):
        self.memory.save(args)


class QAgent(Agent):
    def __init__(
        self,
        states_size,
        actions_size,
        epsilon=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.999,
        gamma=0.95,
        lr=0.8,
    ):
        self.states_size = states_size
        self.actions_size = actions_size
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.gamma = gamma
        self.lr = lr
        self.Q = self.build_model(states_size, actions_size)

    def build_model(self, states_size, actions_size):
        return np.zeros([states_size, actions_size])

    def train(self, s, a, r, s_next):
        self.Q[s, a] += self.lr * (
            r + self.gamma * np.max(self.Q[s_next, :]) - self.Q[s, a]
        )
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def act(self, s):
        q = self.Q[s, :]
        if np.random.rand() > self.epsilon:
            return np.argmax(q)
        return np.random.randint(self.actions_size)


class DeliveryEnvironment(object):
    def __init__(self, stops, method="distance"):
        # Initialization
        self.n_stops = len(stops)
        self.action_space = self.n_stops
        self.observation_space = self.n_stops
        self.stops = stops
        self.method = method

        # Generate stops
        self._generate_constraints()
        self._generate_q_values()

    def _generate_constraints(self):
        pass  # For simplicity, I'm leaving this empty. You can add your logic.

    def _generate_q_values(self):
        pass  # For simplicity, I'm leaving this empty. You can add your logic.

    def reset(self):
        return 0  # This is a basic reset, you should adjust according to your needs

    def step(self, action):
        # Calculate the distance to the next stop (for simplicity, we'll use Euclidean distance)
        current_stop = self.stops[self.current_state]
        next_stop = self.stops[action]

        distance = np.sqrt(
            (next_stop[0] - current_stop[0]) ** 2
            + (next_stop[1] - current_stop[1]) ** 2
        )

        # Penalize revisiting cities
        if action in self.visited_states:
            distance *= 2.0

        reward = distance

        self.current_state = action
        self.visited_states.append(action)

        # Check if all stops have been visited
        done = len(self.visited_states) == self.n_stops

        return self.current_state, reward, done

    def reset(self):
        self.current_state = 0
        self.visited_states = [0]
        return self.current_state

    def draw_stops(self, canvas):
        for stop in self.stops:
            canvas.create_oval(
                stop[0] - 5, stop[1] - 5, stop[0] + 5, stop[1] + 5, fill="red"
            )  # Represent cities as red dots


def run_episode(env, agent, canvas, episode_var, reward_var, verbose=1):
    agent.reset_memory()  # Reset the agent's memory at the start of each episode.
    s = env.reset()
    agent.remember_state(s)  # Remember the initial state.

    episode_reward = 0
    max_step = env.n_stops
    i = 0
    while i < max_step:
        a = agent.act(s)

        s_next, r, done = env.step(a)
        r = -1 * r
        if verbose:
            print(s_next, r, done)
        agent.train(s, a, r, s_next)

        agent.remember_state(s_next)  # Remember the next state.

        episode_reward += r
        s = s_next
        i += 1
        if i % 100 == 0:  # Draw lines every 10 iterations
            # Visualization
            canvas.delete("all")  # Clear previous drawing
            env.draw_stops(canvas)
            for idx in range(1, len(agent.states_memory)):
                start = env.stops[agent.states_memory[idx - 1]]
                end = env.stops[agent.states_memory[idx]]
                canvas.create_line(
                    start[0], start[1], end[0], end[1], fill="blue"
                )  # Draw routes between cities
            canvas.update()  # Update the canvas to show the new route
            reward_var.set(f"Reward: {episode_reward}")
            time.sleep(0.00001)  # Add delay for visualization
        if done:
            break
    return env, agent, episode_reward


class DeliveryQAgent(QAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reset_memory()

    def act(self, s):
        q = np.copy(self.Q[s, :])
        q[self.states_memory] = -np.inf

        # Check if there are unvisited actions left
        unvisited_actions = [
            x for x in range(self.actions_size) if x not in self.states_memory
        ]

        if unvisited_actions:
            # Choose from unvisited actions
            return np.random.choice(unvisited_actions)
        else:
            # Handle the case where all actions have been visited
            # Here, you can return a default action or handle it based on your problem's requirements
            return 0  # For example, return the first action

    def remember_state(self, s):
        self.states_memory.append(s)

    def reset_memory(self):
        self.states_memory = []


def run_n_episodes(env, agent, canvas, episode_var, reward_var, n_episodes=1000):
    rewards = []
    for i in tqdm(range(n_episodes)):
        episode_var.set(f"Episode: {i+1}/{n_episodes}")
        env, agent, episode_reward = run_episode(
            env, agent, canvas, episode_var, reward_var, verbose=0
        )
        rewards.append(episode_reward)
    return env, agent


def visualize_best_path(env, agent, canvas):
    current_state = 0
    path = [current_state]
    while len(path) < env.n_stops:
        q_values = agent.Q[current_state, :]
        # Set Q-values for visited cities to negative infinity
        q_values[path] = -np.inf
        next_city = np.argmax(q_values)
        path.append(next_city)
        current_state = next_city

    # Draw the path
    canvas.delete("all")
    env.draw_stops(canvas)
    for idx in range(
        1, len(path)
    ):  # Here we should start at 1 since path[0] is the initial state.
        start = env.stops[path[idx - 1]]
        end = env.stops[path[idx]]
        canvas.create_line(start[0], start[1], end[0], end[1], fill="blue")
    # Connect the last city to the first to complete the loop
    if len(path) > 1:
        start = env.stops[path[-1]]
        end = env.stops[path[0]]
        canvas.create_line(start[0], start[1], end[0], end[1], fill="blue")


def thread_complete_callback():
    """Callback function to be executed after thread completes"""
    visualize_best_path(env, agent, canvas)


def check_thread_status(thread, callback):
    """Checks if a thread is still alive and schedules a check again if it is, else executes callback"""
    if thread.is_alive():
        root.after(100, check_thread_status, thread, callback)
    else:
        callback()


class SettingsFrame(tk.Frame):
    def __init__(self, master=None, **kwargs):
        super().__init__(master, **kwargs)
        self.init_widgets()

    def init_widgets(self):
        # Number of stops (cities)
        tk.Label(self, text="Number of Cities:").grid(row=0, column=0, padx=10, pady=10)
        self.n_stops_var = tk.IntVar(value=10)
        tk.Entry(self, textvariable=self.n_stops_var).grid(
            row=0, column=1, padx=10, pady=10
        )

        # Explanation
        tk.Label(self, text="Determines the number of stops or cities to visit.").grid(
            row=0, column=2, padx=10
        )

        # Number of episodes
        tk.Label(self, text="Number of Episodes:").grid(
            row=1, column=0, padx=10, pady=10
        )
        self.n_episodes_var = tk.IntVar(value=1000)
        tk.Entry(self, textvariable=self.n_episodes_var).grid(
            row=1, column=1, padx=10, pady=10
        )

        # Explanation
        tk.Label(self, text="Determines the number of episodes to train.").grid(
            row=1, column=2, padx=10
        )

        # Learning Rate
        tk.Label(self, text="Learning Rate (α):").grid(
            row=2, column=0, padx=10, pady=10
        )
        self.lr_var = tk.DoubleVar(value=0.8)
        tk.Entry(self, textvariable=self.lr_var).grid(row=2, column=1, padx=10, pady=10)

        # Explanation
        tk.Label(
            self, text="Range [0,1]. Controls speed of learning. Lower is slower."
        ).grid(row=2, column=2, padx=10)

        # Initial Epsilon
        tk.Label(self, text="Initial Epsilon (ε):").grid(
            row=4, column=0, padx=10, pady=10
        )
        self.epsilon_var = tk.DoubleVar(value=1.0)
        tk.Entry(self, textvariable=self.epsilon_var).grid(
            row=4, column=1, padx=10, pady=10
        )

        # Explanation
        tk.Label(
            self, text="Range [0,1]. Higher means more exploration initially."
        ).grid(row=4, column=2, padx=10)

        # Epsilon Decay
        tk.Label(self, text="Epsilon Decay:").grid(row=5, column=0, padx=10, pady=10)
        self.epsilon_decay_var = tk.DoubleVar(value=0.999)
        tk.Entry(self, textvariable=self.epsilon_decay_var).grid(
            row=5, column=1, padx=10, pady=10
        )

        # Explanation
        tk.Label(self, text="Range [0.9,1]. Closer to 1 means slower decay.").grid(
            row=5, column=2, padx=10
        )

        # Epsilon Min
        tk.Label(self, text="Epsilon Min:").grid(row=6, column=0, padx=10, pady=10)
        self.epsilon_min_var = tk.DoubleVar(value=0.01)
        tk.Entry(self, textvariable=self.epsilon_min_var).grid(
            row=6, column=1, padx=10, pady=10
        )

        # Explanation
        tk.Label(self, text="Range [0,0.1]. Minimum exploration probability.").grid(
            row=6, column=2, padx=10
        )

    def get_parameters(self):
        return {
            "n_stops": self.n_stops_var.get(),
            "n_episodes": self.n_episodes_var.get(),
            "lr": self.lr_var.get(),
            "epsilon": self.epsilon_var.get(),
            "epsilon_decay": self.epsilon_decay_var.get(),
            "epsilon_min": self.epsilon_min_var.get(),
        }


# Button to start the training
def start_training():
    global env, agent

    # Get parameters from the settings frame
    params = settings_frame.get_parameters()

    # Get the first N city locations from the fixed_city_locations list
    n_cities = params["n_stops"]
    # cities = fixed_city_locations[:n_cities]
    cities = generate_random_city_locations(n_cities)

    env = DeliveryEnvironment(stops=cities)

    env.draw_stops(canvas)
    agent = DeliveryQAgent(
        states_size=env.observation_space,
        actions_size=env.action_space,
        epsilon=params["epsilon"],
        epsilon_min=params["epsilon_min"],
        epsilon_decay=params["epsilon_decay"],
        lr=params["lr"],
    )

    # Start the thread and pass the callback function
    thread = threading.Thread(
        target=run_n_episodes,
        args=(env, agent, canvas, episode_var, reward_var, params["n_episodes"]),
    )
    thread.start()

    # Using the main thread's after method to poll for thread's completion
    root.after(100, check_thread_status, thread, thread_complete_callback)


root = tk.Tk()
root.title("Traveling Salesman Problem Visualization")

canvas_frame = tk.Frame(root)
canvas_frame.pack(pady=20)
canvas = tk.Canvas(canvas_frame, bg="white", width=500, height=500)
canvas.pack()

info_frame = tk.Frame(root)
info_frame.pack(pady=10)
episode_var = tk.StringVar(value="Episode: 0/0")
episode_label = tk.Label(info_frame, textvariable=episode_var)
episode_label.pack(side=tk.LEFT, padx=10)
reward_var = tk.StringVar(value="Reward: 0")
reward_label = tk.Label(info_frame, textvariable=reward_var)
reward_label.pack(side=tk.LEFT, padx=10)

settings_frame = SettingsFrame(root)
settings_frame.pack(pady=20, padx=10, fill=tk.X)

btn = tk.Button(root, text="Start Training", command=start_training)
btn.pack(pady=20)

root.mainloop()

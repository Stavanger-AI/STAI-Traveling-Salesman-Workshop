# STAI-Traveling-Salesman-Workshop

The AI uses a technique called Q-learning to help a "traveling salesperson" find the shortest route to visit a list of cities exactly once and return to the starting city. It does this by learning which city to go to next, based on the distances between cities and the goal of minimizing total travel distance. The AI explores different routes, learns from its choices, and gradually figures out the best order to visit the cities.

# Get started

## Clone the code

```
https://github.com/Stavanger-AI/STAI_TSP.git
```

or download the zip file

## Installing Packages from `requirements.txt`

To install the Python packages listed in a `requirements.txt` file, you can use the `pip` package manager. The `requirements.txt` file typically contains a list of package names and their versions that are required for a Python project. Here are the steps to install the packages from a `requirements.txt` file:

1.  **Open a Terminal or Command Prompt**: To begin, open a terminal or command prompt on your computer.

2.  **Navigate to Your Project Directory**: Use the `cd` command to navigate to the directory where your `requirements.txt` file is located. For example:

    ```bash
    cd /path/to/your/project
    ```

    ```
    pip install -r requirements.txt
    ```

## Brief Explanation of Parameters

### Learning Rate (`lr`)

- **Purpose:** Determines how quickly the agent updates its knowledge.
- **Impact:** A higher learning rate leads to rapid adoption of new information, while a lower rate results in greater reliance on past knowledge.

### Epsilon (`epsilon`)

- **Purpose:** Controls the balance between exploration (trying new routes) and exploitation (using known routes).
- **Impact:** A higher epsilon value encourages more exploration.

### Epsilon Decay (`epsilon_decay`)

- **Purpose:** Gradually reduces the value of epsilon over time.
- **Impact:** Leads the agent to shift from exploring new routes to exploiting the best-known routes.

### Epsilon Minimum (`epsilon_min`)

- **Purpose:** Sets the lowest possible value for epsilon.
- **Impact:** Ensures that there's always some level of exploration, regardless of the agent's experience.

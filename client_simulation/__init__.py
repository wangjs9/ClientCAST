import os

ROOT_DIR = (
        os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)) + os.path.sep
)
from client_simulation.client_agent import SimulatedClient

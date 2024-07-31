import os

ROOT_DIR = (
        os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)) + os.path.sep
)
from therapist_simulation.therapist_agent import SimulatedTherapist

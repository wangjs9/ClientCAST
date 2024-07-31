from chatarena.backends import Human
import sys

SYSTEM_NAME = "System"


class HumanInteract(Human):
    def query(
            self,
            agent_name: str,
            **kwargs,
    ) -> str:
        """
        Format the input and call the ChatGPT/GPT-4 API.
        """
        try:
            response = sys.stdin.strip()
        except AttributeError:
            response = input(f"{agent_name}: ")
        return response

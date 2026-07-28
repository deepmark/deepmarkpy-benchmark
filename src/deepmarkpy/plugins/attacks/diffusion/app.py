from deepmarkpy.plugins.attacks.diffusion.implementation import DiffusionImplementation
from deepmarkpy.server.attack_service import create_attack_app


app = create_attack_app(DiffusionImplementation, attack_name="diffusion")

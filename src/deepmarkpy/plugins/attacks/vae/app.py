from deepmarkpy.plugins.attacks.vae.implementation import VAEImplementation
from deepmarkpy.server.attack_service import create_attack_app


app = create_attack_app(VAEImplementation, attack_name="vae")

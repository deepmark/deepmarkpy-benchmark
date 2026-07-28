from deepmarkpy.plugins.attacks.encodec.implementation import EncodecImplementation
from deepmarkpy.server.attack_service import create_attack_app


app = create_attack_app(EncodecImplementation, attack_name="encodec")

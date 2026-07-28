from deepmarkpy.plugins.attacks.neural_vocoder.implementation import (
    NeuralVocoderImplementation,
)
from deepmarkpy.server.attack_service import create_attack_app


app = create_attack_app(NeuralVocoderImplementation, attack_name="neural-vocoder")

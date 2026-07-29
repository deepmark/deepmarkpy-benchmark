from deepmarkpy.plugins.attacks.speech_tokenization.implementation import (
    SpeechTokenizationImplementation,
)
from deepmarkpy.server.attack_service import create_attack_app


app = create_attack_app(
    SpeechTokenizationImplementation,
    attack_name="speech-tokenization",
)

from deepmarkpy.plugins.attacks.speech_enhancement_2.implementation import (
    SpeechEnhancement2Implementation,
)
from deepmarkpy.server.attack_service import create_attack_app


app = create_attack_app(
    SpeechEnhancement2Implementation,
    attack_name="speech-enhancement-2",
)

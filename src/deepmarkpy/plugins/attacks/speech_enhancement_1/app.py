from deepmarkpy.plugins.attacks.speech_enhancement_1.implementation import (
    SpeechEnhancement1Implementation,
)
from deepmarkpy.server.attack_service import create_attack_app


app = create_attack_app(
    SpeechEnhancement1Implementation,
    attack_name="speech-enhancement-1",
)

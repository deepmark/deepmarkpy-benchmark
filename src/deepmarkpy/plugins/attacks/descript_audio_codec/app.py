from deepmarkpy.plugins.attacks.descript_audio_codec.implementation import (
    DescriptAudioCodecImplementation,
)
from deepmarkpy.server.attack_service import create_attack_app


app = create_attack_app(
    DescriptAudioCodecImplementation,
    attack_name="descript-audio-codec",
)

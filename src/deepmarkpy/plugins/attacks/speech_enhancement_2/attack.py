from deepmarkpy.core.remote_attack import RemoteAttack


class SpeechEnhancement2Attack(RemoteAttack):
    """Invoke the isolated ClearVoice enhancement service."""

    port_env = "SPEECH_ENHANCEMENT_PORT2"
    default_port = 10006

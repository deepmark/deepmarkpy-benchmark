from deepmarkpy.core.remote_attack import RemoteAttack


class SpeechEnhancement1Attack(RemoteAttack):
    """Invoke the isolated SpeechBrain enhancement service."""

    port_env = "SPEECH_ENHANCEMENT_PORT1"
    default_port = 10005

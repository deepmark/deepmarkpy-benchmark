from deepmarkpy.core.remote_attack import RemoteAttack


class SpeechTokenizationAttack(RemoteAttack):
    """Invoke the isolated XCodec2 speech-tokenization service."""

    port_env = "SPEECH_TOKENIZATION_PORT"
    default_port = 10003

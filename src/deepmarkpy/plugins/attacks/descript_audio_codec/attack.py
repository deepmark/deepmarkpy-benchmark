from deepmarkpy.core.remote_attack import RemoteAttack


class DescriptAudioCodecAttack(RemoteAttack):
    """Invoke the isolated Descript Audio Codec service."""

    port_env = "DESCRIPT_AUDIO_CODEC_PORT"
    default_port = 10008

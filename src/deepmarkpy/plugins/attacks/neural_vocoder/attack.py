from deepmarkpy.core.remote_attack import RemoteAttack


class NeuralVocoderAttack(RemoteAttack):
    """Invoke the isolated BigVGAN neural-vocoder service."""

    port_env = "NEURAL_VOCODER_PORT"
    default_port = 10004

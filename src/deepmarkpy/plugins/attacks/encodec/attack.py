from deepmarkpy.core.remote_attack import RemoteAttack


class EncodecAttack(RemoteAttack):
    """Invoke the isolated Encodec service."""

    port_env = "ENCODEC_PORT"
    default_port = 10007

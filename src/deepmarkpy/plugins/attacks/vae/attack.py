from deepmarkpy.core.remote_attack import RemoteAttack


class VAEAttack(RemoteAttack):
    """Invoke the isolated VAE attack service."""

    port_env = "VAE_PORT"
    default_port = 10001

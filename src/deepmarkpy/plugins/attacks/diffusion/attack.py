from deepmarkpy.core.remote_attack import RemoteAttack


class DiffusionAttack(RemoteAttack):
    """Invoke the isolated diffusion attack service."""

    port_env = "DIFFUSION_PORT"
    default_port = 10002

from typing import Awaitable, cast

import grpc
from ...type import GetRouteRequest, GetRouteResponse
from pycityproto.city.routing.v2 import routing_service_pb2_grpc as routing_grpc

__all__ = [
    "RoutingClient",
]


def _create_aio_channel(server_address: str, secure: bool = False) -> grpc.aio.Channel:
    """
    Create a grpc asynchronous channel

    Args:
    - server_address (str): server address.
    - secure (bool, optional): Defaults to False. Whether to use a secure connection. Defaults to False.

    Returns:
    - grpc.aio.Channel: grpc asynchronous channel.
    """
    if server_address.startswith("http://"):
        server_address = server_address.split("//")[1]
        if secure:
            raise ValueError("secure channel must use `https` or not use `http`")
    elif server_address.startswith("https://"):
        server_address = server_address.split("//")[1]
        if not secure:
            secure = True

    if secure:
        return grpc.aio.secure_channel(server_address, grpc.ssl_channel_credentials())
    else:
        return grpc.aio.insecure_channel(server_address)


class RoutingClient:
    """
    Client side of Routing service
    """

    def __init__(self, server_address: str, secure: bool = False):
        """
        Constructor of RoutingClient

        Args:
        - server_address (str): Routing server address
        - secure (bool, optional): Defaults to False. Whether to use a secure connection. Defaults to False.
        """
        aio_channel = _create_aio_channel(server_address, secure)
        self._aio_stub = routing_grpc.RoutingServiceStub(aio_channel)

    async def GetRoute(
        self,
        req: GetRouteRequest,
    ) -> GetRouteResponse:
        """
        Request navigation

        Args:
        - req (routing_service.GetRouteRequest): https://cityproto.sim.fiblab.net/#city.routing.v2.GetRouteRequest

        Returns:
        - https://cityproto.sim.fiblab.net/#city.routing.v2.GetRouteResponse
        """
        res = cast(Awaitable[GetRouteResponse], self._aio_stub.GetRoute(req))
        return await res

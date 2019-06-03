import numpy as np
import pathlib
import time
import sys
import os
import matplotlib
import pandas as pd
from dataclasses import asdict
import functools
from typing import Optional, Tuple
from pvtrace.light.utils import wavelength_to_rgb
from pvtrace.scene.scene import Scene
from pvtrace.algorithm import photon_tracer
from pvtrace.scene.node import Node
from pvtrace.common.errors import TraceError
from pvtrace.light.light import Light
from pvtrace.light.ray import Ray
from pvtrace.geometry.mesh import Mesh
from pvtrace.geometry.box import Box
from pvtrace.geometry.sphere import Sphere
from pvtrace.material.dielectric import Dielectric, LossyDielectric
from pvtrace.material.lumophore import Lumophore
from pvtrace.material.host import Host
from pvtrace.material.coating import Coating, CoatingDelegate
from pvtrace.scene.node import Node
from pvtrace.scene.renderer import MeshcatRenderer
from pvtrace.geometry.utils import magnitude
import logging

# We want to see pvtrace logging here
#logging.getLogger('pvtrace').setLevel(logging.CRITICAL)
logging.getLogger('trimesh').setLevel(logging.CRITICAL)
logging.getLogger('matplotlib').setLevel(logging.CRITICAL)

wavelength_range = (200, 800)
wavelength = np.linspace(*wavelength_range, 1000)
# Make a world coordinate system
world_node = Node(name='world')
world_node.geometry = Sphere(
    radius=10.0,
    material=Dielectric.air()
)

# Coating and delegate
class BoxCoatingDelegate(CoatingDelegate):
    """ An object that implements the coating delegate protocol.
        
        The edge surfaces are covered with solar cells and the bottom surface is 
        metalised with an evaporated mirror.
    """
    TOP_SURFACE_METALISATION = "METAL"
    BOTTOM_SURFACE_METALISATION = "BOOTY METAL"
    EDGE_SOLAR_CELL = "CELL"

    def coating_identifier(self, coating: Coating, geometry: Box, surface_point: Tuple[float, float, float]) -> Optional[str]:
        if not isinstance(geometry, Box):
            raise ValueError("Requires a box geometry.")
        zmax = geometry.size[2] * 0.5
        zmin = geometry.size[2] * 0.5
        z = surface_point[2]
        if np.isclose(z, zmax):
            return self.TOP_SURFACE_METALISATION
        elif np.isclose(z, zmin):
            return self.BOTTOM_SURFACE_METALISATION
        elif np.any([np.isclose( 0.5*geometry.size[0], surface_point[0]),
                     np.isclose(-0.5*geometry.size[0], surface_point[0]),
                     np.isclose( 0.5*geometry.size[1], surface_point[1]),
                     np.isclose(-0.5*geometry.size[1], surface_point[1])]):
            return self.EDGE_SOLAR_CELL
        return None

    def event_probabilities(self, coating: Coating, geometry: Box, coating_identifier: str, ray: Ray, normal: Tuple[float, float, float], angle: float) -> Tuple:
        if coating_identifier in (self.BOTTOM_SURFACE_METALISATION, self.TOP_SURFACE_METALISATION):
            # 95% reflectivity; 0% transmissive; 5% absorptivity
            return (0.95, 0.0, 0.05)
        elif coating_identifier == self.EDGE_SOLAR_CELL:
            # 0% reflectivity; 0% transmissive; 100% absorptivity
            return (0.0, 0.0, 1.0)
        else:
            raise ValueError("Coating does not have a known identifier.")

    def wants_specular_reflection(self, coating: Coating, geometry: Box, ray: Ray, normal: Tuple[float, float, float], angle: float) -> bool:
        return True

    def reflect(self, coating: Coating, geometry: Box, ray: Ray, normal: Tuple[float, float, float], angle: float) -> Ray:
        raise NotImplementedError("Not needed")

    def transmit(self, coating: Coating, geometry: Box, ray: Ray, normal: Tuple[float, float, float], angle: float) -> Ray:
        raise NotImplementedError("Not needed")


# Add LSC
size = (1.0, 1.0, 0.25)
lsc = Node(name="box", parent=world_node)
lsc.geometry = Box(
    size, 
    material=LossyDielectric.make_constant((300, 1000), 1.5, 100.0),
    coating=Coating(delegate=BoxCoatingDelegate())
)

# Light source hitting top surface
light_node1 = Node(
    name='light top',
    parent=world_node,
    location=(0.0, 0.0, 1.0),
    light=Light(
        divergence_delegate=functools.partial(
            Light.cone_divergence,
            np.radians(20)
        )
    )
)
light_node1.rotate(np.radians(180), (1, 0, 0))

# Light node hitting an edge surface
light_node2 = Node(
    name="light edge",
    parent=world_node,
    location=(2.0, 0.0, 0.0),
    light=Light(
        divergence_delegate=functools.partial(
            Light.cone_divergence,
            np.radians(5.0)
        )
    )
)
light_node2.rotate(np.radians(-90), (0, 1, 0))

scene = Scene(root=world_node)
renderer = MeshcatRenderer(max_histories=None, open_browser=True)
renderer.render(scene)


if __name__ == "__main__":
    import time
    time.sleep(0.1)
    np.random.seed(1)
    for ray in scene.emit(20):
        steps = photon_tracer.follow(ray, scene, renderer=renderer)
        path, decisions = zip(*steps)
        print(list(map(lambda x: x.position, path)))
        print(decisions)
        renderer.add_ray_path(path)
        time.sleep(0.1)

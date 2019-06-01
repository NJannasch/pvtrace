import pytest
from typing import Tuple, Optional
import numpy as np
from dataclasses import replace
from pvtrace.geometry.box import Box
from pvtrace.geometry.utils import EPS_ZERO, distance_between
from pvtrace.material.material import Decision
from pvtrace.material.coating import Coating, CoatingDelegate
from pvtrace.light.ray import Ray


class TopSurfaceCoating(CoatingDelegate):
    """ Coating applied to the top surface of a box geometry.
    """

    def __init__(self):
        super(TopSurfaceCoating, self).__init__()
        self._base = np.array([100.0, 100.0, 1.0])

    def coating_identifier(self, coating: Coating, geometry: Box, surface_point: Tuple[float, float, float]) -> Optional[str]:
        if not isinstance(geometry, Box):
            raise ValueError("Requires a box geometry.")
        zmax = geometry.size[2] * 0.5
        z = surface_point[2]
        if np.isclose(z, zmax):
            return "COATING"
        return None  # Is not a coating

    def event_probabilities(self, coating: Coating, geometry: Box, coating_identifier: str, ray: Ray, normal: Tuple[float, float, float], angle: float) -> Tuple:
        p = np.array(self._base)
        p = p/np.sum(p)
        return p

    def wants_specular_reflection(self, coating: Coating, geometry: Box, ray: Ray, normal: Tuple[float, float, float], angle: float) -> bool:
        return True

    def reflect(self, coating: Coating, geometry: Box, ray: Ray, normal: Tuple[float, float, float], angle: float) -> Ray:
        pass

    def transmit(self, coating: Coating, geometry: Box, ray: Ray, normal: Tuple[float, float, float], angle: float) -> Ray:
        return ray.propagate(2*EPS_ZERO)


def test_init():
    assert isinstance(Coating(None), Coating)
    assert isinstance(TopSurfaceCoating(), TopSurfaceCoating)
    
    
def test_delegation():
    
    delegate = TopSurfaceCoating()
    coating = Coating(delegate=delegate)
    geometry = Box((1.0, 1.0, 1.0), material=None, coating=coating)
    assert coating == geometry.coating
    assert delegate.coating_identifier(coating, geometry, (0.0, 0.0, 0.5)) == "COATING"
    assert delegate.coating_identifier(coating, geometry, (0.0, 0.0, 0.6)) is None
    delegate._base = [0.0, 1.0, 0.0]
    assert np.allclose(
        delegate.event_probabilities(
            coating, 
            geometry,
            "COATING",  # identifier
            None,  # Ray
            None,  # normal
            None   # angle
        ),
        [0.0, 1.0, 0.0]
    )
    

def test_transform():
    delegate = TopSurfaceCoating()
    coating = Coating(delegate=delegate)
    geometry = Box((1.0, 1.0, 1.0), material=None, coating=coating)
    
    ray = Ray(position=(0.0, 0.0, 0.5), direction=(0.0, 0.0, -1.0), wavelength=555.0, is_alive=True)
    assert coating.is_hit(ray, geometry) == True
    
    miss_ray = Ray(position=(0.0, 0.0, -0.5), direction=(0.0, 0.0, -1.0), wavelength=555.0, is_alive=True)
    assert coating.is_hit(miss_ray, geometry) == False

    delegate._base = [1.0, 0.0, 0.0]  # always reflect
    new_ray, decision = coating.transform(geometry, ray, {"normal": (0.0, 0.0, 1.0)})
    assert np.allclose(new_ray.direction, (0.0, 0.0, 1.0)) == True
    assert distance_between(new_ray.position, ray.position) > EPS_ZERO
    assert geometry.is_on_surface(new_ray.position) == False
    assert decision == Decision.RETURN

    delegate._base = [0.0, 1.0, 0.0]  # always transmit
    new_ray, decision = coating.transform(geometry, ray, {"normal": (0.0, 0.0, 1.0)})
    assert np.allclose(new_ray.direction, (0.0, 0.0, -1.0)) == True
    assert distance_between(new_ray.position, ray.position) > EPS_ZERO
    assert geometry.is_on_surface(new_ray.position) == False
    assert decision == Decision.TRANSIT

    delegate._base = [0.0, 0.0, 1.0]  # always absorb
    new_ray, decision = coating.transform(geometry, ray, {"normal": (0.0, 0.0, 1.0)})
    assert np.allclose(new_ray.direction, ray.direction) == True
    assert distance_between(new_ray.position, ray.position) < EPS_ZERO
    assert geometry.is_on_surface(new_ray.position) == True
    assert new_ray.is_alive == False
    assert decision == Decision.ABSORB


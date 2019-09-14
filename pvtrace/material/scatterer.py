from pvtrace.material.material import Material, Decision
from pvtrace.material.properties import Diffusive
from pvtrace.material.mechanisms import Scatter, CrossInterface, TravelPath
from pvtrace.geometry.transformations import identity_matrix, rotation_matrix, inverse_matrix
from dataclasses import replace
from typing import Tuple
import numpy as np
import pandas as pd
import logging
logger = logging.getLogger(__name__)


class Scatterer(Diffusive, Material):
    """ A material scatters light.

        This is initial attempt at scattering is very limited.
    
        The class takes a scattering coefficient (cm-1) and a scattering half-angle
        in radians. The scattering angle will be found by uniformly sampling the 
        cone of the same half-angle pointing in the same direction as the ray.
    """

    def __init__(
        self,
        scattering_coefficient: np.ndarray,
        cone_half_angle: float,
    ):
        super(Scatterer, self).__init__(
            scattering_coefficient=scattering_coefficient,
            cone_half_angle=cone_half_angle
        )
        self._transit_mechanism = CrossInterface()  # Only transmits at interfaces
        self._return_mechanism = None  # Never reflects at interfaces
        self._path_mechanism = TravelPath()
        self._emit_mechanism = Scatter()  # Never emits

    def deflect(self, direction: np.ndarray) -> np.ndarray:
        """ Scatter a ray.
        """
        v1 = np.array(direction)
        v2 = np.array([0.0, 0.0, 1.0])
        v = np.cross(v1, v2)

        # https://math.stackexchange.com/questions/180418/calculate-rotation-matrix-to-align-vector-a-to-vector-b-in-3d
        def skew(v):
            return np.matrix(
                [
                    [0.0, -v[2], v[1]],
                    [v[2], 0.0, -v[0]],
                    [-v[1], v[0], 0.0]
                ]
            )

        s =  skew(v)
        c = np.dot(v1, v2)
        
        if np.isclose(c, -1.0):
            # anti-parallel ray
            r1 = rotation_matrix(np.radians(180.0), [1.0, 0.0, 0.0])
        else:
            # Rotation matrix to align ray to [0, 0, 1]
            r1 = identity_matrix()[:-1,:-1] + s + s**2 * 1.0 / (1.0 + c)
            temp = identity_matrix()
            temp[:-1,:-1] = r1
            r1 = np.matrix(temp)
        
        phi_max = self.cone_half_angle
        theta = np.random.uniform(0, 2 * np.pi)
        phi = np.arccos(np.random.uniform(np.cos(phi_max), 1.0))
        x = np.sin(phi) * np.cos(theta)
        y = np.sin(phi) * np.sin(theta)
        z = np.cos(phi)
        v3 = np.array([x, y, z])
        # Angle between [0, 0, 1] and new scattering angle
        c = np.dot(v2, v3)
        # Rotation matrix align from [0, 0, 1] to scattered vector
        r2 = np.matrix(rotation_matrix(np.cos(c), v3))
    
        # Transform ray to [0, 0, 1], apply scattering rotation, transform ray back
        v1 = np.array([*v1, 0])
        v1 = np.matrix(v1).transpose()
        v3 = r1 * r2 * inverse_matrix(r1) * v1
        new_direction = np.array(v3)[0:3, 0]
        return new_direction

    def trace_path(
            self, 
            local_ray: "Ray",
            container_geometry: "Geometry",
            distance: float
    ) -> Tuple[Decision, dict]:
        
        # Sample the exponential distribution and get a distance at which the
        # ray is absorbed.
        sampled_distance = self._emit_mechanism.path_length(
            local_ray.wavelength, container_geometry.material
        )
        logger.debug("Host.trace_path args: {}".format((local_ray, container_geometry, distance)))
        # If the sampled distance is less than the full distance the ray can travel
        # then the ray is absorbed.
        if sampled_distance < distance:
            # Apply the absorption transformation to the ray; this updates the rays
            # position to the absorption location.
            info = {
                "distance": sampled_distance,
                "material": container_geometry.material
            }
            new_ray = self._emit_mechanism.transform(local_ray, info)
            yield new_ray, Decision.EMIT
        else:
            # If not absorbed travel the full distance
            info = {"distance": distance}
            new_ray = self._path_mechanism.transform(local_ray, info)
            yield new_ray, Decision.TRAVEL



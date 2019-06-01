from __future__ import annotations
import sys
import abc
import numpy as np
from typing import Tuple
from dataclasses import replace
from pvtrace.light.ray import Ray
from pvtrace.material.material import Decision
from pvtrace.material.mechanisms import FresnelReflection
from pvtrace.geometry.utils import flip, angle_between
from pvtrace.common.errors import TraceError
import logging
logger = logging.getLogger(__name__)


class CoatingDelegate(abc.ABC):
    """ Supplies coating with data. User implemented.
    """

    @abc.abstractmethod
    def coating_identifier(self, coating: Coating, geometry: Geometry, surface_point: Tuple[float, float, float]) -> Optional[str]:
        """ Returns an identifier for this coating that was hit or None if a coating was not hit.
        """
        pass

    @abc.abstractmethod
    def event_probabilities(self, coating: Coating, geometry: Geometry, coating_identifier: str, ray: Ray, normal: Tuple[float, float, float], angle: float) -> Tuple[float, float, float]:
        """ Returns a tuple of probabilities for the different processes
        
            Notes
            -----
            The returned tuple should be of exactly three floats::
        
                (p1, p2, p3)
            
            where p1 + p2 + p3 = 1.
        
            p1 is the probability of reflection, p2 is the probability of trasmission
            and p3 is the probability of being non-radiatively absorbed.
        """
        pass

    @abc.abstractmethod
    def wants_specular_reflection(self, coating: Coating, geometry: Geometry, ray: Ray, normal: Tuple[float, float, float], angle: float) -> bool:
        """ Choose between automatic specular reflection or custom reflection mode.
            
            Returns
            -------
            bool
                True to indicate the ray should be specularly reflected, False otherwise.

            Notes
            -----
            If this method returns False the `reflect` method will be called on the 
            delegate object and it is up to the user to reflect the ray. Use this
            to implement non-specular reflections, scattering or diffraction.
        """
        pass

    def reflect(self, coating: Coating, geometry: Geometry, ray: Ray, normal: Tuple[float, float, float], angle: float) -> Ray:
        """ Returns a reflected ray.
        
            Notes
            -----
            The new ray should have a reflected trajectory and be moved sufficiently
            forward that is no longer on the surface of the geometry object.
        """
        pass

    def transmit(self, coating: Coating, geometry: Geometry, ray: Ray, normal: Tuple[float, float, float], angle: float) -> Ray:
        """ Returns a transmitted ray.
        
            Notes
            -----
            The new ray should have a transmitted trajectory and be moved sufficiently
            forward that is no longer on the surface of the geometry object.
        """
        pass


class Coating(object):
    """ An property of a geometry that implements custom interactions at surfaces.
    
        Coating transform rays when they hit a surface, either by reflecting it
        transmitting it or killing the ray. Unlike the Fresnel interactions, no physical
        limitions are imposed on how the surface interaction is treated. For example,
        coatings are used to simulate surfaces which have custom reflective, 
        diffractive or scattering properties.
    
        A spacial mask can be applied if the coating is only partially applied to a
        surface.
    
        By default a coating covers the *full* surface of the geometry.

        Coatings use the delegate pattern. Users should implemented the methods of 
        CoatingDelegate.
    """
    
    def __init__(self, delegate: CoatingDelegate):
        """
        """
        self.delegate = delegate
    
    def is_hit(self, local_surface_ray: Ray, geometry: Geometry) -> bool:
        """ Determine if the ray hits a part of the surface that is coated.
        """
        surface_point = local_surface_ray.position
        if self.delegate.coating_identifier(self, geometry, surface_point) is None:
            return False
        return True

    def transform(self, geometry: Geometry, local_ray: Ray, context: dict) -> Tuple[Ray, Decision]:
        
        identifier = self.delegate.coating_identifier(
            self, geometry, local_ray.position
        )
        if identifier is None:
            raise TraceError("Hit coating returned null identifier.")

        normal = context["normal"]
        if np.dot(normal, local_ray.direction) < 0.0:
            normal_for_angle = flip(normal)
        else:
            normal_for_angle = normal
        angle = angle_between(normal_for_angle, np.array(local_ray.direction))
        if angle < 0.0 or angle > 0.5 * np.pi:
            raise TraceError("The incident angle must be between 0 and pi/2.")
    
        # tuple like (p_reflection, p_transmission, p_absorption)
        p = self.delegate.event_probabilities(
            self, geometry, identifier, local_ray, normal, angle
        )
        if len(p) != 3:
            raise TraceError("Three probabilities are needed.")
        elif not np.isclose(np.sum(p), 1.0):
            raise TraceError("Probabilities must sum to unity.")

        outcomes = [Decision.RETURN, Decision.TRANSIT, Decision.ABSORB]
        decision = np.random.choice(outcomes, p=p)
        if decision == Decision.ABSORB:
            new_ray = replace(local_ray, is_alive=False)
            return new_ray, decision
        elif decision == Decision.RETURN:
            if self.delegate.wants_specular_reflection(
                self, geometry, local_ray, normal, angle
            ):
                new_ray = FresnelReflection().transform(local_ray, {"normal": normal})
            else:
                new_ray = self.delegate.reflect(
                    self, geometry, local_ray, normal, angle
                )
            return new_ray, decision
        elif decision == Decision.TRANSIT:
            new_ray = self.delegate.transmit(
                self, geometry, local_ray, normal, angle
            )
            return new_ray, decision

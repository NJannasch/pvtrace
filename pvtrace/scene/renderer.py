import numpy as np
import os
import time
import io
from typing import Tuple
from contextlib import contextmanager
from collections import deque
from anytree import LevelOrderIter, PostOrderIter
from pvtrace.geometry.sphere import Sphere
from pvtrace.geometry.cylinder import Cylinder
from pvtrace.geometry.mesh import Mesh
from pvtrace.light.ray import Ray
from pvtrace.light.utils import wavelength_to_rgb, rgb_to_hex_int, wavelength_to_hex_int
import trimesh
import meshcat
import meshcat.geometry as g
import meshcat.transformations as tf
import logging
logger = logging.getLogger(__name__)


class MeshcatRenderer(object):
    """Renders a scene nodes structure."""

    def __init__(self, zmq_url=None, max_histories=10000, open_browser=False):
        super(MeshcatRenderer, self).__init__()
        self.vis = meshcat.Visualizer(zmq_url=zmq_url)
        if open_browser:
            self.vis.open()
        self.ray_histories = deque(maxlen=max_histories)
        self.max_histories = max_histories
        self.added_index = 0

    def render(self, scene, show_root=False):
        """
        """
        vis = self.vis
        for node in LevelOrderIter(scene.root):
            if node == scene.root:
                continue
            self.add_node(node)

    def add_node(self, node):
        pathname = "/".join([x.name for x in node.path])
        transform = node.pose
        if node.geometry is not None:
            self.add_geometry(node.geometry, pathname, transform)

    def add_geometry(self, geometry, pathname, transform):
        vis = self.vis
        material = g.MeshLambertMaterial(reflectivity=1.0, sides=0)
        material.transparency = True
        material.opacity = 0.5
        if isinstance(geometry, Sphere):
            sphere = geometry
            vis[pathname].set_object(
                g.Sphere(sphere.radius),
                material)
            vis[pathname].set_transform(transform)
        elif isinstance(geometry, Cylinder):
            cyl = geometry
            vis[pathname].set_object(
                g.Cylinder(cyl.length, cyl.radius),
                material
            )
            # meshcat cylinder is aligned along y-axis. Align along z then apply the
            # node's transform as normal.
            vis[pathname].set_transform(
                transform.dot(
                    tf.rotation_matrix(np.radians(-90), [1, 0, 0])
                )
            )
        elif isinstance(geometry, Mesh):
                obj = meshcat.geometry.StlMeshGeometry.from_stream(
                    io.BytesIO(trimesh.exchange.stl.export_stl(geometry.trimesh))
                )
                vis[pathname].set_object(obj, material)
                vis[pathname].set_transform(transform)
        else:
            raise NotImplementedError("Cannot yet add {} to visualiser".format(type(geometry)))

    def remove(self, scene):
        vis = self.vis
        vis.delete()
    
    def update_transform(self, node):
        vis = self.vis
        pathname = "/".join([x.name for x in node.path])
        vis[pathname].set_transform(node.pose)

    def get_next_identifer(self):
        self.added_index += 1
        return "rays/{}".format(str(self.added_index))

    def add_line_segment(self,
        start: Tuple[float, float, float],
        end: Tuple[float, float, float],
        colour=0xffffff) -> str:
        """ Add a line segment to the scene and return the identifier.
        
            Parameters
            ----------
            start : tuple
                The starting point of the line as (x, y, z) coordinates.
            end : tuple
                The ending point of the line as (x, y, z) coordinates.
            colour : int (optional)
                An optional colour specified as a hex integer. The default colour is
                white.

            Returns
            -------
            identifier : str
                The string identifier used to add the line to the scene.
        """
        vis = self.vis
        line = (start, end)
        self._will_add_expendable_to_scene(line)
        vertices = np.column_stack(line)
        assert vertices.shape[0] == 3  # easy to get this wrong
        identifier = self.get_next_identifer()
        vis[identifier].set_object(
            g.Line(g.PointsGeometry(vertices),
            g.MeshBasicMaterial(color=colour, transparency=True, opacity=0.5))
        )
        self._did_add_expendable_to_scene(identifier)
        return identifier

    def add_path(self, vertices: Tuple[Tuple[float, float, float]], colour=0xffffff) -> str:
        """ Add a line to the scene and return the identifier. The line is made from 
            multiple line segments. The line will be drawn with a single colour.
        
            Parameters
            ----------
            vertices : tuple of tuple of float
                The starting point of the line as (x, y, z) coordinates.
            colour : int (optional)
                An optional colour specified as a hex integer. The default colour is
                white.

            See also
            --------
            add_ray_path : Draws the line using individual line segments. Use this 
            method when each line segment needs to be drawn with a different colour.
        
            Returns
            -------
            identifier : str
                The string identifier used to add the line to the scene.
        """
        vis = self.vis
        self._will_add_expendable_to_scene(vertices)
        vertices = np.array(vertices)
        assert vertices.shape[0] == 3  # easy to get this wrong
        identifier = self.get_next_identifer()
        vis[identifier].set_object(
            g.Line(g.PointsGeometry(vertices),
            g.MeshBasicMaterial(color=colour, transparency=True, opacity=0.5))
        )
        self._did_add_expendable_to_scene(identifier)
        return identifier
    
    def add_ray(self, ray : Ray, length: float) -> str:
        """ Add the ray path as a single connected line and return an identifier. 
        
            Parameters
            ----------
            ray : Ray
                The ray to add to the scene.

            Notes
            -----
            Internally the line is drawn using `add_line_segment` because the colour of
            each segment could be unique. If this proves too inefficiency use 
            `add_path`.

            See also
            --------
            add_ray_path : Adds multiple rays to the scene.

            Returns
            -------
            identifier : str
                The string identifier used to add the object to the scene.
        """
        nanometers = ray.wavelength
        start = ray.position
        end = np.array(start) + np.array(ray.direction) * length
        colour = wavelength_to_hex_int(nanometers)
        identifier = self.add_line_segment(start, end, colour=colour)
        return identifier

    def add_ray_path(self, rays: [Ray]) -> str:
        """ Add the ray path as a single connected line and return an identifier. 
        
            Parameters
            ----------
            rays : list of Ray
                List of ray objects.
            length : float
                The length of the line to render. Default to 1000.

            See also
            --------
            add_path : Draws the line in more efficient way than `add_ray_path` but
                limits the line to be a single colour.

            Returns
            -------
            identifier : str
                The string identifier used to add the line to the scene.
        """
        vis = self.vis
        if len(rays) < 2:
            raise AppError("Need at least two points to render a line.")
        for (start_ray, end_ray) in zip(rays[:-1], rays[1:]):
            nanometers = start_ray.wavelength
            start = start_ray.position
            end = end_ray.position
            colour = wavelength_to_hex_int(nanometers)
            self.add_line_segment(start, end, colour=colour)

    def remove_object(self, identifier):
        """ Remove object by its identifier.
        """
        vis = self.vis
        vis[identifier].delete()

    def _will_add_expendable_to_scene(self, item):
        """ Private method used to notify buffer that a line or ray object will be
            added to the scene.
        
            Notes
            -----
            This is used to manage the buffer size and will remove the oldest object
            to keep the scene size constant.
        """
        if len(self.ray_histories) == self.max_histories:
            self.remove_object(self.ray_histories.popleft())
    
    def _did_add_expendable_to_scene(self, identifier):
        """ Private method use to notify the buffer that an expendable object has been
            added to the scene. 
        
            Notes
            -----
            The identifier is used to remove the object when it is becomes the oldest
            item in the buffer.
        """
        self.ray_histories.append(identifier)


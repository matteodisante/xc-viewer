"""Compose thermal colour and background in one shader on one DEM mesh.

There is no second plane or offset surface: both texture coordinates are computed
from the same terrain vertex. This prevents ground occlusion and depth fighting.
"""

import numpy as np
import pyqtgraph.opengl as gl
from OpenGL import GL
from pyqtgraph.opengl.shaders import FragmentShader, ShaderProgram, VertexShader

from .terrain_surface import TerrainMesh, upload_image


class _DrapeShader(ShaderProgram):
    """Extend pyqtgraph's float-only uniform setter with a second texture sampler."""

    def __enter__(self):
        """Bind the density sampler to texture unit one after the float uniforms."""
        super().__enter__()
        GL.glUniform1i(self.uniform("u_density"), 1)


DRAPE_SHADER = _DrapeShader(
    "soaring_thermal_drape",
    [
        VertexShader("""#version 140
        uniform mat4 u_mvp;
        uniform float u_bounds[4];
        uniform float u_density_bounds[4];
        in vec4 a_position;
        in vec4 a_color;
        out vec4 v_color;
        out vec2 v_photo;
        out vec2 v_density;
        void main() {
            gl_Position = u_mvp * a_position;
            v_color = a_color;
            v_photo = vec2(
                (a_position.x-u_bounds[0])/(u_bounds[2]-u_bounds[0]),
                (u_bounds[3]-a_position.y)/(u_bounds[3]-u_bounds[1]));
            v_density = vec2(
                (a_position.x-u_density_bounds[0])/(u_density_bounds[2]-u_density_bounds[0]),
                (u_density_bounds[3]-a_position.y)/(u_density_bounds[3]-u_density_bounds[1]));
        }
    """),
        FragmentShader("""#version 140
        uniform sampler2D u_texture;
        uniform sampler2D u_density;
        uniform float u_photo;
        uniform float u_terrain;
        uniform float u_density_opacity;
        in vec4 v_color;
        in vec2 v_photo;
        in vec2 v_density;
        out vec4 fragColor;
        void main() {
            vec4 base = v_color;
            if (u_photo > 0.5) base = texture(u_texture, v_photo);
            base.a *= u_terrain;
            vec4 heat = vec4(0.0);
            if (u_density_opacity > 0.0) {
                heat = texture(u_density, v_density);
                heat.a *= u_density_opacity;
            }
            float alpha = heat.a + base.a * (1.0-heat.a);
            if (alpha <= 0.0) discard;
            vec3 rgb = (heat.rgb*heat.a + base.rgb*base.a*(1.0-heat.a))/alpha;
            fragColor = vec4(rgb, alpha);
        }
    """),
    ],
)


class DrapedTerrain(TerrainMesh):
    """One terrain geometry with independent background and thermal visibility."""

    def __init__(self, **kwargs):
        """Keep density pixels separate from imagery but render a single surface."""
        super().__init__(**kwargs)
        self._density_image = self._density_texture = None
        self._empty_texture = None
        self._density_bounds = self._bounds
        self.density_opacity = 0.75
        self.density_visible = True
        self.terrain_visible = True
        self.setShader(DRAPE_SHADER)
        self.setGLOptions("translucent")
        self.updateGLOptions({"glDepthMask": (True,)})

    def set_aerial(self, enabled):
        """Choose imagery or vertex colours without changing the shared shader."""
        self._aerial = bool(enabled and self._image is not None)
        self.update()

    def release_density(self):
        """Destroy only the density texture with the view context current."""
        if self._density_texture is not None:
            self._density_texture.destroy()
            self._density_texture = None

    def release_texture(self):
        """Release both textures when the scene is replaced or closed."""
        self.release_density()
        if self._empty_texture is not None:
            self._empty_texture.destroy()
            self._empty_texture = None
        super().release_texture()

    def set_density(self, image, bounds):
        """Install pixels after the caller has released the previous texture."""
        self._density_image = image
        self._density_bounds = list(bounds)
        self._visibility()

    def set_density_visible(self, visible):
        """Hide thermal colour independently of the underlying terrain."""
        self.density_visible = visible
        self._visibility()

    def set_terrain_visible(self, visible):
        """Hide the background while retaining thermal colour on its 3-D geometry."""
        self.terrain_visible = visible
        self._visibility()

    def _visibility(self):
        """Keep the mesh drawable whenever either material layer is enabled."""
        self.setVisible(
            self.terrain_visible
            or (self.density_visible and self._density_image is not None)
        )
        self.update()

    def paint(self):
        """Blend both georeferenced layers before depth-testing the single surface."""
        density = self.density_visible and self._density_image is not None
        DRAPE_SHADER["u_bounds"] = self._bounds
        DRAPE_SHADER["u_density_bounds"] = self._density_bounds
        DRAPE_SHADER["u_photo"] = [float(self._aerial)]
        DRAPE_SHADER["u_terrain"] = [float(self.terrain_visible)]
        DRAPE_SHADER["u_density_opacity"] = [self.density_opacity if density else 0.0]
        GL.glActiveTexture(GL.GL_TEXTURE0)
        if (not self._aerial or not density) and self._empty_texture is None:
            self._empty_texture = upload_image(np.zeros((1, 1, 4), dtype=np.uint8))
        if self._aerial and self._texture is None:
            self._upload_texture()
        photo_texture = self._texture if self._aerial else self._empty_texture
        photo_texture.bind()
        GL.glActiveTexture(GL.GL_TEXTURE1)
        if density and self._density_texture is None:
            self._density_texture = upload_image(self._density_image)
        density_texture = self._density_texture if density else self._empty_texture
        density_texture.bind()
        try:
            gl.GLMeshItem.paint(self)
        finally:
            GL.glActiveTexture(GL.GL_TEXTURE1)
            density_texture.release()
            GL.glActiveTexture(GL.GL_TEXTURE0)
            photo_texture.release()

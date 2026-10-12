"""One DEM mesh, displayed with elevation colours or a georeferenced orthophoto."""

from __future__ import annotations

import pyqtgraph.opengl as gl
from OpenGL import GL
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QImage
from PyQt6.QtOpenGL import QOpenGLTexture
from pyqtgraph.opengl.shaders import FragmentShader, ShaderProgram, VertexShader

ORTHOPHOTO_SHADER = ShaderProgram(
    "soaring_orthophoto",
    [
        VertexShader("""#version 140
            uniform mat4 u_mvp;
            uniform float u_bounds[4];
            in vec4 a_position;
            out vec2 v_texcoord;
            void main() {
                gl_Position = u_mvp * a_position;
                // Image row zero is north. Do not flip or transpose the photo.
                v_texcoord = vec2(
                    (a_position.x - u_bounds[0]) / (u_bounds[2] - u_bounds[0]),
                    (u_bounds[3] - a_position.y) / (u_bounds[3] - u_bounds[1])
                );
            }
        """),
        FragmentShader("""#version 140
            uniform sampler2D u_texture;
            uniform float u_opacity;
            in vec2 v_texcoord;
            out vec4 fragColor;
            void main() {
                fragColor = texture(u_texture, v_texcoord);
                fragColor.a *= u_opacity;
            }
        """),
    ],
)


class _Orthophoto:
    """Share a north-up georeferenced texture between surface and mesh items."""

    def _prepare_image(self, image, bounds, base_shader):
        """Retain image data until a valid graphics context uploads it."""
        self._image = image
        self._texture = None
        self._aerial = False
        self.opacity = 1.0
        self._bounds = list(bounds)
        self._base_shader = base_shader

    def set_aerial(self, enabled):
        """Switch shaders without changing vertices, heights or camera state."""
        self._aerial = bool(enabled and self._image is not None)
        self.setShader(ORTHOPHOTO_SHADER if self._aerial else self._base_shader)

    def release_texture(self):
        """Release GPU storage while the view's shared context is current."""
        if self._texture is not None:
            self._texture.destroy()
            self._texture = None

    def _upload_texture(self):
        """Upload RGB imagery or an RGBA density; preserve transparent pixels."""
        self._texture = upload_image(self._image)

    def paint(self):
        """Sample the orthophoto by local east/north coordinates on each triangle."""
        if not self._aerial:
            super().paint()
            return
        if self._texture is None:
            self._upload_texture()
        ORTHOPHOTO_SHADER["u_bounds"] = self._bounds
        ORTHOPHOTO_SHADER["u_opacity"] = [self.opacity]
        # GLSL initializes the sampler uniform to texture unit zero.
        GL.glActiveTexture(GL.GL_TEXTURE0)
        self._texture.bind()
        try:
            super().paint()
        finally:
            self._texture.release()


class TerrainSurface(_Orthophoto, gl.GLSurfacePlotItem):
    """Keep a regular DEM surface while changing its appearance in place."""

    def __init__(self, *, image=None, **kwargs):
        """Retain the north-up RGB image and upload it only in photo mode."""
        super().__init__(**kwargs)
        self._prepare_image(
            image,
            [kwargs["x"][0], kwargs["y"][0], kwargs["x"][-1], kwargs["y"][-1]],
            "shaded",
        )


class TerrainMesh(_Orthophoto, gl.GLMeshItem):
    """Drape an orthophoto over valid DEM triangles, preserving coverage holes."""

    def __init__(self, *, image=None, bounds, **kwargs):
        """Map image corners to the mesh's west/south/east/north coordinates."""
        super().__init__(**kwargs)
        self._prepare_image(image, bounds, kwargs.get("shader"))


def upload_image(data):
    """Upload north-first RGB/RGBA pixels in the caller's current texture unit."""
    height, width = data.shape[:2]
    image_format = (
        QImage.Format.Format_RGBA8888
        if data.shape[2] == 4
        else QImage.Format.Format_RGB888
    )
    image = QImage(data.data, width, height, data.strides[0], image_format)
    limit = int(GL.glGetIntegerv(GL.GL_MAX_TEXTURE_SIZE))
    if max(width, height) > limit:
        image = image.scaled(
            limit,
            limit,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
    texture = QOpenGLTexture(image, QOpenGLTexture.MipMapGeneration.GenerateMipMaps)
    texture.setWrapMode(QOpenGLTexture.WrapMode.ClampToEdge)
    texture.setMinMagFilters(
        QOpenGLTexture.Filter.LinearMipMapLinear, QOpenGLTexture.Filter.Linear
    )
    return texture

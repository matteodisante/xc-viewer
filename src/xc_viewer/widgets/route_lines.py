"""Outlined trajectory strokes with real pixel widths on core-profile OpenGL.

Core-profile line primitives are one pixel wide on macOS. Expand each retained
edge into a screen-space capsule instead; no fix is thinned or moved in 3-D.
"""

import numpy as np
from OpenGL import GL
from OpenGL.GL import shaders
from pyqtgraph.opengl.items.GLLinePlotItem import DirtyFlag, GLLinePlotItem

VERTEX = """#version 150
uniform mat4 u_mvp;
in vec3 a_position;
void main() { gl_Position = u_mvp * vec4(a_position, 1.0); }
"""

GEOMETRY = """#version 150
layout(lines) in;
layout(triangle_strip, max_vertices=4) out;
uniform vec2 u_viewport;
uniform float u_radius;
noperspective out vec2 v_pixel;
flat out float v_length;
float edge_length;
void corner(vec4 position, vec2 offset, vec2 pixel) {
    gl_Position = position;
    gl_Position.xy += offset * 2.0 / u_viewport * position.w;
    v_pixel = pixel;
    v_length = edge_length;
    EmitVertex();
}
void main() {
    vec4 a = gl_in[0].gl_Position;
    vec4 b = gl_in[1].gl_Position;
    // Clip at the near plane before dividing by w, including camera fly-throughs.
    float an = a.z+a.w, bn = b.z+b.w;
    if (an < 0.0 && bn < 0.0) return;
    if (an < 0.0) a = mix(a, b, an/(an-bn));
    if (bn < 0.0) b = mix(b, a, bn/(bn-max(an, 0.0)));
    if (a.w <= 0.0 || b.w <= 0.0) return;
    vec2 delta = (b.xy/b.w-a.xy/a.w) * u_viewport * 0.5;
    float len = length(delta);
    vec2 along = len > 0.0001 ? delta/len : vec2(1.0, 0.0);
    vec2 normal = vec2(-along.y, along.x);
    edge_length = len;
    corner(a, (-along-normal)*u_radius, vec2(-u_radius, -u_radius));
    corner(a, (-along+normal)*u_radius, vec2(-u_radius, u_radius));
    corner(b, (along-normal)*u_radius, vec2(len+u_radius, -u_radius));
    corner(b, (along+normal)*u_radius, vec2(len+u_radius, u_radius));
    EndPrimitive();
}
"""

FRAGMENT = """#version 150
uniform vec4 u_colour;
uniform float u_half_width;
uniform float u_outline;
uniform float u_aa;
uniform int u_fill;
noperspective in vec2 v_pixel;
flat in float v_length;
out vec4 fragColor;
void main() {
    float cap = max(max(-v_pixel.x, v_pixel.x-v_length), 0.0);
    float d = length(vec2(cap, v_pixel.y));
    float outer = u_half_width + (u_fill == 0 ? u_outline : 0.0);
    float alpha = 1.0-smoothstep(outer-u_aa*0.5, outer+u_aa*0.5, d);
    if (alpha <= 0.0) discard;
    vec3 rgb = u_fill == 0 ? vec3(0.025, 0.045, 0.07) : u_colour.rgb;
    fragColor = vec4(rgb, u_colour.a*alpha);
}
"""


class RouteLine(GLLinePlotItem):
    """Constant-width, dark-outlined strokes using each original cleaned edge."""

    _program = None

    def __init__(self, *, outline=1.0, **kwargs):
        """Retain normal line-item controls while replacing native wide lines."""
        super().__init__(**kwargs)
        self.outline = outline
        self.updateGLOptions({"glDepthMask": (False,)})

    @classmethod
    def program(cls):
        """Compile once in the viewer's shared OpenGL context."""
        if cls._program is None:
            cls._program = shaders.compileProgram(
                shaders.compileShader(VERTEX, GL.GL_VERTEX_SHADER),
                shaders.compileShader(GEOMETRY, GL.GL_GEOMETRY_SHADER),
                shaders.compileShader(FRAGMENT, GL.GL_FRAGMENT_SHADER),
            )
        return cls._program

    def paint(self):
        """Expand edges on the GPU; width changes never upload new geometry."""
        if self.pos is None or len(self.pos) < 2:
            return
        self.setupGLState()
        if DirtyFlag.POSITION in self.dirty_bits:
            self.upload_vbo(self.m_vbo_position, self.pos)
        self.dirty_bits = DirtyFlag(0)
        program = self.program()
        location = GL.glGetAttribLocation(program, "a_position")
        self.m_vbo_position.bind()
        GL.glVertexAttribPointer(location, 3, GL.GL_FLOAT, False, 0, None)
        self.m_vbo_position.release()
        GL.glEnableVertexAttribArray(location)
        ratio = self.view().devicePixelRatioF()
        half_width, border, aa = (
            self.width * ratio / 2,
            self.outline * ratio,
            ratio * 0.75,
        )
        viewport = GL.glGetIntegerv(GL.GL_VIEWPORT)
        try:
            with program:
                GL.glUniformMatrix4fv(
                    GL.glGetUniformLocation(program, "u_mvp"),
                    1,
                    False,
                    np.asarray(self.mvpMatrix().data(), dtype=np.float32),
                )
                GL.glUniform2f(
                    GL.glGetUniformLocation(program, "u_viewport"),
                    float(viewport[2]),
                    float(viewport[3]),
                )
                GL.glUniform4f(
                    GL.glGetUniformLocation(program, "u_colour"), *self.color
                )
                for name, value in (
                    ("u_half_width", half_width),
                    ("u_outline", border),
                    ("u_aa", aa),
                    ("u_radius", half_width + border + aa),
                ):
                    GL.glUniform1f(GL.glGetUniformLocation(program, name), value)
                mode = GL.GL_LINE_STRIP if self.mode == "line_strip" else GL.GL_LINES
                # Outline the complete strip before its fill. Interleaving borders
                # and fills per tiny edge would cover neighbouring edge interiors.
                for fill in (0, 1):
                    GL.glUniform1i(GL.glGetUniformLocation(program, "u_fill"), fill)
                    GL.glDrawArrays(mode, 0, len(self.pos))
        finally:
            GL.glDisableVertexAttribArray(location)

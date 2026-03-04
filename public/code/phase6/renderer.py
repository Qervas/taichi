"""
Phase 6 — HDR rendering pipeline for SWE flood simulator.

Render flow:
  1. Sky / Terrain / Water  →  HDR FBO (RGBA16F)
  2. Bloom extract + separable Gaussian blur (half-res ping-pong)
  3. Final composite: HDR scene + bloom + rain + ACES tonemapping  →  screen
"""

from pathlib import Path
import numpy as np
import moderngl


SHADER_DIR = Path(__file__).parent / "shaders"


def _load_shader(name):
    return (SHADER_DIR / name).read_text()


# ─── Bloom shader (inline — bright extract + separable Gaussian blur) ───
BLOOM_FRAG = """#version 430
in vec2 v_uv;
out vec4 frag_color;

uniform sampler2D u_source;
uniform vec2 u_direction;
uniform float u_threshold;

void main() {
    vec3 color;

    if (u_threshold > 0.0) {
        // Bright extract mode
        color = texture(u_source, v_uv).rgb;
        float brightness = dot(color, vec3(0.2126, 0.7152, 0.0722));
        color *= smoothstep(u_threshold - 0.2, u_threshold + 0.5, brightness);
    } else {
        // 9-tap separable Gaussian blur
        float w[5] = float[](0.227027, 0.1945946, 0.1216216, 0.054054, 0.016216);
        color = texture(u_source, v_uv).rgb * w[0];
        for (int i = 1; i < 5; i++) {
            vec2 off = u_direction * float(i);
            color += texture(u_source, v_uv + off).rgb * w[i];
            color += texture(u_source, v_uv - off).rgb * w[i];
        }
    }

    frag_color = vec4(color, 1.0);
}
"""


def _build_grid_mesh(nx, ny):
    """Build a flat grid of (nx * ny) vertices with UVs and triangle indices."""
    u = np.linspace(0, 1, nx, dtype=np.float32)
    v = np.linspace(0, 1, ny, dtype=np.float32)
    U, V = np.meshgrid(u, v, indexing='ij')
    verts = np.stack([U.ravel(), V.ravel()], axis=-1)  # (nx*ny, 2)

    # Triangle indices: two triangles per quad
    idx = []
    for i in range(nx - 1):
        for j in range(ny - 1):
            v00 = i * ny + j
            v10 = (i + 1) * ny + j
            v01 = i * ny + (j + 1)
            v11 = (i + 1) * ny + (j + 1)
            idx.extend([v00, v10, v11, v00, v11, v01])
    indices = np.array(idx, dtype=np.uint32)
    return verts, indices


class FloodRenderer:
    """OpenGL 4.3 HDR renderer for shallow water flood simulation."""

    def __init__(self, ctx: moderngl.Context, nx: int, ny: int, dx: float,
                 z_bed_np: np.ndarray, window_size=(1920, 1080)):
        self.ctx = ctx
        self.nx = nx
        self.ny = ny
        self.dx = dx
        self.domain_x = nx * dx
        self.domain_y = ny * dx
        self.max_elevation = float(z_bed_np.max())

        # ─── Compile shader programs ───
        self.prog_sky = ctx.program(
            vertex_shader=_load_shader("sky.vert"),
            fragment_shader=_load_shader("sky.frag"),
        )
        self.prog_terrain = ctx.program(
            vertex_shader=self._terrain_vert_with_mvp(),
            fragment_shader=_load_shader("terrain.frag"),
        )
        self.prog_water = ctx.program(
            vertex_shader=self._water_vert_with_mvp(),
            fragment_shader=_load_shader("water.frag"),
        )

        # Post-processing programs
        post_vert = _load_shader("post.vert")
        self.prog_bloom = ctx.program(
            vertex_shader=post_vert,
            fragment_shader=BLOOM_FRAG,
        )
        self.prog_post = ctx.program(
            vertex_shader=post_vert,
            fragment_shader=_load_shader("post.frag"),
        )

        # ─── Build grid mesh VBO (uploaded once) ───
        verts, indices = _build_grid_mesh(nx, ny)
        self.n_indices = len(indices)

        vbo = ctx.buffer(verts.tobytes())
        ibo = ctx.buffer(indices.tobytes())

        self.vao_terrain = ctx.vertex_array(
            self.prog_terrain,
            [(vbo, '2f', 'a_uv')],
            index_buffer=ibo,
        )
        self.vao_water = ctx.vertex_array(
            self.prog_water,
            [(vbo, '2f', 'a_uv')],
            index_buffer=ibo,
        )

        # Fullscreen triangle VAOs (no VBO, uses gl_VertexID)
        self.vao_sky = ctx.vertex_array(self.prog_sky, [])
        self.vao_bloom = ctx.vertex_array(self.prog_bloom, [])
        self.vao_post = ctx.vertex_array(self.prog_post, [])

        # ─── Textures ───
        # z_bed (static, uploaded once)
        self.tex_z_bed = ctx.texture((nx, ny), 1, dtype='f4')
        self.tex_z_bed.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.tex_z_bed.write(z_bed_np.astype(np.float32).tobytes())

        # h (dynamic, updated each frame)
        self.tex_h = ctx.texture((nx, ny), 1, dtype='f4')
        self.tex_h.filter = (moderngl.LINEAR, moderngl.LINEAR)

        # hu, hv packed as RG (dynamic)
        self.tex_huv = ctx.texture((nx, ny), 2, dtype='f4')
        self.tex_huv.filter = (moderngl.LINEAR, moderngl.LINEAR)

        # ─── Bind static texture units ───
        self._bind_terrain_textures()
        self._bind_water_textures()

        # ─── HDR + bloom FBOs ───
        self.hdr_fbo = None  # initialized in _create_hdr_fbos
        self._create_hdr_fbos(window_size[0], window_size[1])

    # ─── HDR / Bloom FBO management ───

    def _create_hdr_fbos(self, w, h):
        """Create (or recreate) HDR and bloom framebuffers."""
        # Release old resources
        for attr in ('hdr_fbo', 'hdr_color', 'hdr_depth',
                     'bloom_fbo_a', 'bloom_fbo_b', 'bloom_a', 'bloom_b'):
            obj = getattr(self, attr, None)
            if obj is not None:
                obj.release()

        self.win_w, self.win_h = w, h
        self.bloom_w = max(w // 2, 1)
        self.bloom_h = max(h // 2, 1)

        # HDR scene FBO (RGBA16F + depth)
        self.hdr_color = self.ctx.texture((w, h), 4, dtype='f2')
        self.hdr_color.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.hdr_depth = self.ctx.depth_renderbuffer((w, h))
        self.hdr_fbo = self.ctx.framebuffer(
            color_attachments=[self.hdr_color],
            depth_attachment=self.hdr_depth,
        )

        # Bloom ping-pong textures (half-res)
        self.bloom_a = self.ctx.texture((self.bloom_w, self.bloom_h), 4, dtype='f2')
        self.bloom_a.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.bloom_b = self.ctx.texture((self.bloom_w, self.bloom_h), 4, dtype='f2')
        self.bloom_b.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.bloom_fbo_a = self.ctx.framebuffer(color_attachments=[self.bloom_a])
        self.bloom_fbo_b = self.ctx.framebuffer(color_attachments=[self.bloom_b])

    def resize(self, w, h):
        """Recreate HDR/bloom FBOs on window resize."""
        self._create_hdr_fbos(w, h)

    # ─── Shader injection helpers ───

    def _terrain_vert_with_mvp(self):
        """Inject MVP transform into terrain vertex shader."""
        src = _load_shader("terrain.vert")
        inject = """
uniform mat4 u_mvp;
"""
        main_append = """
    gl_Position = u_mvp * vec4(v_world_pos, 1.0);
"""
        src = src.replace("void main() {", inject + "\nvoid main() {")
        src = src.replace("// gl_Position set in renderer via uniform MVP",
                          main_append)
        return src

    def _water_vert_with_mvp(self):
        """Inject MVP transform into water vertex shader."""
        src = _load_shader("water.vert")
        inject = """
uniform mat4 u_mvp;
"""
        main_append = """
    gl_Position = u_mvp * vec4(v_world_pos, 1.0);
"""
        src = src.replace("void main() {", inject + "\nvoid main() {")
        src = src.replace("// gl_Position set by renderer via MVP uniform",
                          main_append)
        return src

    # ─── Static uniform bindings (called once) ───

    def _bind_terrain_textures(self):
        p = self.prog_terrain
        if 'u_z_bed' in p:
            p['u_z_bed'].value = 0
        if 'u_h_tex' in p:
            p['u_h_tex'].value = 1
        if 'u_domain_x' in p:
            p['u_domain_x'].value = self.domain_x
        if 'u_domain_y' in p:
            p['u_domain_y'].value = self.domain_y
        if 'u_max_elevation' in p:
            p['u_max_elevation'].value = self.max_elevation

    def _bind_water_textures(self):
        p = self.prog_water
        if 'u_h_tex' in p:
            p['u_h_tex'].value = 1
        if 'u_z_bed' in p:
            p['u_z_bed'].value = 0
        if 'u_huv_tex' in p:
            p['u_huv_tex'].value = 2
        if 'u_domain_x' in p:
            p['u_domain_x'].value = self.domain_x
        if 'u_domain_y' in p:
            p['u_domain_y'].value = self.domain_y
        if 'u_swell_gain' in p:
            p['u_swell_gain'].value = 1.0
        if 'u_chop_gain' in p:
            p['u_chop_gain'].value = 1.0
        if 'u_foam_gain' in p:
            p['u_foam_gain'].value = 1.0

    def update_z_bed(self, z_bed_np):
        """Re-upload bed elevation (e.g., on scene change)."""
        self.max_elevation = float(z_bed_np.max())
        self.tex_z_bed.write(z_bed_np.astype(np.float32).tobytes())
        if 'u_max_elevation' in self.prog_terrain:
            self.prog_terrain['u_max_elevation'].value = self.max_elevation

    # ─── Main render method ───

    def render(self, h_np, hu_np, hv_np, view_mat, proj_mat, cam_pos,
               sun_dir, sim_time, storm=0.0, rain=0.0, wall_time=0.0,
               swell_gain=1.0, chop_gain=1.0, foam_gain=1.0):
        """Render a full frame: scene → HDR FBO → bloom → composite to screen."""
        ctx = self.ctx

        # Upload dynamic textures
        self.tex_h.write(h_np.astype(np.float32).tobytes())
        huv = np.stack([hu_np, hv_np], axis=-1).astype(np.float32)
        self.tex_huv.write(huv.tobytes())

        mvp = (proj_mat @ view_mat).astype(np.float32)
        inv_vp = np.linalg.inv(mvp).astype(np.float32)

        # ═══════════════════════════════════════════════════
        # Render scene into HDR FBO
        # ═══════════════════════════════════════════════════
        self.hdr_fbo.use()
        ctx.viewport = (0, 0, self.win_w, self.win_h)
        ctx.clear(0.0, 0.0, 0.0)

        # ─── Pass 1: Sky (fullscreen, no depth) ───
        ctx.disable(moderngl.DEPTH_TEST)
        ctx.disable(moderngl.BLEND)

        p = self.prog_sky
        if 'u_inv_vp' in p:
            p['u_inv_vp'].write(inv_vp.T.tobytes())
        if 'u_sun_dir' in p:
            p['u_sun_dir'].value = tuple(sun_dir)
        if 'u_storm' in p:
            p['u_storm'].value = storm
        if 'u_swell_gain' in p:
            p['u_swell_gain'].value = swell_gain
        if 'u_chop_gain' in p:
            p['u_chop_gain'].value = chop_gain
        if 'u_foam_gain' in p:
            p['u_foam_gain'].value = foam_gain
        if 'u_time' in p:
            p['u_time'].value = wall_time

        self.vao_sky.render(moderngl.TRIANGLES, vertices=3)

        # ─── Pass 2: Terrain (depth, polygon offset) ───
        ctx.enable(moderngl.DEPTH_TEST)
        ctx.depth_func = '<'
        ctx.polygon_offset = (1.0, 1.0)

        self.tex_z_bed.use(location=0)
        self.tex_h.use(location=1)

        p = self.prog_terrain
        if 'u_mvp' in p:
            p['u_mvp'].write(mvp.T.tobytes())
        if 'u_sun_dir' in p:
            p['u_sun_dir'].value = tuple(sun_dir)
        if 'u_cam_pos' in p:
            p['u_cam_pos'].value = tuple(cam_pos)
        if 'u_storm' in p:
            p['u_storm'].value = storm

        self.vao_terrain.render()
        ctx.polygon_offset = (0.0, 0.0)

        # ─── Pass 3: Water (alpha blend, depth) ───
        ctx.enable(moderngl.BLEND)
        ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)

        self.tex_z_bed.use(location=0)
        self.tex_h.use(location=1)
        self.tex_huv.use(location=2)

        p = self.prog_water
        if 'u_mvp' in p:
            p['u_mvp'].write(mvp.T.tobytes())
        if 'u_sun_dir' in p:
            p['u_sun_dir'].value = tuple(sun_dir)
        if 'u_cam_pos' in p:
            p['u_cam_pos'].value = tuple(cam_pos)
        if 'u_time' in p:
            p['u_time'].value = sim_time
        if 'u_inv_vp' in p:
            p['u_inv_vp'].write(inv_vp.T.tobytes())
        if 'u_storm' in p:
            p['u_storm'].value = storm

        self.vao_water.render()
        ctx.disable(moderngl.BLEND)

        # ═══════════════════════════════════════════════════
        # Bloom (half-res ping-pong)
        # ═══════════════════════════════════════════════════
        ctx.disable(moderngl.DEPTH_TEST)

        p = self.prog_bloom
        if 'u_source' in p:
            p['u_source'].value = 0

        # Bright extract → bloom_a
        self.bloom_fbo_a.use()
        ctx.viewport = (0, 0, self.bloom_w, self.bloom_h)
        self.hdr_color.use(location=0)
        if 'u_threshold' in p:
            p['u_threshold'].value = 1.0
        if 'u_direction' in p:
            p['u_direction'].value = (0.0, 0.0)
        self.vao_bloom.render(moderngl.TRIANGLES, vertices=3)

        # Iteration 1: H-blur bloom_a → bloom_b
        self.bloom_fbo_b.use()
        self.bloom_a.use(location=0)
        if 'u_threshold' in p:
            p['u_threshold'].value = 0.0
        if 'u_direction' in p:
            p['u_direction'].value = (1.0 / self.bloom_w, 0.0)
        self.vao_bloom.render(moderngl.TRIANGLES, vertices=3)

        # Iteration 1: V-blur bloom_b → bloom_a
        self.bloom_fbo_a.use()
        self.bloom_b.use(location=0)
        if 'u_direction' in p:
            p['u_direction'].value = (0.0, 1.0 / self.bloom_h)
        self.vao_bloom.render(moderngl.TRIANGLES, vertices=3)

        # Iteration 2: H-blur bloom_a → bloom_b (wider bloom)
        self.bloom_fbo_b.use()
        self.bloom_a.use(location=0)
        if 'u_direction' in p:
            p['u_direction'].value = (1.0 / self.bloom_w, 0.0)
        self.vao_bloom.render(moderngl.TRIANGLES, vertices=3)

        # Iteration 2: V-blur bloom_b → bloom_a
        self.bloom_fbo_a.use()
        self.bloom_b.use(location=0)
        if 'u_direction' in p:
            p['u_direction'].value = (0.0, 1.0 / self.bloom_h)
        self.vao_bloom.render(moderngl.TRIANGLES, vertices=3)

        # ═══════════════════════════════════════════════════
        # Final composite to screen
        # ═══════════════════════════════════════════════════
        ctx.screen.use()
        ctx.viewport = (0, 0, self.win_w, self.win_h)

        self.hdr_color.use(location=0)
        self.bloom_a.use(location=1)

        p = self.prog_post
        if 'u_scene' in p:
            p['u_scene'].value = 0
        if 'u_bloom' in p:
            p['u_bloom'].value = 1
        if 'u_bloom_strength' in p:
            p['u_bloom_strength'].value = 0.3
        if 'u_exposure' in p:
            p['u_exposure'].value = 1.2
        if 'u_rain_intensity' in p:
            p['u_rain_intensity'].value = rain
        if 'u_time' in p:
            p['u_time'].value = wall_time
        if 'u_storm' in p:
            p['u_storm'].value = storm

        self.vao_post.render(moderngl.TRIANGLES, vertices=3)

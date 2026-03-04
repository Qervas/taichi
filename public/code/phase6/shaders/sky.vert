#version 430
// Fullscreen triangle — no VBO needed, use gl_VertexID
out vec2 v_uv;

void main() {
    // 3 vertices covering the whole screen
    vec2 pos = vec2((gl_VertexID & 1) * 2.0 - 1.0,
                    (gl_VertexID >> 1) * 2.0 - 1.0);
    // Expand to cover full screen with a single oversized triangle
    if (gl_VertexID == 2) pos = vec2(3.0, 1.0);
    if (gl_VertexID == 1) pos = vec2(-1.0, 3.0);
    if (gl_VertexID == 0) pos = vec2(-1.0, -1.0);

    v_uv = pos * 0.5 + 0.5;
    gl_Position = vec4(pos, 0.9999, 1.0);
}

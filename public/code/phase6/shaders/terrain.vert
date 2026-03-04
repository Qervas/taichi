#version 430
in vec2 a_uv;  // grid UV [0,1]^2

uniform sampler2D u_z_bed;
uniform sampler2D u_h_tex;   // water depth for wet surface detection
uniform float u_domain_x;    // NX * dx
uniform float u_domain_y;    // NY * dx

out vec3 v_world_pos;
out vec3 v_normal;
out vec2 v_uv;
out float v_water_depth;

void main() {
    float z_bed = texture(u_z_bed, a_uv).r;
    float water_h = texture(u_h_tex, a_uv).r;

    // World position: UV maps to XZ plane, z_bed is Y (up)
    v_world_pos = vec3(a_uv.x * u_domain_x, z_bed, a_uv.y * u_domain_y);
    v_uv = a_uv;
    v_water_depth = water_h;

    // Compute normal from z_bed texture gradients
    vec2 texel = 1.0 / textureSize(u_z_bed, 0);
    float hL = texture(u_z_bed, a_uv + vec2(-texel.x, 0.0)).r;
    float hR = texture(u_z_bed, a_uv + vec2( texel.x, 0.0)).r;
    float hD = texture(u_z_bed, a_uv + vec2(0.0, -texel.y)).r;
    float hU = texture(u_z_bed, a_uv + vec2(0.0,  texel.y)).r;

    float dx_world = texel.x * u_domain_x;
    float dy_world = texel.y * u_domain_y;
    v_normal = normalize(vec3(
        (hL - hR) / (2.0 * dx_world),
        1.0,
        (hD - hU) / (2.0 * dy_world)
    ));

    // gl_Position set in renderer via uniform MVP
}

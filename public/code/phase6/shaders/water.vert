#version 430
in vec2 a_uv;

uniform sampler2D u_h_tex;
uniform sampler2D u_z_bed;
uniform sampler2D u_huv_tex;
uniform float u_domain_x;
uniform float u_domain_y;
uniform float u_time;
uniform float u_storm;

out vec3 v_world_pos;
out vec2 v_uv;
out float v_depth;
out float v_turb;

// ─── Noise ───
float hash(vec2 p) {
    vec3 p3 = fract(vec3(p.xyx) * 0.1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}

float vnoise(vec2 p) {
    vec2 i = floor(p);
    vec2 f = fract(p);
    f = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);  // quintic
    float a = hash(i);
    float b = hash(i + vec2(1.0, 0.0));
    float c = hash(i + vec2(0.0, 1.0));
    float d = hash(i + vec2(1.0, 1.0));
    return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
}

const float GOLDEN_ANGLE = 2.39996322;

void main() {
    float h = texture(u_h_tex, a_uv).r;
    float z = texture(u_z_bed, a_uv).r;
    vec2 huv = texture(u_huv_tex, a_uv).rg;

    v_depth = h;
    v_uv = a_uv;

    float vel_mag = length(huv) / max(h, 0.01);
    float turb = smoothstep(0.3, 4.0, vel_mag);
    v_turb = turb * smoothstep(0.1, 0.5, h);

    vec3 pos = vec3(a_uv.x * u_domain_x, z + h, a_uv.y * u_domain_y);

    // ─── 4-octave Gerstner vertex displacement ───
    if (h > 0.15) {
        vec2 flow = huv / max(length(huv), 0.001);

        // Wind gust — patchy amplitude
        float wind = vnoise(pos.xz * 0.04 + u_time * 0.2) * 0.6 + 0.4;
        wind *= mix(0.7, 1.3, u_storm);

        // Domain warp — break spatial repetition
        vec2 warp = vec2(
            vnoise(pos.xz * 0.07 + 17.0),
            vnoise(pos.xz * 0.07 + 43.0)
        ) * 8.0 - 4.0;
        vec2 w = pos.xz + warp;

        float depth_scale = smoothstep(0.15, 2.0, h);
        // Even calm water moves — base energy from wind
        float energy = max(turb, mix(0.15, 0.35, u_storm));

        float dy = 0.0;
        for (int i = 0; i < 4; i++) {
            float fi = float(i);

            // Golden-angle direction + frequency-scaled noise
            float dir_scale = 0.025 * pow(1.8, fi);
            float angle = fi * GOLDEN_ANGLE
                        + vnoise(w * dir_scale + fi * 7.3) * 0.7;
            vec2 d = vec2(cos(angle), sin(angle));
            d = normalize(mix(d, flow, turb * 0.25));

            float freq = 1.5 * pow(2.0, fi);
            float amp = 0.12 / pow(1.9, fi) * depth_scale;
            float speed = 1.0 + fi * 0.4;
            float travel = (i % 2 == 0) ? 1.0 : -1.0;

            // Continuous phase modulation (no cell seams)
            float pm = (vnoise(w * freq * 0.15 + fi * 3.17) * 2.0 - 1.0) * 1.5;
            float phase = dot(d, w) * freq + u_time * speed * travel + pm;
            dy += amp * cos(phase) * energy * wind;
        }

        pos.y += dy;
    }

    v_world_pos = pos;

    // gl_Position set by renderer via MVP uniform
}

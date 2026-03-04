#version 430
in vec2 v_uv;
out vec4 frag_color;

uniform mat4 u_inv_vp;
uniform vec3 u_sun_dir;
uniform float u_storm;  // 0 = golden hour, 1 = full storm
uniform float u_time;   // wall-clock time for cloud animation

// ─── Hash / noise for procedural clouds ───
float hash(vec2 p) {
    vec3 p3 = fract(vec3(p.xyx) * 0.1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}

float value_noise(vec2 p) {
    vec2 i = floor(p);
    vec2 f = fract(p);
    f = f * f * (3.0 - 2.0 * f);
    float a = hash(i);
    float b = hash(i + vec2(1.0, 0.0));
    float c = hash(i + vec2(0.0, 1.0));
    float d = hash(i + vec2(1.0, 1.0));
    return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
}

float fbm(vec2 p) {
    float v = 0.0;
    float amp = 0.5;
    for (int i = 0; i < 5; i++) {
        v += amp * value_noise(p);
        p *= 2.0;
        amp *= 0.5;
    }
    return v;
}

// Procedural sky — reused by water shader for reflections
vec3 sky_color(vec3 ray) {
    vec3 rd = normalize(ray);
    float t = rd.y * 0.5 + 0.5;

    // Golden-hour tones
    float sun_elev = u_sun_dir.y;
    float golden = smoothstep(0.1, 0.5, 1.0 - sun_elev);

    // Clear sky palettes
    vec3 zenith_clear  = mix(vec3(0.15, 0.30, 0.65), vec3(0.20, 0.20, 0.45), golden);
    vec3 horizon_clear = mix(vec3(0.60, 0.75, 0.90), vec3(0.85, 0.55, 0.30), golden);

    // Storm palettes — dark, oppressive
    vec3 zenith_storm  = vec3(0.08, 0.08, 0.12);
    vec3 horizon_storm = vec3(0.25, 0.25, 0.30);

    vec3 zenith  = mix(zenith_clear, zenith_storm, u_storm);
    vec3 horizon = mix(horizon_clear, horizon_storm, u_storm);
    vec3 col = mix(horizon, zenith, clamp(t, 0.0, 1.0));

    // Warm glow band near horizon (dimmed in storm)
    float horizon_band = exp(-abs(rd.y) * 8.0);
    vec3 glow_tint = mix(vec3(0.15, 0.08, 0.02), vec3(0.40, 0.20, 0.05), golden);
    col += glow_tint * horizon_band * mix(1.0, 0.2, u_storm);

    // Sun attenuation — barely peeks through in storm
    float sun_atten = mix(1.0, 0.1, u_storm);

    // Sun disc
    float cos_angle = dot(rd, u_sun_dir);
    float sun_disc = smoothstep(0.9995, 0.9999, cos_angle);
    col += vec3(10.0, 9.0, 7.0) * sun_disc * sun_atten;

    // Mie-like halo
    float halo = pow(max(cos_angle, 0.0), 256.0);
    col += vec3(1.0, 0.8, 0.5) * halo * 3.0 * sun_atten;

    // Broader sun glow
    float glow = pow(max(cos_angle, 0.0), 8.0);
    col += vec3(0.4, 0.25, 0.10) * glow * sun_atten;

    // Procedural clouds — animated, denser/darker in storm
    if (rd.y > 0.01) {
        vec2 cloud_uv = rd.xz / (rd.y + 0.1) * 3.0;
        cloud_uv += u_time * 0.02;  // animate clouds

        float cloud = fbm(cloud_uv * 1.5 + vec2(0.3, 0.7));

        // Storm: lower threshold → more coverage, higher opacity
        float cloud_thresh = mix(0.35, 0.15, u_storm);
        float cloud_top    = mix(0.65, 0.45, u_storm);
        cloud = smoothstep(cloud_thresh, cloud_top, cloud);

        float cloud_opacity = mix(0.6, 0.9, u_storm);

        // Cloud brightness — darker in storm
        float cloud_bright = mix(0.95, 1.2, max(dot(u_sun_dir, vec3(0, 1, 0)), 0.0));
        cloud_bright *= mix(1.0, 0.4, u_storm);
        vec3 cloud_col = vec3(cloud_bright) * mix(vec3(1.0), vec3(1.0, 0.85, 0.7), golden);

        // Storm clouds are gray-dark
        cloud_col = mix(cloud_col, vec3(0.2, 0.2, 0.22), u_storm * 0.7);

        float fade = smoothstep(0.01, 0.15, rd.y);
        col = mix(col, cloud_col, cloud * cloud_opacity * fade);
    }

    return col;
}

void main() {
    // Reconstruct world-space ray from screen UV
    vec2 ndc = v_uv * 2.0 - 1.0;
    vec4 clip_near = vec4(ndc, -1.0, 1.0);
    vec4 clip_far  = vec4(ndc,  1.0, 1.0);

    vec4 world_near = u_inv_vp * clip_near;
    vec4 world_far  = u_inv_vp * clip_far;
    world_near /= world_near.w;
    world_far  /= world_far.w;

    vec3 ray = normalize(world_far.xyz - world_near.xyz);
    vec3 col = sky_color(ray);

    // Output HDR — no tonemapping (handled by post-process)
    frag_color = vec4(col, 1.0);
}

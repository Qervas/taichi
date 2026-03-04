#version 430
in vec3 v_world_pos;
in vec2 v_uv;
in float v_depth;
in float v_turb;

uniform sampler2D u_h_tex;
uniform sampler2D u_huv_tex;
uniform sampler2D u_z_bed;
uniform vec3 u_sun_dir;
uniform vec3 u_cam_pos;
uniform float u_time;
uniform float u_domain_x;
uniform float u_domain_y;
uniform float u_storm;
uniform mat4 u_inv_vp;

out vec4 frag_color;

// ─── Constants ───
const float PI = 3.14159265;
const float GOLDEN_ANGLE = 2.39996322;
const vec3 WATER_F0 = vec3(0.02);
const vec3 ABSORPTION = vec3(4.0, 3.0, 1.5);

// Rotated octave matrix: scale ~2x + 37° rotation — breaks axis alignment
const mat2 OCTAVE_M = mat2(1.6, 1.2, -1.2, 1.6);

// ─── Noise ───
float hash(vec2 p) {
    vec3 p3 = fract(vec3(p.xyx) * 0.1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}

float value_noise(vec2 p) {
    vec2 i = floor(p);
    vec2 f = fract(p);
    f = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);  // quintic
    float a = hash(i);
    float b = hash(i + vec2(1.0, 0.0));
    float c = hash(i + vec2(0.0, 1.0));
    float d = hash(i + vec2(1.0, 1.0));
    return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
}

// Analytical gradient noise — returns vec3(dfdx, value, dfdz)
vec3 grad_vnoise(vec2 p) {
    vec2 i = floor(p);
    vec2 f = fract(p);
    vec2 u  = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);
    vec2 du = 30.0 * f * f * (f - 1.0) * (f - 1.0);
    float a = hash(i);
    float b = hash(i + vec2(1.0, 0.0));
    float c = hash(i + vec2(0.0, 1.0));
    float d = hash(i + vec2(1.0, 1.0));
    float val = mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
    float dx  = du.x * mix(b - a, d - c, u.y);
    float dz  = du.y * mix(c - a, d - b, u.x);
    return vec3(dx, val, dz);
}

float fbm(vec2 p) {
    float v = 0.0, amp = 0.5;
    for (int i = 0; i < 5; i++) {
        v += amp * value_noise(p);
        p *= 2.0;
        amp *= 0.5;
    }
    return v;
}

// ─── Sky color (matches sky.frag) ───
vec3 sky_color(vec3 ray) {
    vec3 rd = normalize(ray);
    float t = rd.y * 0.5 + 0.5;

    float sun_elev = u_sun_dir.y;
    float golden = smoothstep(0.1, 0.5, 1.0 - sun_elev);

    vec3 zenith_clear  = mix(vec3(0.15, 0.30, 0.65), vec3(0.20, 0.20, 0.45), golden);
    vec3 horizon_clear = mix(vec3(0.60, 0.75, 0.90), vec3(0.85, 0.55, 0.30), golden);
    vec3 zenith_storm  = vec3(0.08, 0.08, 0.12);
    vec3 horizon_storm = vec3(0.25, 0.25, 0.30);

    vec3 zenith  = mix(zenith_clear,  zenith_storm,  u_storm);
    vec3 horizon = mix(horizon_clear, horizon_storm, u_storm);
    vec3 col = mix(horizon, zenith, clamp(t, 0.0, 1.0));

    float horizon_band = exp(-abs(rd.y) * 8.0);
    vec3 glow_tint = mix(vec3(0.15, 0.08, 0.02), vec3(0.40, 0.20, 0.05), golden);
    col += glow_tint * horizon_band * mix(1.0, 0.2, u_storm);

    float sun_atten = mix(1.0, 0.1, u_storm);
    float cos_angle = dot(rd, u_sun_dir);
    col += vec3(10.0, 9.0, 7.0) * smoothstep(0.9995, 0.9999, cos_angle) * sun_atten;
    col += vec3(1.0, 0.8, 0.5) * pow(max(cos_angle, 0.0), 256.0) * 3.0 * sun_atten;
    col += vec3(0.4, 0.25, 0.10) * pow(max(cos_angle, 0.0), 8.0) * sun_atten;

    if (rd.y > 0.01) {
        vec2 cloud_uv = rd.xz / (rd.y + 0.1) * 3.0 + u_time * 0.02;
        float cloud = fbm(cloud_uv * 1.5 + vec2(0.3, 0.7));
        float cloud_thresh = mix(0.35, 0.15, u_storm);
        cloud = smoothstep(cloud_thresh, mix(0.65, 0.45, u_storm), cloud);
        float cloud_bright = mix(0.95, 1.2, max(u_sun_dir.y, 0.0)) * mix(1.0, 0.4, u_storm);
        vec3 cloud_col = vec3(cloud_bright) * mix(vec3(1.0), vec3(1.0, 0.85, 0.7), golden);
        cloud_col = mix(cloud_col, vec3(0.2, 0.2, 0.22), u_storm * 0.7);
        col = mix(col, cloud_col, cloud * mix(0.6, 0.9, u_storm) * smoothstep(0.01, 0.15, rd.y));
    }

    return col;
}

// ═══════════════════════════════════════════════════════════════
// GERSTNER DERIVATIVES — 6 octaves, golden-angle, domain-warped
// Returns vec2 slope derivative for normal reconstruction
// ═══════════════════════════════════════════════════════════════

vec2 gerstner_deriv(vec2 pos, float time, vec2 flow, float turb) {
    vec2 deriv = vec2(0.0);

    // Wind gust — patchy amplitude
    float wind = value_noise(pos * 0.04 + time * 0.2) * 0.6 + 0.4;
    wind *= mix(0.7, 1.3, u_storm);

    // Domain warp — break spatial repetition
    vec2 warp = vec2(
        value_noise(pos * 0.07 + 17.0),
        value_noise(pos * 0.07 + 43.0)
    ) * 8.0 - 4.0;
    vec2 w = pos + warp;

    // Energy: turbulent water = strong waves, calm water = gentle wind waves
    float energy = max(0.6 + 0.4 * turb, mix(0.25, 0.5, u_storm));

    for (int i = 0; i < 6; i++) {
        float fi = float(i);

        // Golden-angle direction + frequency-scaled noise perturbation
        float dir_scale = 0.02 * pow(1.8, fi);
        float angle = fi * GOLDEN_ANGLE
                    + value_noise(w * dir_scale + fi * 7.3) * 0.8;
        vec2 d = vec2(cos(angle), sin(angle));
        // Steer toward flow in turbulent water, free direction in calm
        d = normalize(mix(d, flow, turb * 0.3));

        float freq  = 2.0 * pow(1.8, fi);
        float amp   = 0.06 / pow(1.7, fi);
        float speed = 0.8 + fi * 0.3;
        float travel = (i % 2 == 0) ? 1.0 : -1.0;  // alternating travel

        // Continuous phase modulation (no cell seams)
        float pm = (value_noise(w * freq * 0.12 + fi * 3.17) * 2.0 - 1.0) * 1.8;
        float phase = dot(d, w) * freq + time * speed * travel + pm;

        deriv += d * amp * freq * cos(phase) * energy * wind;

        // Second-order domain warp at octave 3
        if (i == 3) w += deriv * 0.4;
    }

    return deriv;
}

// ═══════════════════════════════════════════════════════════════
// CAPILLARY WAVES — gradient-noise FBM, rotated octave matrix
// Sub-meter wind ripples with analytical derivatives
// ═══════════════════════════════════════════════════════════════

vec2 capillary_deriv(vec2 pos, float time, float wind) {
    vec2 deriv = vec2(0.0);
    float amp = 0.5;

    // Primary layer
    vec2 p = pos * 35.0 + time * vec2(2.0, 1.5);
    for (int i = 0; i < 4; i++) {
        vec3 gn = grad_vnoise(p);
        deriv += vec2(gn.x, gn.z) * amp;
        p = OCTAVE_M * p + vec2(1.7, 9.2);
        amp *= 0.5;
    }

    // Counter-direction layer for cross-chop
    amp = 0.25;
    p = pos * 50.0 - time * vec2(1.3, 1.8) + vec2(37.0, 13.0);
    for (int i = 0; i < 3; i++) {
        vec3 gn = grad_vnoise(p);
        deriv += vec2(gn.x, gn.z) * amp;
        p = OCTAVE_M * p + vec2(3.1, 5.7);
        amp *= 0.5;
    }

    return deriv * wind * 0.10;
}

// ─── GGX BRDF ───
float D_GGX(float NdotH, float roughness) {
    float a = roughness * roughness;
    float a2 = a * a;
    float denom = NdotH * NdotH * (a2 - 1.0) + 1.0;
    return a2 / (PI * denom * denom + 1e-7);
}

float G_Smith(float NdotV, float NdotL, float roughness) {
    float k = (roughness + 1.0) * (roughness + 1.0) / 8.0;
    float g1 = NdotV / (NdotV * (1.0 - k) + k);
    float g2 = NdotL / (NdotL * (1.0 - k) + k);
    return g1 * g2;
}

vec3 F_Schlick(float cos_theta, vec3 F0) {
    return F0 + (1.0 - F0) * pow(1.0 - cos_theta, 5.0);
}

// ─── Foam texture ───
float foam_noise(vec2 uv, float time) {
    float n = 0.0, amp = 0.5;
    vec2 p = uv;
    for (int i = 0; i < 4; i++) {
        n += amp * value_noise(p + time * 0.3);
        p *= 2.3;
        amp *= 0.45;
    }
    return n;
}

// ═══════════════════════════════════════════════════════════════

void main() {
    float h = v_depth;
    if (h < 1e-4) discard;

    // ─── Velocity ───
    vec2 huv = texture(u_huv_tex, v_uv).rg;
    vec2 vel = huv / max(h, 0.01);
    float vel_mag = length(vel);
    vec2 flow_dir = vel / max(vel_mag, 0.001);
    float turb = v_turb;

    // ─── Macro normal from SWE height field (smoothed) ───
    vec2 texel = 1.0 / textureSize(u_h_tex, 0);
    float dx_w = texel.x * u_domain_x;
    float dy_w = texel.y * u_domain_y;

    // 1-texel central difference
    float etaL = texture(u_h_tex, v_uv + vec2(-texel.x, 0)).r
               + texture(u_z_bed, v_uv + vec2(-texel.x, 0)).r;
    float etaR = texture(u_h_tex, v_uv + vec2( texel.x, 0)).r
               + texture(u_z_bed, v_uv + vec2( texel.x, 0)).r;
    float etaD = texture(u_h_tex, v_uv + vec2(0, -texel.y)).r
               + texture(u_z_bed, v_uv + vec2(0, -texel.y)).r;
    float etaU = texture(u_h_tex, v_uv + vec2(0,  texel.y)).r
               + texture(u_z_bed, v_uv + vec2(0,  texel.y)).r;
    vec3 N1 = normalize(vec3((etaL - etaR) / (2.0 * dx_w), 1.0,
                             (etaD - etaU) / (2.0 * dy_w)));

    // 2-texel baseline to smooth SWE grid faceting
    float etaL2 = texture(u_h_tex, v_uv + vec2(-2.0*texel.x, 0)).r
                + texture(u_z_bed, v_uv + vec2(-2.0*texel.x, 0)).r;
    float etaR2 = texture(u_h_tex, v_uv + vec2( 2.0*texel.x, 0)).r
                + texture(u_z_bed, v_uv + vec2( 2.0*texel.x, 0)).r;
    float etaD2 = texture(u_h_tex, v_uv + vec2(0, -2.0*texel.y)).r
                + texture(u_z_bed, v_uv + vec2(0, -2.0*texel.y)).r;
    float etaU2 = texture(u_h_tex, v_uv + vec2(0,  2.0*texel.y)).r
                + texture(u_z_bed, v_uv + vec2(0,  2.0*texel.y)).r;
    vec3 N2 = normalize(vec3((etaL2 - etaR2) / (4.0 * dx_w), 1.0,
                             (etaD2 - etaU2) / (4.0 * dy_w)));
    vec3 macro_N = normalize(mix(N1, N2, 0.4));

    // ─── Gerstner micro derivatives ───
    vec2 gerstner_d = gerstner_deriv(v_world_pos.xz, u_time, flow_dir, turb);

    // ─── Capillary derivatives ───
    float wind_cap = value_noise(v_world_pos.xz * 0.04 + u_time * 0.3) * 0.6 + 0.4;
    wind_cap *= mix(0.3, 1.0, u_storm);
    vec2 cap_d = capillary_deriv(v_world_pos.xz, u_time, wind_cap);

    // ─── Derivative-space normal blend ───
    // All sources produce slope derivatives; sum is physically correct
    vec2 d_macro = macro_N.xz / max(macro_N.y, 0.01);
    vec2 d_total = d_macro + gerstner_d + cap_d;
    vec3 N = normalize(vec3(-d_total.x, 1.0, -d_total.y));

    // ─── View and light ───
    vec3 V = normalize(u_cam_pos - v_world_pos);
    vec3 L = normalize(u_sun_dir);
    vec3 H_vec = normalize(V + L);

    float NdotV = max(dot(N, V), 0.001);
    float NdotL = max(dot(N, L), 0.0);
    float NdotH = max(dot(N, H_vec), 0.0);
    float VdotH = max(dot(V, H_vec), 0.0);

    // ─── Roughness (velocity + wind modulated) ───
    float roughness = mix(0.08, 0.40, turb);
    roughness += value_noise(v_world_pos.xz * 0.04 + u_time * 0.3) * 0.08 * u_storm;

    // Specular anti-aliasing via screen-space normal variance
    float nvar = clamp(dot(dFdx(N), dFdx(N)) + dot(dFdy(N), dFdy(N)), 0.0, 1.0);
    roughness = sqrt(roughness * roughness + nvar * 0.4);
    roughness = clamp(roughness, 0.06, 0.65);

    // ─── Fresnel ───
    vec3 F = F_Schlick(NdotV, WATER_F0);

    // ─── Sky reflection ───
    vec3 R = reflect(-V, N);
    vec3 reflected = sky_color(R);

    // ─── Beer's law absorption ───
    vec3 transmittance = exp(-ABSORPTION * h);

    // ─── Refracted color (bed through water column) ───
    float z_bed = texture(u_z_bed, v_uv).r;
    vec3 bed_color = mix(vec3(0.70, 0.65, 0.45), vec3(0.40, 0.35, 0.22),
                         clamp(z_bed / 5.0, 0.0, 1.0));
    // Sediment entrainment — fast water is milkier
    bed_color = mix(bed_color, vec3(0.55, 0.48, 0.35), turb * 0.4);
    vec3 refracted = bed_color * transmittance;

    // ─── Subsurface scattering (back-lit glow) ───
    float sss = pow(max(dot(-L, V), 0.0), 4.0) * exp(-h * 2.5) * 0.3;
    vec3 sss_color = vec3(0.04, 0.25, 0.15) * sss;

    // ─── Deep scatter ───
    vec3 deep_scatter = vec3(0.03, 0.08, 0.06) * (1.0 - transmittance) * (0.3 + 0.7 * NdotL);

    // ─── GGX specular ───
    float D = D_GGX(NdotH, roughness);
    float G = G_Smith(NdotV, NdotL, roughness);
    vec3 Fs = F_Schlick(VdotH, WATER_F0);
    vec3 spec = (D * G * Fs) / (4.0 * NdotV * NdotL + 1e-4);
    float sun_atten = mix(1.0, 0.15, u_storm);
    spec *= NdotL * vec3(1.0, 0.95, 0.85) * 5.0 * sun_atten;

    // ─── Foam ───
    // Bed gradient for obstacle detection
    float zL = texture(u_z_bed, v_uv + vec2(-2.0*texel.x, 0)).r;
    float zR = texture(u_z_bed, v_uv + vec2( 2.0*texel.x, 0)).r;
    float zD = texture(u_z_bed, v_uv + vec2(0, -2.0*texel.y)).r;
    float zU = texture(u_z_bed, v_uv + vec2(0,  2.0*texel.y)).r;
    float bed_grad = max(abs(zR - zL), abs(zU - zD)) / (4.0 * dx_w);

    float vel_foam      = smoothstep(2.0, 5.0, vel_mag);
    float obstacle_foam = smoothstep(0.3, 1.0, bed_grad) * smoothstep(0.05, 0.4, h);
    float wake_foam     = smoothstep(0.5, 2.0, vel_mag) * smoothstep(0.15, 0.6, bed_grad) * 0.5;
    float edge_foam     = smoothstep(0.08, 0.01, h) * 0.4;

    float foam = clamp(vel_foam + obstacle_foam + wake_foam + edge_foam, 0.0, 1.0);
    // Bubbly texture — not flat white
    foam *= mix(0.65, 1.0, foam_noise(v_world_pos.xz * 3.0, u_time));

    vec3 foam_col = vec3(0.92, 0.94, 0.96);

    // ─── Compose ───
    vec3 water_body = refracted + deep_scatter + sss_color;
    vec3 color = mix(water_body, reflected, F) + spec;
    color = mix(color, foam_col, foam * 0.9);

    // ─── Mist near turbulent water ───
    float dist = length(v_world_pos - u_cam_pos);
    float mist = turb * smoothstep(200.0, 20.0, dist) * 0.12;
    vec3 mist_col = mix(vec3(0.6, 0.65, 0.7), vec3(0.3, 0.3, 0.35), u_storm);
    color = mix(color, mist_col, mist);

    // ─── Distance fog ───
    float fog = 1.0 - exp(-dist * mix(0.003, 0.006, u_storm));
    vec3 fog_col = mix(vec3(0.60, 0.70, 0.80), vec3(0.30, 0.30, 0.35), u_storm);
    color = mix(color, fog_col, clamp(fog, 0.0, 0.7));

    float alpha = smoothstep(0.0, 0.01, h);
    frag_color = vec4(color, alpha);
}

#version 430
in vec3 v_world_pos;
in vec3 v_normal;
in vec2 v_uv;
in float v_water_depth;

uniform vec3 u_sun_dir;
uniform vec3 u_cam_pos;
uniform float u_max_elevation;
uniform float u_storm;

out vec4 frag_color;

// Hash for building material variety
float hash_f(vec2 p) {
    vec3 p3 = fract(vec3(p.xyx) * 0.1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}

void main() {
    vec3 N = normalize(v_normal);
    float h = v_world_pos.y;
    float h_frac = clamp(h / max(u_max_elevation, 0.01), 0.0, 1.0);

    // View / light vectors
    vec3 V = normalize(u_cam_pos - v_world_pos);
    vec3 L = normalize(u_sun_dir);
    vec3 H_vec = normalize(V + L);

    // ─── Height-based coloring ───
    vec3 sandy = vec3(0.76, 0.70, 0.50);
    vec3 green = vec3(0.30, 0.50, 0.20);
    vec3 brown = vec3(0.45, 0.32, 0.20);
    vec3 rocky = vec3(0.50, 0.48, 0.45);

    vec3 base_col;
    if (h_frac < 0.15) {
        base_col = mix(sandy, green, h_frac / 0.15);
    } else if (h_frac < 0.5) {
        base_col = mix(green, brown, (h_frac - 0.15) / 0.35);
    } else {
        base_col = mix(brown, rocky, (h_frac - 0.5) / 0.5);
    }

    float specular = 0.0;
    float flatness = abs(N.y);

    // ─── Window patterns on vertical building faces ───
    if (h_frac > 0.6 && flatness < 0.3) {
        // Wall face — tile 3m x 3.5m windows
        vec2 wall_uv = vec2(
            fract(v_world_pos.x / 3.0 + v_world_pos.z / 3.0),
            fract(v_world_pos.y / 3.5)
        );

        bool is_window = wall_uv.x > 0.15 && wall_uv.x < 0.85 &&
                         wall_uv.y > 0.2  && wall_uv.y < 0.8;

        if (is_window) {
            base_col = vec3(0.08, 0.10, 0.15);  // dark glass
            float NdotH = max(dot(N, H_vec), 0.0);
            specular = pow(NdotH, 128.0) * 3.0;
        } else {
            // Wall frame material — varies per building
            vec2 bid = floor(v_world_pos.xz / 12.0);
            float mh = hash_f(bid);
            if (mh < 0.4)       base_col = vec3(0.50, 0.48, 0.45);  // concrete frame
            else if (mh < 0.7)  base_col = vec3(0.50, 0.32, 0.22);  // brick
            else                 base_col = vec3(0.35, 0.38, 0.42);  // steel frame
        }
    }
    // ─── Building rooftop: material variety ───
    else if (h_frac > 0.6 && flatness > 0.95) {
        vec2 building_id = floor(v_world_pos.xz / 12.0);
        float mat_hash = hash_f(building_id);

        if (mat_hash < 0.4) {
            base_col = vec3(0.55, 0.53, 0.50);  // concrete
        } else if (mat_hash < 0.7) {
            base_col = vec3(0.55, 0.35, 0.25);  // brick
        } else {
            base_col = vec3(0.25, 0.28, 0.35);  // glass/steel
            float NdotH = max(dot(N, H_vec), 0.0);
            specular = pow(NdotH, 64.0) * 2.0;
        }
    }

    // ─── Diffuse + ambient lighting ───
    float NdotL = max(dot(N, L), 0.0);
    float sun_strength = mix(0.75, 0.35, u_storm);
    vec3 ambient = mix(vec3(0.25, 0.28, 0.35), vec3(0.15, 0.15, 0.18), u_storm);
    vec3 sun_col = mix(vec3(1.0, 0.95, 0.85), vec3(0.6, 0.6, 0.6), u_storm);
    vec3 lit = base_col * (ambient + sun_col * NdotL * sun_strength) + vec3(specular) * sun_col;

    // ─── Wet surfaces (ground only — walls get garbage interpolated depth) ───
    float is_ground = smoothstep(0.3, 0.8, flatness);
    float wet = step(0.01, v_water_depth) * is_ground;

    if (wet > 0.01) {
        // Subtle darkening for wet ground
        lit *= mix(1.0, 0.7, wet);

        // Gentle wet specular sheen
        float NdotH = max(dot(N, H_vec), 0.0);
        float wet_spec = pow(NdotH, 32.0) * 0.3 * wet;
        lit += vec3(wet_spec) * sun_col * NdotL;
    }

    // ─── Water stain line (ground only) ───
    float stain_band = smoothstep(0.0, 0.05, v_water_depth) * smoothstep(0.3, 0.05, v_water_depth);
    stain_band *= is_ground;
    vec3 stain_color = vec3(0.35, 0.28, 0.18);  // muddy brown
    lit = mix(lit, stain_color, stain_band * 0.4);

    // ─── Distance fog (denser in storm) ───
    float dist = length(v_world_pos - u_cam_pos);
    float fog_density = mix(0.003, 0.006, u_storm);
    float fog = 1.0 - exp(-dist * fog_density);
    vec3 fog_col = mix(vec3(0.60, 0.75, 0.90), vec3(0.30, 0.30, 0.35), u_storm);
    lit = mix(lit, fog_col, clamp(fog, 0.0, 0.7));

    // Output HDR — no tonemapping (handled by post-process)
    frag_color = vec4(lit, 1.0);
}

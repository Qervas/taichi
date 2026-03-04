#version 430
in vec2 v_uv;
out vec4 frag_color;

uniform sampler2D u_scene;       // HDR scene texture
uniform sampler2D u_bloom;       // Blurred bloom texture
uniform float u_bloom_strength;  // Bloom intensity
uniform float u_exposure;        // Exposure control
uniform float u_rain_intensity;  // Rain amount [0,1]
uniform float u_time;            // Wall-clock time for animation
uniform float u_storm;           // Storm intensity [0,1]

// ─── ACES filmic tonemapping ───
vec3 aces_tonemap(vec3 x) {
    float a = 2.51, b = 0.03, c = 2.43, d = 0.59, e = 0.14;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0);
}

// ─── Hash for rain ───
float hash(vec2 p) {
    vec3 p3 = fract(vec3(p.xyx) * 0.1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}

// ─── Value noise for rain wind gusts ───
float vnoise(vec2 p) {
    vec2 i = floor(p);
    vec2 f = fract(p);
    f = f * f * (3.0 - 2.0 * f);
    float a = hash(i), b = hash(i + vec2(1,0));
    float c = hash(i + vec2(0,1)), d = hash(i + vec2(1,1));
    return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
}

// ─── Screen-space rain streaks (natural, wind-driven) ───
float rain(vec2 uv, float time) {
    float rain_acc = 0.0;

    for (int layer = 0; layer < 3; layer++) {
        float fl = float(layer);
        float scale_x = 60.0 + fl * 30.0;
        float scale_y = 3.5 + fl * 0.8;
        float speed   = 12.0 + fl * 4.0;
        float alpha   = 0.08 - fl * 0.015;

        // Wind angle varies per layer AND spatially via noise
        float base_angle = 0.08 + fl * 0.03;
        float wind_gust = vnoise(uv * vec2(2.0, 1.0) + time * 0.15 + fl * 11.0);
        float angle = base_angle + (wind_gust - 0.5) * 0.12;

        vec2 rain_uv = uv * vec2(scale_x, scale_y);
        rain_uv.y -= time * speed;
        rain_uv.x += rain_uv.y * angle;

        // Per-streak random horizontal offset to break regularity
        vec2 cell = floor(rain_uv);
        vec2 f = fract(rain_uv);

        float h = hash(cell + fl * 73.0);
        float streak_offset = hash(cell * 3.7 + fl * 19.0) * 0.3 - 0.15;

        // Sparser: fewer streaks, more natural
        if (h < 0.08) {
            float center = 0.5 + streak_offset;
            float streak = exp(-pow((f.x - center) * 12.0, 2.0));
            // Variable length per drop
            float drop_len = 0.3 + hash(cell * 5.3 + fl * 41.0) * 0.5;
            streak *= smoothstep(0.0, 0.08, f.y) * smoothstep(drop_len, max(drop_len - 0.3, 0.0), f.y);
            // Spatial density variation — wind gusts make rain patchy
            float density = vnoise(uv * 3.0 + time * 0.08 + fl * 7.0);
            density = smoothstep(0.25, 0.55, density);
            rain_acc += streak * alpha * density;
        }
    }

    return rain_acc;
}

void main() {
    vec3 scene = texture(u_scene, v_uv).rgb;
    vec3 bloom = texture(u_bloom, v_uv).rgb;

    // Add bloom
    vec3 color = scene + bloom * u_bloom_strength;

    // Rain streaks
    if (u_rain_intensity > 0.01) {
        float r = rain(v_uv, u_time);
        // Rain color: cool blue in storm, warmer in clear weather
        vec3 rain_color = mix(vec3(0.6, 0.55, 0.45), vec3(0.4, 0.45, 0.5), u_storm);
        color += rain_color * r * u_rain_intensity;
    }

    // Exposure
    color *= u_exposure;

    // ACES tonemapping
    color = aces_tonemap(color);

    // Gamma correction
    color = pow(color, vec3(1.0 / 2.2));

    frag_color = vec4(color, 1.0);
}

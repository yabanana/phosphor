#version 460
#extension GL_EXT_ray_tracing : require

// ---------------------------------------------------------------------------
// DDGI Probe Ray Miss Shader
//
// Returns a simple sky gradient (blue to white from horizon to zenith) and
// marks the hit distance as negative to indicate a miss.
// ---------------------------------------------------------------------------

layout(location = 0) rayPayloadInEXT vec4 payload; // xyz = radiance, w = hitDistance

void main() {
    // Reconstruct the ray direction from the built-in
    vec3 dir = normalize(gl_WorldRayDirectionEXT);

    // Simple analytic sky: lerp between horizon colour and zenith colour
    // based on the vertical component of the ray direction.
    vec3 horizonColor = vec3(0.8, 0.85, 0.95);   // pale blue-white
    vec3 zenithColor  = vec3(0.3, 0.5,  0.9);     // deeper blue

    float t = max(dir.y, 0.0);   // 0 at horizon, 1 at zenith
    vec3 skyColor = mix(horizonColor, zenithColor, t);

    // Ground: darken below horizon to avoid energy from below
    if (dir.y < 0.0) {
        skyColor = vec3(0.15, 0.12, 0.10);  // dark ground approximation
    }

    payload = vec4(skyColor, -1.0); // negative distance => miss
}

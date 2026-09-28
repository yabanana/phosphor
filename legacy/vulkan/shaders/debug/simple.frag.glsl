#version 460

layout(push_constant) uniform PC {
    mat4 mvp;
    vec4 cameraPos;
    vec4 lightDir;
} pc;

layout(location = 0) in vec3 fragNormal;
layout(location = 1) in vec3 fragWorldPos;

layout(location = 0) out vec4 outColor;

void main() {
    vec3 N = normalize(fragNormal);
    vec3 L = normalize(pc.lightDir.xyz);
    vec3 V = normalize(pc.cameraPos.xyz - fragWorldPos);
    vec3 H = normalize(L + V);

    float NdotL = max(dot(N, L), 0.0);
    float NdotH = max(dot(N, H), 0.0);

    vec3 ambient  = vec3(0.05, 0.05, 0.08);
    vec3 diffuse  = vec3(0.8, 0.6, 0.2) * NdotL; // gold color
    vec3 specular = vec3(1.0) * pow(NdotH, 64.0);

    vec3 color = ambient + diffuse + specular;
    // Simple Reinhard tonemap
    color = color / (color + vec3(1.0));
    // Gamma correction
    color = pow(color, vec3(1.0 / 2.2));

    outColor = vec4(color, 1.0);
}

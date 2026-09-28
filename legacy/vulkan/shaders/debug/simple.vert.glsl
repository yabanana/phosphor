#version 460

layout(push_constant) uniform PC {
    mat4 mvp;
    vec4 cameraPos;
    vec4 lightDir;
} pc;

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;

layout(location = 0) out vec3 fragNormal;
layout(location = 1) out vec3 fragWorldPos;

void main() {
    gl_Position = pc.mvp * vec4(inPosition, 1.0);
    fragWorldPos = inPosition; // object space for now
    fragNormal = inNormal;
}

export const vertex = `#version 300 es
precision highp float;
precision highp int;
uniform highp sampler2D model;
uniform int textureWidth;
uniform int texels;
uniform int basisCount;
uniform mat4 view;
uniform vec3 cameraPosition;
uniform vec2 viewport;
uniform vec2 focal;
uniform bool prefilter;
uniform bool conservativeBounds;
layout(location=0) in uint primitive;
flat out vec3 planeU;
flat out vec3 planeV;
flat out vec3 planeW;
flat out vec3 baseColor;
flat out vec3 gradientU;
flat out vec3 gradientV;
flat out vec3 normal;
flat out vec3 softness;
flat out vec2 projectedCenter;
flat out float opacity;
flat out uint primitiveId;
vec4 attributeAt(int field) {
    int index = int(primitive) * texels + field;
    return texelFetch(model, ivec2(index % textureWidth, index / textureWidth), 0);
}
bool supportBounds(float apothem, out vec2 lower, out vec2 upper) {
    lower = vec2(1e30);
    upper = vec2(-1e30);
    apothem += 4e-5 * max(1.0, abs(apothem));
    if (apothem <= 0.0) return false;
    float radius = 1.15470053838 * apothem;
    vec2 vertices[6] = vec2[6](vec2(radius, 0), vec2(radius * 0.5, apothem),
        vec2(-radius * 0.5, apothem), vec2(-radius, 0),
        vec2(-radius * 0.5, -apothem), vec2(radius * 0.5, -apothem));
    for (int index = 0; index < 6; index++) {
        vec3 a = vec3(vertices[index], 1), b = vec3(vertices[(index + 1) % 6], 1);
        float za = dot(a, planeW), zb = dot(b, planeW);
        if (za >= 0.2) {
            vec2 p = vec2(dot(a, planeU), dot(a, planeV)) / za;
            lower = min(lower, p); upper = max(upper, p);
        }
        if ((za < 0.2) != (zb < 0.2)) {
            vec3 edge = a + (0.2 - za) / (zb - za) * (b - a);
            vec2 p = vec2(dot(edge, planeU), dot(edge, planeV)) / 0.2;
            lower = min(lower, p); upper = max(upper, p);
        }
    }
    lower = max(vec2(0), floor(lower - 2.0));
    upper = min(viewport, ceil(upper + 2.0));
    return all(greaterThan(upper, lower));
}
void main() {
    vec4 centerAlpha = attributeAt(0);
    vec4 axisU = attributeAt(1);
    vec4 axisV = attributeAt(2);
    vec4 colorSoft = attributeAt(3);
    vec3 center = (view * vec4(centerAlpha.xyz, 1)).xyz;
    vec3 tangentU = (view * vec4(axisU.xyz, 0)).xyz;
    vec3 tangentV = (view * vec4(axisV.xyz, 0)).xyz;
    vec2 principal = (viewport - 1.0) * 0.5;
    planeW = vec3(tangentU.z, tangentV.z, center.z);
    planeU = focal.x * vec3(tangentU.x, tangentV.x, center.x) + principal.x * planeW;
    planeV = focal.y * vec3(tangentU.y, tangentV.y, center.y) + principal.y * planeW;
    opacity = centerAlpha.w;
    softness = vec3(axisU.w, axisV.w, colorSoft.w);
    float cutoff = 1.15470053838 * (1.0 + 5.537334267 / min(softness.x, min(softness.y, softness.z)));
    vec3 metric = vec3(cutoff * cutoff, cutoff * cutoff, -1);
    float denominator = dot(metric, planeW * planeW);
    vec3 form = metric / denominator;
    vec2 point = vec2(dot(form, planeU * planeW), dot(form, planeV * planeW));
    vec2 extent = sqrt(max(vec2(0.0001), point * point - vec2(dot(form, planeU * planeU), dot(form, planeV * planeV))));
    float radius = ceil(max(max(extent.x, extent.y), cutoff * 0.707106));
    if (prefilter) radius = max(radius, ceil(3.0 * 0.707106));
    vec2 lower = max(vec2(0), floor((point - radius) / 16.0) * 16.0);
    vec2 upper = min(viewport, floor((point + radius + 15.0) / 16.0) * 16.0);
    bool valid = denominator != 0.0 && all(greaterThan(upper, lower));
    if (conservativeBounds && !prefilter) {
        float margin = 5.537334267 / min(softness.x, min(softness.y, softness.z));
        if (opacity > 1.0 / 255.0) {
            float logCutoff = log((255.0 * (1.0 + 8.0 * 1.192092896e-7)) * opacity - 1.0);
            margin = logCutoff / (logCutoff < 0.0 ? max(softness.x, max(softness.y, softness.z))
                                                 : min(softness.x, min(softness.y, softness.z)));
        }
        valid = opacity >= 1.0 / 255.0 && supportBounds(1.0 + margin, lower, upper);
    }
    vec2 corner = vec2(float(gl_VertexID & 1), float(gl_VertexID >> 1));
    vec2 pixel = mix(lower, upper, corner);
    gl_Position = vec4(pixel.x / viewport.x * 2.0 - 1.0, 1.0 - pixel.y / viewport.y * 2.0, 0, 1);
    if (center.z <= 0.2 || !valid) gl_Position = vec4(2, 2, 2, 1);
    projectedCenter = center.xy / center.z * focal + principal;
    baseColor = colorSoft.xyz;
    gradientU = attributeAt(4).xyz;
    gradientV = attributeAt(5).xyz;
    vec3 direction = centerAlpha.xyz - cameraPosition;
    direction *= inversesqrt(dot(direction, direction) + 1e-8);
    normal = normalize(cross(axisU.xyz, axisV.xyz));
    normal *= dot(normal, direction) > 0.0 ? -1.0 : 1.0;
    primitiveId = primitive;
    if (basisCount == 3) {
        baseColor += attributeAt(6).xyz * direction.x + attributeAt(7).xyz * direction.y + attributeAt(8).xyz * direction.z;
    }
    if (basisCount == 15) {
        float dx = direction.x, dy = direction.y, dz = direction.z;
        float xx = dx*dx, yy = dy*dy, zz = dz*dz;
        float basis[15] = float[15](
            -0.4886025119*dy, 0.4886025119*dz, -0.4886025119*dx,
            1.0925484306*dx*dy, -1.0925484306*dy*dz, 0.3153915653*(2.0*zz-xx-yy),
            -1.0925484306*dx*dz, 0.5462742153*(xx-yy),
            -0.5900435899*dy*(3.0*xx-yy), 2.8906114426*dx*dy*dz,
            -0.4570457995*dy*(4.0*zz-xx-yy), 0.3731763326*dz*(2.0*zz-3.0*xx-3.0*yy),
            -0.4570457995*dx*(4.0*zz-xx-yy), 1.4453057213*dz*(xx-yy),
            -0.5900435899*dx*(xx-3.0*yy));
        for (int index=0; index<15; index++) baseColor += attributeAt(6+index).xyz * basis[index];
    }
}`;

export const fragment = `#version 300 es
precision highp float;
precision highp int;
uniform vec2 viewport;
uniform bool prefilter;
uniform int mode;
uniform float depthScale;
flat in vec3 planeU;
flat in vec3 planeV;
flat in vec3 planeW;
flat in vec3 baseColor;
flat in vec3 gradientU;
flat in vec3 gradientV;
flat in vec3 normal;
flat in vec3 softness;
flat in vec2 projectedCenter;
flat in float opacity;
flat in uint primitiveId;
out vec4 outputColor;
void main() {
    vec2 pixel = vec2(gl_FragCoord.x - 0.5, viewport.y - gl_FragCoord.y - 0.5);
    vec3 intersection = cross(pixel.x * planeW - planeU, pixel.y * planeW - planeV);
    if (intersection.z == 0.0) discard;
    vec2 local = intersection.xy / intersection.z;
    float depth = dot(vec3(local, 1), planeW);
    if (depth < 0.2) discard;
    vec3 edges = abs(vec3(local.y, 0.8660254*local.x + 0.5*local.y, 0.8660254*local.x - 0.5*local.y));
    int edge = edges.z >= edges.y ? 2 : 1;
    if (edges.x > edges[edge]) edge = 0;
    float sdf = edges[edge] - 1.0;
    float alpha = min(0.99, opacity / (1.0 + exp(sdf * softness[edge])));
    if (prefilter && mode == 0) {
        vec2 delta = pixel - projectedCenter;
        float filtered = min(0.99, opacity * exp(-dot(delta, delta)));
        if (filtered > alpha) { alpha = filtered; local = vec2(0); depth = planeW.z; }
    }
    if (alpha < 1.0 / 255.0) discard;
    vec3 color = baseColor + gradientU * local.x + gradientV * local.y;
    if (mode == 1 || mode == 2) {
        if (sdf > 0.0) discard;
        uint hashed = primitiveId * 1664525u + 1013904223u;
        color = vec3(float(hashed & 255u), float((hashed >> 8) & 255u), float((hashed >> 16) & 255u)) / 255.0;
        alpha = 1.0;
        if (mode == 2) {
            float line = 1.0 - smoothstep(0.0, max(fwidth(sdf) * 1.5, 0.001), -sdf);
            color = mix(clamp(baseColor, 0.0, 1.0) * 0.25, vec3(0.35, 1.0, 0.8), line);
        }
    }
    if (mode == 3) color = vec3(clamp(depth / depthScale, 0.0, 1.0));
    if (mode == 4) color = normal * 0.5 + 0.5;
    if (mode == 5) color = vec3(1);
    outputColor = vec4(color * alpha, alpha);
}`;

export const screenVertex = `#version 300 es
precision highp float;
out vec2 uv;
void main() {
    uv = vec2(float((gl_VertexID << 1) & 2), float(gl_VertexID & 2));
    gl_Position = vec4(uv * 2.0 - 1.0, 0, 1);
}`;

export const screenFragment = `#version 300 es
precision highp float;
uniform highp sampler2D accumulated;
uniform vec3 background;
in vec2 uv;
out vec4 outputColor;
void main() {
    vec4 value = texture(accumulated, uv);
    outputColor = vec4(clamp(value.rgb + (1.0 - value.a) * background, 0.0, 1.0), 1);
}`;

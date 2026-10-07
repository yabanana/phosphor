#include "renderer/offline_reference.h"
#include <algorithm>
#include <bit>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <unordered_map>

namespace phosphor {
namespace {
bool finite(const float* data,size_t n) {
    for(size_t i=0;i<n;++i) if(!std::isfinite(data[i])) return false;
    return true;
}
float norm2(const float* p) {return p[0]*p[0]+p[1]*p[1]+p[2]*p[2];}
void little(std::ostream& out,u32 word) {
    const char bytes[4]{char(word&255u),char((word>>8)&255u),char((word>>16)&255u),char((word>>24)&255u)};
    out.write(bytes,4);
}
void numberArray(std::ostream& out,const float* data,size_t n) {
    out<<'[';for(size_t i=0;i<n;++i) {if(i) out<<',';out<<data[i];}out<<']';
}
ReferenceExportResult fail(std::string why) {return {false,std::move(why),0,0,0};}
}
ReferenceExportResult validateReferenceScene(const OfflineReferenceScene& s) {
    const auto& c=s.camera;
    if(!c.width || !c.height || c.width>32768 || c.height>32768 || !finite(c.position,3)||!finite(c.direction,3)||
       !finite(c.up,3)||norm2(c.direction)<1e-12f||norm2(c.up)<1e-12f ||
       !std::isfinite(c.fovYRadians)||c.fovYRadians<=0 || c.fovYRadians>=3.14159f ||
       !std::isfinite(c.nearPlane)||!std::isfinite(c.farPlane)||c.nearPlane<=0||c.farPlane<=c.nearPlane)
        return fail("invalid reference camera");
    const float cross[3]{c.direction[1]*c.up[2]-c.direction[2]*c.up[1],
        c.direction[2]*c.up[0]-c.direction[0]*c.up[2],c.direction[0]*c.up[1]-c.direction[1]*c.up[0]};
    if(norm2(cross)<1e-12f) return fail("parallel reference camera axes");
    std::unordered_map<u32,const ReferenceTexture*> textures;
    for(const auto& t:s.textures) {
        if(!t.width||!t.height||t.width>32768||t.height>32768 ||
            t.rgba.size()!=size_t(t.width)*t.height*4 || !finite(t.rgba.data(),t.rgba.size()) ||
            !textures.emplace(t.index,&t).second) return fail("invalid or duplicated reference texture");
    }
    for(const auto& m:s.materials) {
        if(!finite(m.baseColor,4)||!finite(m.emissive,3)||!std::isfinite(m.metallic)||
           !std::isfinite(m.roughness)||!std::isfinite(m.normalScale)||!std::isfinite(m.alphaCutoff)||
           !std::isfinite(m.occlusionStrength)) return fail("nonfinite reference material");
        for(u32 id:{m.baseColorTex,m.normalTex,m.metallicRoughnessTex,m.occlusionTex,m.emissiveTex})
            if(id!=INVALID_TEXTURE_INDEX && !textures.contains(id)) return fail("missing exact material texture texels");
    }
    for(const auto& mesh:s.meshes) {
        if(mesh.indexCount%3 || size_t(mesh.indexOffset)+mesh.indexCount>s.indices.size())
            return fail("invalid reference mesh index range");
        for(u32 k=0;k<mesh.indexCount;++k) {
            const size_t vertex=size_t(mesh.vertexOffset)+s.indices[mesh.indexOffset+k];
            if(vertex>=s.vertices.size()) return fail("invalid reference vertex range");
            const auto& v=s.vertices[vertex];
            const float fields[]{v.px,v.py,v.pz,v.nx,v.ny,v.nz,v.tx,v.ty,v.tz,v.tw,v.u,v.v};
            if(!finite(fields,12)) return fail("nonfinite reference vertex");
        }
    }
    u32 live=0;
    for(const auto& i:s.worldInstances) if(i.flags&INSTANCE_FLAG_VALID) {
        if(!finite(i.modelMatrix,16)||i.meshIndex>=s.meshes.size()||i.materialIndex>=s.materials.size())
            return fail("invalid same-frame reference instance");
        // Require an affine, invertible WORLD matrix.
        const float* m=i.modelMatrix;
        const float det=m[0]*(m[5]*m[10]-m[9]*m[6])-m[4]*(m[1]*m[10]-m[9]*m[2])+m[8]*(m[1]*m[6]-m[5]*m[2]);
        if(std::abs(det)<1e-12f || m[3]!=0||m[7]!=0||m[11]!=0||m[15]!=1)
            return fail("noninvertible or nonaffine reference world matrix");
        if(i.flags&1u) ++live;
    }
    for(const auto& l:s.lights) if(l.type>LIGHT_SPOT || !finite(l.position,3)||!finite(l.direction,3)||
        !finite(l.color,3)||!std::isfinite(l.intensity)||!std::isfinite(l.range)||!std::isfinite(l.innerCone)||
        !std::isfinite(l.outerCone)) return fail("invalid reference light");
    for(const auto& l:s.sampledLights) if(l.type<1 || l.type>6 || !finite(l.position,3)||!finite(l.axisU,3)||
        !finite(l.axisV,3)||!finite(l.emission,3)||!std::isfinite(l.range)||!std::isfinite(l.radius)||
        !std::isfinite(l.innerCone)||!std::isfinite(l.outerCone)||!finite(l.uv0,2)||!finite(l.uv1,2)||!finite(l.uv2,2)||
        ((l.flags&2u)&&l.materialIndex>=s.materials.size())) return fail("invalid reference sampled light");
    if(!finite(s.skyRadiance,3)) return fail("invalid reference sky");
    return {true,{},u32(s.meshes.size()),live,u32(s.textures.size())};
}
bool writeLinearPfm(const std::filesystem::path& path,u32 w,u32 h,std::span<const float> rgb,std::string& error) {
    if(!w||!h||rgb.size()!=size_t(w)*h*3||!finite(rgb.data(),rgb.size())) {error="invalid linear PFM data";return false;}
    std::ofstream out(path,std::ios::binary);
    if(!out) {error="cannot open linear PFM";return false;}
    out<<"PF\n"<<w<<' '<<h<<"\n-1.0\n";
    for(u32 y=h;y>0;--y) for(u32 x=0;x<w;++x) for(u32 c=0;c<3;++c)
        little(out,std::bit_cast<u32>(rgb[(size_t(y-1)*w+x)*3+c]));
    if(!out) {error="failed linear PFM write";return false;}
    return true;
}
ReferenceExportResult exportOfflineReference(const OfflineReferenceScene& s,const std::filesystem::path& destination) {
    auto result=validateReferenceScene(s);if(!result.ok) return result;
    std::error_code ec;
    if(destination.empty()||std::filesystem::exists(destination,ec)) return fail("reference destination already exists or is empty");
    const auto staging=std::filesystem::path(destination.string()+".partial");
    if(std::filesystem::exists(staging,ec)) return fail("reference staging already exists");
    if(!std::filesystem::create_directories(staging,ec)||ec) return fail("cannot create reference staging");
    // Every failure keeps the partial snapshot for diagnosis; caller never sees
    // an apparently complete manifest at destination until the final rename.
    for(size_t meshID=0;meshID<s.meshes.size();++meshID) {
        const auto& mesh=s.meshes[meshID];
        u32 vertices=0;
        for(u32 k=0;k<mesh.indexCount;++k) vertices=std::max(vertices,s.indices[mesh.indexOffset+k]+1u);
        std::ofstream out(staging/("mesh_"+std::to_string(meshID)+".ply"),std::ios::binary);
        if(!out) return fail("cannot open reference PLY");
        out<<"ply\nformat binary_little_endian 1.0\nelement vertex "<<vertices
           <<"\nproperty float x\nproperty float y\nproperty float z\nproperty float nx\nproperty float ny\nproperty float nz\n"
           <<"property float s\nproperty float t\nelement face "<<mesh.indexCount/3
           <<"\nproperty list uchar uint vertex_indices\nend_header\n";
        for(u32 v=0;v<vertices;++v) {
            const auto& a=s.vertices[mesh.vertexOffset+v];
            for(float value:{a.px,a.py,a.pz,a.nx,a.ny,a.nz,a.u,a.v}) little(out,std::bit_cast<u32>(value));
        }
        for(u32 k=0;k<mesh.indexCount;k+=3) {out.put(char(3));for(u32 v=0;v<3;++v) little(out,s.indices[mesh.indexOffset+k+v]);}
        if(!out) return fail("failed reference PLY write");
    }
    for(const auto& t:s.textures) {
        std::vector<float> rgb(size_t(t.width)*t.height*3),alpha(rgb.size());
        for(size_t pixel=0;pixel<size_t(t.width)*t.height;++pixel) for(size_t c=0;c<3;++c) {
            rgb[pixel*3+c]=t.rgba[pixel*4+c];alpha[pixel*3+c]=t.rgba[pixel*4+3];
        }
        if(!writeLinearPfm(staging/("texture_"+std::to_string(t.index)+".pfm"),t.width,t.height,rgb,result.error)||
           !writeLinearPfm(staging/("alpha_"+std::to_string(t.index)+".pfm"),t.width,t.height,alpha,result.error)) {
            result.ok=false;return result;
        }
    }
    std::ofstream out(staging/"scene.json");
    if(!out) return fail("cannot open reference manifest");
    out<<std::setprecision(std::numeric_limits<float>::max_digits10);
    out<<"{\n\"schema\":1,\"state\":\"UNVERIFIED_REFERENCE_EXPORT\",\"linear\":true,\"units\":\"metres\","
       <<"\"frame\":"<<s.frame<<",\"revisions\":["<<s.geometryRevision<<','<<s.materialRevision<<','<<s.lightRevision<<"],"
       <<"\"camera\":{\"position\":";numberArray(out,s.camera.position,3);
    out<<",\"direction\":";numberArray(out,s.camera.direction,3);out<<",\"up\":";numberArray(out,s.camera.up,3);
    out<<",\"fov_y_radians\":"<<s.camera.fovYRadians<<",\"near\":"<<s.camera.nearPlane<<",\"far\":"<<s.camera.farPlane
       <<",\"width\":"<<s.camera.width<<",\"height\":"<<s.camera.height<<"},\"sky\":";numberArray(out,s.skyRadiance,3);
    out<<",\"textures\":[";
    for(size_t k=0;k<s.textures.size();++k) {if(k) out<<',';const auto& t=s.textures[k];
        out<<"{\"id\":"<<t.index<<",\"width\":"<<t.width<<",\"height\":"<<t.height<<",\"rgb\":\"texture_"<<t.index
           <<".pfm\",\"alpha\":\"alpha_"<<t.index<<".pfm\"}";}
    out<<"],\"materials\":[";
    for(size_t k=0;k<s.materials.size();++k) {if(k) out<<',';const auto& m=s.materials[k];
        out<<"{\"base\":";numberArray(out,m.baseColor,4);out<<",\"emissive\":";numberArray(out,m.emissive,3);
        out<<",\"metallic\":"<<m.metallic<<",\"roughness\":"<<m.roughness<<",\"normal_scale\":"<<m.normalScale
           <<",\"occlusion_strength\":"<<m.occlusionStrength<<",\"alpha_cutoff\":"<<m.alphaCutoff<<",\"flags\":"<<m.flags
           <<",\"textures\":["<<m.baseColorTex<<','<<m.normalTex<<','<<m.metallicRoughnessTex<<','<<m.occlusionTex<<','<<m.emissiveTex<<"]}";}
    out<<"],\"meshes\":[";
    for(size_t k=0;k<s.meshes.size();++k) {if(k) out<<',';out<<"\"mesh_"<<k<<".ply\"";}
    out<<"],\"instances\":[";bool first=true;
    for(size_t k=0;k<s.worldInstances.size();++k) {const auto& i=s.worldInstances[k];if(!(i.flags&INSTANCE_FLAG_VALID)||!(i.flags&1u)) continue;
        if(!first) out<<',';first=false;
        out<<"{\"slot\":"<<k<<",\"generation\":"<<i.generation<<",\"flags\":"<<i.flags<<",\"mesh\":"<<i.meshIndex
           <<",\"material\":"<<i.materialIndex<<",\"world\":";numberArray(out,i.modelMatrix,16);out<<'}';}
    out<<"],\"lights\":[";
    for(size_t k=0;k<s.lights.size();++k) {if(k) out<<',';const auto& l=s.lights[k];
        out<<"{\"type\":"<<l.type<<",\"position\":";numberArray(out,l.position,3);out<<",\"direction\":";numberArray(out,l.direction,3);
        out<<",\"color\":";numberArray(out,l.color,3);out<<",\"intensity\":"<<l.intensity<<",\"range\":"<<l.range
           <<",\"inner\":"<<l.innerCone<<",\"outer\":"<<l.outerCone<<'}';}
    out<<"],\"sampled_lights\":[";
    for(size_t k=0;k<s.sampledLights.size();++k) {if(k) out<<',';const auto& l=s.sampledLights[k];
        out<<"{\"id\":"<<l.id<<",\"generation\":"<<l.generation<<",\"type\":"<<l.type<<",\"flags\":"<<l.flags
           <<",\"position\":";numberArray(out,l.position,3);out<<",\"u\":";numberArray(out,l.axisU,3);out<<",\"v\":";numberArray(out,l.axisV,3);
        out<<",\"emission\":";numberArray(out,l.emission,3);out<<",\"radius\":"<<l.radius<<",\"range\":"<<l.range
           <<",\"inner\":"<<l.innerCone<<",\"outer\":"<<l.outerCone<<",\"material\":"<<l.materialIndex
           <<",\"uv0\":";numberArray(out,l.uv0,2);out<<",\"uv1\":";numberArray(out,l.uv1,2);out<<",\"uv2\":";numberArray(out,l.uv2,2);out<<'}';}
    out<<"]}\n";out.flush();
    if(!out) return fail("failed reference manifest write");out.close();
    std::filesystem::rename(staging,destination,ec);
    if(ec) return fail("failed reference publication rename");
    return result;
}
} // namespace phosphor

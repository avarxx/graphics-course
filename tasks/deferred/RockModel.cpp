#include "RockModel.hpp"

#include <cstring>

#include <etna/BlockingTransferHelper.hpp>
#include <etna/GlobalContext.hpp>
#include <glm/ext.hpp>
#include <spdlog/spdlog.h>
#include <tiny_gltf.h>

#include "HeightmapGenerator.hpp"


namespace
{

template <class T>
const T* accessor_data(
  const tinygltf::Model& model, const tinygltf::Accessor& acc, std::size_t& stride)
{
  const auto& view = model.bufferViews[static_cast<std::size_t>(acc.bufferView)];
  const auto& buf = model.buffers[static_cast<std::size_t>(view.buffer)];
  stride = view.byteStride != 0 ? view.byteStride : sizeof(T);
  return reinterpret_cast<const T*>(buf.data.data() + view.byteOffset + acc.byteOffset);
}

glm::mat4 node_matrix(const tinygltf::Node& node)
{
  if (node.matrix.size() == 16)
    return glm::make_mat4(node.matrix.data());
  glm::mat4 m{1.0f};
  if (node.translation.size() == 3)
    m = glm::translate(m, glm::vec3(glm::make_vec3(node.translation.data())));
  if (node.rotation.size() == 4)
    m *= glm::mat4_cast(glm::quat(
      static_cast<float>(node.rotation[3]),
      static_cast<float>(node.rotation[0]),
      static_cast<float>(node.rotation[1]),
      static_cast<float>(node.rotation[2])));
  if (node.scale.size() == 3)
    m = glm::scale(m, glm::vec3(glm::make_vec3(node.scale.data())));
  return m;
}

bool load_gltf(
  const std::filesystem::path& path,
  std::vector<RockVertex>& vertices,
  std::vector<std::uint32_t>& indices,
  glm::vec3& color)
{
  if (!std::filesystem::exists(path))
    return false;

  tinygltf::TinyGLTF loader;
  tinygltf::Model model;
  std::string err;
  std::string warn;
  if (!loader.LoadASCIIFromFile(&model, &err, &warn, path.string()))
  {
    spdlog::warn("Failed to load {}: {}", path.string(), err);
    return false;
  }

  if (!model.materials.empty())
  {
    const auto& f = model.materials[0].pbrMetallicRoughness.baseColorFactor;
    color = glm::vec3(f[0], f[1], f[2]) * 0.55f;
  }

  for (std::size_t n = 0; n < model.nodes.size(); ++n)
  {
    const auto& node = model.nodes[n];
    if (node.mesh < 0)
      continue;
    const glm::mat4 tm = node_matrix(node);
    const glm::mat3 ntm = glm::transpose(glm::inverse(glm::mat3(tm)));

    for (const auto& prim : model.meshes[static_cast<std::size_t>(node.mesh)].primitives)
    {
      auto posIt = prim.attributes.find("POSITION");
      auto nrmIt = prim.attributes.find("NORMAL");
      if (posIt == prim.attributes.end() || prim.indices < 0)
        continue;

      const auto& posAcc = model.accessors[static_cast<std::size_t>(posIt->second)];
      std::size_t posStride = 0;
      const auto* pos =
        reinterpret_cast<const std::byte*>(accessor_data<glm::vec3>(model, posAcc, posStride));
      std::size_t nrmStride = 0;
      const std::byte* nrm = nullptr;
      if (nrmIt != prim.attributes.end())
        nrm = reinterpret_cast<const std::byte*>(accessor_data<glm::vec3>(
          model, model.accessors[static_cast<std::size_t>(nrmIt->second)], nrmStride));

      const auto base = static_cast<std::uint32_t>(vertices.size());
      for (std::size_t i = 0; i < posAcc.count; ++i)
      {
        glm::vec3 p;
        std::memcpy(&p, pos + i * posStride, sizeof(p));
        glm::vec3 nn{0, 1, 0};
        if (nrm != nullptr)
          std::memcpy(&nn, nrm + i * nrmStride, sizeof(nn));
        vertices.push_back({glm::vec3(tm * glm::vec4(p, 1.0f)), glm::normalize(ntm * nn)});
      }

      const auto& idxAcc = model.accessors[static_cast<std::size_t>(prim.indices)];
      const auto& view = model.bufferViews[static_cast<std::size_t>(idxAcc.bufferView)];
      const auto* data = model.buffers[static_cast<std::size_t>(view.buffer)].data.data() +
        view.byteOffset + idxAcc.byteOffset;
      for (std::size_t i = 0; i < idxAcc.count; ++i)
      {
        std::uint32_t idx = 0;
        if (idxAcc.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT)
          std::memcpy(&idx, data + i * 4, 4);
        else if (idxAcc.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT)
        {
          std::uint16_t v;
          std::memcpy(&v, data + i * 2, 2);
          idx = v;
        }
        else
          idx = static_cast<std::uint32_t>(data[i]);
        indices.push_back(base + idx);
      }
    }
  }
  return !indices.empty();
}

// Noisy icosphere-like rock, used when the glTF is not available
void make_procedural(
  std::vector<RockVertex>& vertices, std::vector<std::uint32_t>& indices, std::uint32_t seed)
{
  constexpr std::uint32_t ST = 16;
  constexpr std::uint32_t SL = 10;
  for (std::uint32_t j = 0; j <= SL; ++j)
    for (std::uint32_t i = 0; i <= ST; ++i)
    {
      const float th = glm::pi<float>() * static_cast<float>(j) / SL;
      const float ph = 2.0f * glm::pi<float>() * static_cast<float>(i % ST) / ST;
      glm::vec3 d{std::sin(th) * std::cos(ph), std::cos(th), std::sin(th) * std::sin(ph)};
      const float r = 0.8f + 0.25f * prng::to01(prng::hash(i % ST, j, seed));
      vertices.push_back({d * r * glm::vec3(1.2f, 0.7f, 1.0f), d});
    }
  for (std::uint32_t j = 0; j < SL; ++j)
    for (std::uint32_t i = 0; i < ST; ++i)
    {
      const std::uint32_t a = j * (ST + 1) + i;
      const std::uint32_t b = a + ST + 1;
      indices.insert(indices.end(), {a, a + 1, b, a + 1, b + 1, b});
    }
  // Flat-ish normals from geometry
  std::vector<glm::vec3> acc(vertices.size(), glm::vec3(0));
  for (std::size_t t = 0; t < indices.size(); t += 3)
  {
    auto& a = vertices[indices[t]].position;
    auto& b = vertices[indices[t + 1]].position;
    auto& c = vertices[indices[t + 2]].position;
    const glm::vec3 n = glm::cross(b - a, c - a);
    for (int k = 0; k < 3; ++k)
      acc[indices[t + static_cast<std::size_t>(k)]] += n;
  }
  for (std::size_t i = 0; i < vertices.size(); ++i)
  {
    glm::vec3 n = glm::length(acc[i]) > 1e-8f ? glm::normalize(acc[i]) : vertices[i].normal;
    if (glm::dot(n, vertices[i].position) < 0)
      n = -n;
    vertices[i].normal = n;
  }
}

} // namespace

void RockModel::load(
  etna::OneShotCmdMgr& cmd_mgr, const std::filesystem::path& path, std::uint32_t seed)
{
  std::vector<RockVertex> vertices;
  std::vector<std::uint32_t> indices;
  baseColor = glm::vec3(0.45f, 0.42f, 0.38f);

  if (!load_gltf(path, vertices, indices, baseColor))
  {
    spdlog::warn("Rock model {} not found, using a procedural rock", path.string());
    vertices.clear();
    indices.clear();
    make_procedural(vertices, indices, seed);
  }

  // Normalize: center XZ, bottom slightly below 0, max horizontal extent = 1
  glm::vec3 mn{1e30f};
  glm::vec3 mx{-1e30f};
  for (const auto& v : vertices)
  {
    mn = glm::min(mn, v.position);
    mx = glm::max(mx, v.position);
  }
  const glm::vec3 size = mx - mn;
  const float s = 2.0f / std::max({size.x, size.z, 1e-4f});
  const glm::vec3 offset{(mn.x + mx.x) * 0.5f, mn.y + size.y * 0.15f, (mn.z + mx.z) * 0.5f};
  boundingRadius = 0.0f;
  for (auto& v : vertices)
  {
    v.position = (v.position - offset) * s;
    boundingRadius = std::max(boundingRadius, glm::length(v.position));
  }

  indexCount = static_cast<std::uint32_t>(indices.size());

  auto& ctx = etna::get_context();
  vertexBuffer = ctx.createBuffer(etna::Buffer::CreateInfo{
    .size = vertices.size() * sizeof(RockVertex),
    .bufferUsage = vk::BufferUsageFlagBits::eVertexBuffer | vk::BufferUsageFlagBits::eTransferDst,
    .name = "rock_vertices",
  });
  indexBuffer = ctx.createBuffer(etna::Buffer::CreateInfo{
    .size = indices.size() * sizeof(std::uint32_t),
    .bufferUsage = vk::BufferUsageFlagBits::eIndexBuffer | vk::BufferUsageFlagBits::eTransferDst,
    .name = "rock_indices",
  });

  etna::BlockingTransferHelper transfer(etna::BlockingTransferHelper::CreateInfo{
    .stagingSize =
      std::max(vertices.size() * sizeof(RockVertex), indices.size() * sizeof(std::uint32_t)),
  });
  transfer.uploadBuffer<RockVertex>(cmd_mgr, vertexBuffer, 0, vertices);
  transfer.uploadBuffer<std::uint32_t>(cmd_mgr, indexBuffer, 0, indices);
}

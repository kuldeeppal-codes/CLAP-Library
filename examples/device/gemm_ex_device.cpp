// Self-checking test of gemm_ex_strided_batched_device on device memory:
// fp16 inputs with fp16 and fp32 outputs, bf16 inputs with fp32 output.
//
//   gemm_ex_device [-rocm|-cuda]     (default -rocm)
//
// Operands are small integers, exact in fp16 and bf16, so the expected result
// is exact. Exit status: 0 pass, 1 failure, 77 skipped (no GPU runtime).
#include <clap/blas_factory.hpp>

#include <dlfcn.h>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <exception>
#include <memory>
#include <string>
#include <vector>

using namespace clap;

namespace {

struct Runtime {
  int (*malloc_fn)(void**, size_t) = nullptr;
  int (*free_fn)(void*) = nullptr;
  int (*memcpy_fn)(void*, const void*, size_t, int) = nullptr;
  int (*sync_fn)() = nullptr;

  bool load(bool cuda) {
    const char* libs_hip[] = {"libamdhip64.so", "libamdhip64.so.7", "libamdhip64.so.6"};
    const char* libs_cuda[] = {"libcudart.so", "libcudart.so.13", "libcudart.so.12"};
    void* h = nullptr;
    for (const char* lib : cuda ? libs_cuda : libs_hip)
      if ((h = dlopen(lib, RTLD_NOW | RTLD_GLOBAL)) != nullptr) break;
    if (h == nullptr) return false;
    const char* p = cuda ? "cuda" : "hip";
    auto sym = [&](const char* name) { return dlsym(h, (std::string(p) + name).c_str()); };
    malloc_fn = reinterpret_cast<decltype(malloc_fn)>(sym("Malloc"));
    free_fn = reinterpret_cast<decltype(free_fn)>(sym("Free"));
    memcpy_fn = reinterpret_cast<decltype(memcpy_fn)>(sym("Memcpy"));
    sync_fn = reinterpret_cast<decltype(sync_fn)>(sym("DeviceSynchronize"));
    int count = 0;
    auto count_fn = reinterpret_cast<int (*)(int*)>(sym("GetDeviceCount"));
    return malloc_fn && free_fn && memcpy_fn && sync_fn && count_fn && count_fn(&count) == 0 && count > 0;
  }
};

uint16_t to_f16(float f) {
  if (f == 0.0f) return 0;
  uint32_t u;
  std::memcpy(&u, &f, 4);
  const uint32_t sign = (u >> 16) & 0x8000u;
  const int exp = static_cast<int>((u >> 23) & 0xff) - 127 + 15;
  const uint32_t mant = (u >> 13) & 0x3ffu;
  return static_cast<uint16_t>(sign | (static_cast<uint32_t>(exp) << 10) | mant);
}

float from_f16(uint16_t h) {
  if ((h & 0x7fffu) == 0) return 0.0f;
  const uint32_t sign = (h & 0x8000u) << 16;
  const uint32_t exp = ((h >> 10) & 0x1fu) - 15 + 127;
  const uint32_t mant = (h & 0x3ffu) << 13;
  const uint32_t u = sign | (exp << 23) | mant;
  float f;
  std::memcpy(&f, &u, 4);
  return f;
}

uint16_t to_bf16(float f) {
  uint32_t u;
  std::memcpy(&u, &f, 4);
  return static_cast<uint16_t>(u >> 16);
}

float from_bf16(uint16_t h) {
  const uint32_t u = static_cast<uint32_t>(h) << 16;
  float f;
  std::memcpy(&f, &u, 4);
  return f;
}

bool check_case(IBlasBackend& be, const Runtime& rt, GemmElem ab, GemmElem c_type, Transpose ta,
                Transpose tb, int m, int n, int k, int batch, float beta) {
  const bool f16 = ab == GemmElem::F16;
  auto enc = [&](float v) { return f16 ? to_f16(v) : to_bf16(v); };
  auto dec = [&](uint16_t v) { return f16 ? from_f16(v) : from_bf16(v); };
  const int lda = ta == Transpose::NoTrans ? k : m;
  const int ldb = tb == Transpose::NoTrans ? n : k;
  const long long sa = static_cast<long long>(m) * k, sb = static_cast<long long>(k) * n,
                  sc = static_cast<long long>(m) * n;
  std::vector<uint16_t> a(sa * batch), b(sb * batch);
  for (size_t i = 0; i < a.size(); ++i) a[i] = enc(static_cast<float>(int(i * 7 % 5) - 2));
  for (size_t i = 0; i < b.size(); ++i) b[i] = enc(static_cast<float>(int(i * 3 % 7) - 3));
  std::vector<float> c0(sc * batch);
  for (size_t i = 0; i < c0.size(); ++i) c0[i] = static_cast<float>(int(i % 3) - 1);
  std::vector<float> want(c0.size());
  for (int z = 0; z < batch; ++z)
    for (int i = 0; i < m; ++i)
      for (int j = 0; j < n; ++j) {
        float acc = 0;
        for (int p = 0; p < k; ++p)
          acc += dec(a[z * sa + (ta == Transpose::NoTrans ? i * lda + p : p * lda + i)]) *
                 dec(b[z * sb + (tb == Transpose::NoTrans ? p * ldb + j : j * ldb + p)]);
        const size_t o = z * sc + i * n + j;
        want[o] = acc + beta * c0[o];
      }
  const bool c32 = c_type == GemmElem::F32;
  const size_t celem = c32 ? 4 : 2;
  std::vector<unsigned char> c(c0.size() * celem);
  for (size_t i = 0; i < c0.size(); ++i) {
    if (c32) std::memcpy(&c[i * 4], &c0[i], 4);
    else { const uint16_t v = enc(c0[i]); std::memcpy(&c[i * 2], &v, 2); }
  }
  void *da, *db, *dc;
  rt.malloc_fn(&da, a.size() * 2);
  rt.malloc_fn(&db, b.size() * 2);
  rt.malloc_fn(&dc, c.size());
  rt.memcpy_fn(da, a.data(), a.size() * 2, 1);
  rt.memcpy_fn(db, b.data(), b.size() * 2, 1);
  rt.memcpy_fn(dc, c.data(), c.size(), 1);
  bool ok = true;
  try {
    be.gemm_ex_strided_batched_device(Layout::RowMajor, ta, tb, m, n, k, 1.0f, da, lda, sa, db, ldb, sb,
                                      beta, dc, n, sc, batch, ab, c_type);
    rt.sync_fn();
    rt.memcpy_fn(c.data(), dc, c.size(), 2);
    for (size_t i = 0; i < c0.size() && ok; ++i) {
      float got;
      if (c32) std::memcpy(&got, &c[i * 4], 4);
      else { uint16_t v; std::memcpy(&v, &c[i * 2], 2); got = dec(v); }
      if (got != want[i]) {
        std::printf("FAIL gemm_ex %s->%s m=%d n=%d k=%d batch=%d: C[%zu]=%g want %g\n", f16 ? "f16" : "bf16",
                    c32 ? "f32" : (f16 ? "f16" : "bf16"), m, n, k, batch, i, got, want[i]);
        ok = false;
      }
    }
  } catch (const std::exception& e) {
    std::printf("FAIL gemm_ex threw: %s\n", e.what());
    ok = false;
  }
  rt.free_fn(da);
  rt.free_fn(db);
  rt.free_fn(dc);
  return ok;
}

}

int main(int argc, char** argv) {
  const bool cuda = argc > 1 && std::strcmp(argv[1], "-cuda") == 0;
  Runtime rt;
  if (!rt.load(cuda)) {
    std::printf("SKIP: no %s device runtime\n", cuda ? "CUDA" : "HIP");
    return 77;
  }
  std::unique_ptr<IBlasBackend> be;
  try {
    be = BlasFactory::create(cuda ? BackendType::CUDA : BackendType::AMD);
  } catch (const std::exception& e) {
    std::printf("SKIP: %s\n", e.what());
    return 77;
  }
  int run = 0, failed = 0;
  const Transpose ops[] = {Transpose::NoTrans, Transpose::Trans};
  const GemmElem kinds[][2] = {{GemmElem::F16, GemmElem::F16}, {GemmElem::F16, GemmElem::F32},
                               {GemmElem::BF16, GemmElem::F32}};
  for (const auto& kd : kinds)
    for (Transpose ta : ops)
      for (Transpose tb : ops)
        for (int batch : {1, 3})
          for (float beta : {0.0f, 1.0f}) {
            ++run;
            if (!check_case(*be, rt, kd[0], kd[1], ta, tb, 33, 17, 40, batch, beta)) ++failed;
          }
  std::printf("%s: %d / %d cases pass\n", failed ? "FAIL" : "PASS", run - failed, run);
  return failed ? 1 : 0;
}

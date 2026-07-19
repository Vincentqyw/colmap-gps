// Copyright (c), ETH Zurich and UNC Chapel Hill.
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
//     * Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.
//
//     * Redistributions in binary form must reproduce the above copyright
//       notice, this list of conditions and the following disclaimer in the
//       documentation and/or other materials provided with the distribution.
//
//     * Neither the name of ETH Zurich and UNC Chapel Hill nor the names of
//       its contributors may be used to endorse or promote products derived
//       from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

#include "colmap/feature/skywater_segmenter.h"

#include "colmap/feature/onnx_utils.h"
#include "colmap/feature/resources.h"
#include "colmap/util/logging.h"
#include "colmap/util/misc.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <filesystem>
#include <memory>

#ifdef COLMAP_ONNX_ENABLED

namespace colmap {
namespace {

constexpr int kInputH = 384;
constexpr int kInputW = 384;
constexpr int kNumClasses = 4;  // bg, sky, water, person

constexpr float kMean[3] = {0.485f, 0.456f, 0.406f};
constexpr float kStd[3] = {0.229f, 0.224f, 0.225f};
constexpr float kInv255 = 1.0f / 255.0f;

// ─── FP16 → FP32 ──────────────────────────────────────────────────

float HalfToFloat(uint16_t h) {
  const uint32_t sign = (h & 0x8000u) << 16;
  const uint32_t exp_raw = (h >> 10) & 0x1Fu;
  const uint32_t mant = h & 0x3FFu;
  if (exp_raw == 0) {
    if (mant == 0) {
      uint32_t r = sign; float f; std::memcpy(&f, &r, sizeof(f)); return f;
    }
    uint32_t m = mant; int e = 1;
    while ((m & 0x400u) == 0) { m <<= 1; --e; }
    uint32_t bits = sign | ((127 - 15 + e) << 23) | ((m & 0x3FFu) << 13);
    float f; std::memcpy(&f, &bits, sizeof(f)); return f;
  }
  if (exp_raw == 31) {
    uint32_t bits = sign | 0x7F800000u | (mant << 13);
    float f; std::memcpy(&f, &bits, sizeof(f)); return f;
  }
  uint32_t bits = sign | ((exp_raw - 15 + 127) << 23) | (mant << 13);
  float f; std::memcpy(&f, &bits, sizeof(f)); return f;
}

const auto kHalfToFloatLUT = []() {
  std::array<float, 65536> lut{};
  for (uint32_t i = 0; i < 65536; ++i)
    lut[i] = HalfToFloat(static_cast<uint16_t>(i));
  return lut;
}();

// ─── Image resize ─────────────────────────────────────────────────

void BilinearResize(const float* src, int src_h, int src_w,
                    float* dst, int dst_h, int dst_w, int channels) {
  const float scale_h = static_cast<float>(src_h) / dst_h;
  const float scale_w = static_cast<float>(src_w) / dst_w;
  const int src_stride = src_h * src_w;
  for (int c = 0; c < channels; ++c) {
    const float* ch_src = src + c * src_stride;
    float* ch_dst = dst + c * dst_h * dst_w;
    for (int dy = 0; dy < dst_h; ++dy) {
      const float sy = dy * scale_h;
      const int sy0 = static_cast<int>(sy);
      const int sy1 = std::min(sy0 + 1, src_h - 1);
      const float fy = sy - sy0;
      for (int dx = 0; dx < dst_w; ++dx) {
        const float sx = dx * scale_w;
        const int sx0 = static_cast<int>(sx);
        const int sx1 = std::min(sx0 + 1, src_w - 1);
        const float fx = sx - sx0;
        const float v00 = ch_src[sy0 * src_w + sx0];
        const float v01 = ch_src[sy0 * src_w + sx1];
        const float v10 = ch_src[sy1 * src_w + sx0];
        const float v11 = ch_src[sy1 * src_w + sx1];
        const float v0 = v00 + (v01 - v00) * fx;
        const float v1 = v10 + (v11 - v10) * fx;
        ch_dst[dy * dst_w + dx] = v0 + (v1 - v0) * fy;
      }
    }
  }
}

void NearestResizeUint8(const uint8_t* src, int src_h, int src_w,
                        uint8_t* dst, int dst_h, int dst_w) {
  const float scale_h = static_cast<float>(src_h) / dst_h;
  const float scale_w = static_cast<float>(src_w) / dst_w;
  for (int dy = 0; dy < dst_h; ++dy) {
    const int sy = std::min(static_cast<int>(dy * scale_h), src_h - 1);
    for (int dx = 0; dx < dst_w; ++dx) {
      const int sx = std::min(static_cast<int>(dx * scale_w), src_w - 1);
      dst[dy * dst_w + dx] = src[sy * src_w + sx];
    }
  }
}

// ─── Preprocessing ─────────────────────────────────────────────────

std::vector<float> Preprocess(const Bitmap& bitmap) {
  THROW_CHECK(bitmap.IsRGB());
  const int src_h = bitmap.Height();
  const int src_w = bitmap.Width();
  const int pitch = bitmap.Pitch();
  const int num_pixels = src_h * src_w;

  std::vector<float> src_chw(num_pixels * 3);
  const std::vector<uint8_t>& data = bitmap.RowMajorData();
  for (int c = 0; c < 3; ++c)
    for (int y = 0; y < src_h; ++y)
      for (int x = 0; x < src_w; ++x)
        src_chw[c * num_pixels + y * src_w + x] =
            static_cast<float>(data[y * pitch + 3 * x + c]);

  std::vector<float> resized(kInputH * kInputW * 3);
  BilinearResize(
      src_chw.data(), src_h, src_w, resized.data(), kInputH, kInputW, 3);

  std::vector<float> nchw(1 * 3 * kInputH * kInputW);
  for (int c = 0; c < 3; ++c) {
    const float scale = kInv255 / kStd[c];
    const float bias = -kMean[c] / kStd[c];
    const int off = c * kInputH * kInputW;
    for (int i = 0; i < kInputH * kInputW; ++i)
      nchw[off + i] = resized[off + i] * scale + bias;
  }
  return nchw;
}

// ─── Postprocessing ────────────────────────────────────────────────

std::vector<uint8_t> ArgmaxMask(const float* logits, int64_t h, int64_t w) {
  std::vector<uint8_t> mask(h * w);
  const int64_t stride = h * w;
  for (int64_t i = 0; i < stride; ++i) {
    const float v0 = logits[0 * stride + i];
    const float v1 = logits[1 * stride + i];
    const float v2 = logits[2 * stride + i];
    const float v3 = logits[3 * stride + i];
    uint8_t best = 0;
    float best_val = v0;
    if (v1 > best_val) { best_val = v1; best = 1; }
    if (v2 > best_val) { best_val = v2; best = 2; }
    if (v3 > best_val) { best_val = v3; best = 3; }
    mask[i] = best;
  }
  return mask;
}

}  // namespace

// ═══════════════════════════════════════════════════════════════════
// Options
// ═══════════════════════════════════════════════════════════════════

bool SkyWaterSegmentationOptions::Check() const {
  if (!enabled) return true;
  if (classes_to_mask < 0 || classes_to_mask > 15) {
    LOG(ERROR) << "classes_to_mask must be in [0, 15]";
    return false;
  }
  return true;
}

// ═══════════════════════════════════════════════════════════════════
// Segmenter
// ═══════════════════════════════════════════════════════════════════

SkyWaterSegmenter::SkyWaterSegmenter(
    const SkyWaterSegmentationOptions& options)
    : options_(options) {
  THROW_CHECK(options_.Check());
  if (!options_.enabled) return;

  // Try GPU first; fall back to CPU on failure.
  bool using_gpu = InitWithGPU(true);
  if (!using_gpu) {
    LOG(WARNING) << "SkyWaterSegmenter: GPU init failed, falling back to CPU";
    using_gpu = InitWithGPU(false);
  }
  if (!using_gpu) {
    LOG(ERROR) << "SkyWaterSegmenter: both GPU and CPU initialization "
                  "failed, segmentation disabled";
    return;
  }
  LOG(INFO) << "SkyWaterSegmenter: model loaded successfully ("
            << (using_gpu ? "GPU/CUDA" : "CPU") << ")";
}

bool SkyWaterSegmenter::InitWithGPU(bool use_gpu) {
  const std::string& input_path = options_.model_path.empty()
      ? static_cast<const std::string&>(kDefaultSkyWaterSegmenterUri)
      : options_.model_path;

  VLOG(1) << "SkyWaterSegmenter: trying to load model ("
          << (use_gpu ? "GPU" : "CPU") << "): " << input_path;

  try {
    model_ = std::make_unique<ONNXModel>(
        input_path,
        options_.num_threads,
        use_gpu,
        options_.gpu_index);

    // Validate I/O. The model has fully dynamic shapes (batch, H, W), so use
    // -1 as wildcard for all variable dimensions.
    THROW_CHECK_GE(model_->input_shapes().size(), 1);
    ThrowCheckONNXNode(model_->input_names()[0],
                       "input",
                       model_->input_shapes()[0],
                       {-1, 3, -1, -1});

    THROW_CHECK_GE(model_->output_shapes().size(), 1);
    ThrowCheckONNXNode(model_->output_names()[0],
                       "output",
                       model_->output_shapes()[0],
                       {-1, kNumClasses, -1, -1});

    LOG(INFO) << "SkyWaterSegmenter: ONNX session created (GPU=" << use_gpu
              << ")";
    valid_ = true;
    return true;
  } catch (const std::exception& e) {
    LOG(ERROR) << "SkyWaterSegmenter init failed"
               << " (GPU=" << use_gpu << "): " << e.what();
    valid_ = false;
    return false;
  }
}

SkyWaterSegmenter::~SkyWaterSegmenter() = default;

bool SkyWaterSegmenter::IsValid() const { return valid_; }

Bitmap SkyWaterSegmenter::GenerateMask(const Bitmap& bitmap) {
  if (!valid_) {
    LOG(ERROR) << "SkyWaterSegmenter is not valid";
    return Bitmap();
  }

  // The segmentation model expects RGB.  If extraction runs with as_rgb=false
  // (e.g. SIFT), convert the bitmap on-the-fly.
  Bitmap rgb_bitmap(0, 0, false);
  const Bitmap* input = &bitmap;
  if (!bitmap.IsRGB()) {
    rgb_bitmap = bitmap.CloneAsRGB();
    input = &rgb_bitmap;
  }

  const int orig_h = input->Height();
  const int orig_w = input->Width();

  try {
    // Preprocess.
    std::vector<float> nchw = Preprocess(*input);

    // Create input tensor.
    const std::array<int64_t, 4> input_shape = {1, 3, kInputH, kInputW};
    auto memory_info =
        Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);

    std::vector<Ort::Value> input_tensors;
    input_tensors.emplace_back(Ort::Value::CreateTensor<float>(
        memory_info,
        nchw.data(),
        nchw.size(),
        input_shape.data(),
        input_shape.size()));

    // Run inference (ONNXModel::Run takes const ref to vector).
    std::vector<Ort::Value> output_tensors = model_->Run(input_tensors);
    THROW_CHECK_GE(output_tensors.size(), 1);

    // Parse output.
    auto& out_val = output_tensors[0];
    auto out_info = out_val.GetTensorTypeAndShapeInfo();
    auto out_shape = out_info.GetShape();
    const int64_t out_h = (out_shape.size() >= 3) ? out_shape[2] : kInputH;
    const int64_t out_w = (out_shape.size() >= 4) ? out_shape[3] : kInputW;
    const size_t out_count =
        out_shape[0] * out_shape[1] * out_h * out_w;

    // Convert FP16→FP32 if needed (detected on first run).
    const auto elem_type = out_info.GetElementType();
    std::vector<float> logits(out_count);
    if (elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) {
      const auto* fp16_data =
          out_val.GetTensorData<Ort::Float16_t>();
      for (size_t i = 0; i < out_count; ++i)
        logits[i] = kHalfToFloatLUT[fp16_data[i].val];
    } else {
      const auto* fp32_data = out_val.GetTensorData<float>();
      std::copy_n(fp32_data, out_count, logits.begin());
    }

    // Argmax → class indices → resize → binary mask.
    std::vector<uint8_t> class_mask =
        ArgmaxMask(logits.data(), out_h, out_w);

    std::vector<uint8_t> mask_orig(orig_h * orig_w);
    NearestResizeUint8(
        class_mask.data(), out_h, out_w, mask_orig.data(), orig_h, orig_w);

    const int classes_to_mask = options_.classes_to_mask;
    Bitmap mask_bitmap(orig_w, orig_h, /*as_rgb=*/false);
    std::vector<uint8_t>& mask_data = mask_bitmap.RowMajorData();
    for (size_t i = 0; i < mask_orig.size(); ++i) {
      const uint8_t cls = mask_orig[i];
      mask_data[i] =
          ((cls < 8) && (classes_to_mask & (1 << cls))) ? 0 : 255;
    }

    VLOG(3) << "SkyWaterSegmenter: generated mask " << orig_w << "x"
            << orig_h;
    return mask_bitmap;
  } catch (const std::exception& e) {
    LOG(ERROR) << "SkyWaterSegmenter inference failed: " << e.what();
    return Bitmap();
  }
}

}  // namespace colmap

#endif  // COLMAP_ONNX_ENABLED

// ═══════════════════════════════════════════════════════════════════
// Stub when ONNX not compiled
// ═══════════════════════════════════════════════════════════════════
#ifndef COLMAP_ONNX_ENABLED
namespace colmap {

bool SkyWaterSegmentationOptions::Check() const {
  if (enabled) {
    LOG(ERROR)
        << "SkyWaterSegmentation requires ONNX support (not compiled)";
    return false;
  }
  return true;
}

SkyWaterSegmenter::SkyWaterSegmenter(
    const SkyWaterSegmentationOptions& options)
    : options_(options) {
  if (options_.enabled) {
    LOG(ERROR)
        << "SkyWaterSegmentation requires ONNX support (not compiled)";
  }
}

SkyWaterSegmenter::~SkyWaterSegmenter() = default;

bool SkyWaterSegmenter::IsValid() const { return false; }

Bitmap SkyWaterSegmenter::GenerateMask(const Bitmap& /*bitmap*/) {
  LOG(ERROR) << "SkyWaterSegmentation requires ONNX support";
  return Bitmap();
}

}  // namespace colmap
#endif  // !COLMAP_ONNX_ENABLED

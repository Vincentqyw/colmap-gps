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

// Precomputed ImageNet normalization:  (v/255 - mean) / std
constexpr float kNormScale[3] = {
    kInv255 / kStd[0], kInv255 / kStd[1], kInv255 / kStd[2]};
constexpr float kNormBias[3] = {
    -kMean[0] / kStd[0], -kMean[1] / kStd[1], -kMean[2] / kStd[2]};

// ─── Image resize ──────────────────────────────────────────────────

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

// Fused: uint8 HWC → bilinear resize → float CHW → ImageNet normalize.
// Reads directly from the uint8 source during interpolation — avoids a
// full-resolution float CHW intermediate (saves ~88 MB for a 6K×4K image).
std::vector<float> Preprocess(const Bitmap& bitmap) {
  const int src_h = bitmap.Height();
  const int src_w = bitmap.Width();
  const int pitch = bitmap.Pitch();
  const std::vector<uint8_t>& data = bitmap.RowMajorData();

  const float scale_h = static_cast<float>(src_h) / kInputH;
  const float scale_w = static_cast<float>(src_w) / kInputW;

  std::vector<float> output(kInputH * kInputW * 3);

  for (int c = 0; c < 3; ++c) {
    const float s = kNormScale[c];
    const float b = kNormBias[c];
    float* ch_dst = output.data() + c * kInputH * kInputW;
    for (int dy = 0; dy < kInputH; ++dy) {
      const float sy = dy * scale_h;
      const int sy0 = static_cast<int>(sy);
      const int sy1 = std::min(sy0 + 1, src_h - 1);
      const float fy = sy - sy0;
      for (int dx = 0; dx < kInputW; ++dx) {
        const float sx = dx * scale_w;
        const int sx0 = static_cast<int>(sx);
        const int sx1 = std::min(sx0 + 1, src_w - 1);
        const float fx = sx - sx0;

        // Read uint8 pixels directly and interpolate.
        const float v00 = data[sy0 * pitch + 3 * sx0 + c];
        const float v01 = data[sy0 * pitch + 3 * sx1 + c];
        const float v10 = data[sy1 * pitch + 3 * sx0 + c];
        const float v11 = data[sy1 * pitch + 3 * sx1 + c];

        const float v0 = v00 + (v01 - v00) * fx;
        const float v1 = v10 + (v11 - v10) * fx;
        ch_dst[dy * kInputW + dx] = (v0 + (v1 - v0) * fy) * s + b;
      }
    }
  }
  return output;
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

const std::string& SkyWaterSegmentationOptions::ModelPath() const {
  return use_fp16 ? fp16_model_path : fp32_model_path;
}

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
  valid_ = true;
  LOG(INFO) << "SkyWaterSegmenter: model loaded successfully ("
            << (using_gpu ? "GPU/CUDA" : "CPU") << ")";
}

bool SkyWaterSegmenter::InitWithGPU(bool use_gpu) {
  VLOG(1) << "SkyWaterSegmenter: trying to load model ("
          << (use_gpu ? "GPU" : "CPU") << ")";

  try {
    model_ = std::make_unique<ONNXModel>(
        options_.ModelPath(),
        options_.num_threads,
        use_gpu,
        options_.gpu_index);

    // Validate I/O shapes.  The model has fully dynamic dims (batch, H, W);
    // use -1 as wildcard.
    THROW_CHECK_GE(model_->input_shapes().size(), 1);
    ThrowCheckONNXNode(model_->input_names()[0],
                       "input",
                       model_->input_shapes()[0],
                       {-1, 3, -1, -1});

    THROW_CHECK_GE(model_->output_shapes().size(), 1);
    ThrowCheckONNXNode(model_->output_names()[0],
                       "output_fp32",
                       model_->output_shapes()[0],
                       {-1, kNumClasses, -1, -1});

    LOG(INFO) << "SkyWaterSegmenter: ONNX session created (GPU=" << use_gpu
              << ")";
    return true;
  } catch (const std::exception& e) {
    LOG(ERROR) << "SkyWaterSegmenter init failed"
               << " (GPU=" << use_gpu << "): " << e.what();
    return false;
  }
}

SkyWaterSegmenter::~SkyWaterSegmenter() = default;

Bitmap SkyWaterSegmenter::GenerateMask(const Bitmap& bitmap) {
  if (!valid_) {
    LOG(ERROR) << "SkyWaterSegmenter is not valid";
    return Bitmap();
  }

  // Convert grayscale to RGB on-the-fly if needed (e.g. SIFT extraction).
  const Bitmap* input = &bitmap;
  Bitmap rgb_temp;
  if (!bitmap.IsRGB()) {
    rgb_temp = bitmap.CloneAsRGB();
    input = &rgb_temp;
  }

  const int orig_h = input->Height();
  const int orig_w = input->Width();

  try {
    // Preprocess: RGB → normalized NCHW [1, 3, 384, 384].
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

    // Run inference.
    std::vector<Ort::Value> output_tensors = model_->Run(input_tensors);
    THROW_CHECK_GE(output_tensors.size(), 1);

    // Parse output.
    auto& out_val = output_tensors[0];
    auto out_info = out_val.GetTensorTypeAndShapeInfo();
    auto out_shape = out_info.GetShape();
    const int64_t out_h = (out_shape.size() >= 3) ? out_shape[2] : kInputH;
    const int64_t out_w = (out_shape.size() >= 4) ? out_shape[3] : kInputW;
    // Argmax → class indices (model outputs FP32, like ALIKED).
    const float* logits = out_val.GetTensorData<float>();
    std::vector<uint8_t> class_mask = ArgmaxMask(logits, out_h, out_w);

    // Resize class indices directly into the output bitmap and threshold
    // in-place — avoids an intermediate mask_orig allocation.
    const int classes_to_mask = options_.classes_to_mask;
    Bitmap mask_bitmap(orig_w, orig_h, /*as_rgb=*/false);
    std::vector<uint8_t>& mask_data = mask_bitmap.RowMajorData();
    NearestResizeUint8(
        class_mask.data(), out_h, out_w, mask_data.data(), orig_h, orig_w);
    for (size_t i = 0; i < mask_data.size(); ++i) {
      mask_data[i] =
          ((mask_data[i] < 8) && (classes_to_mask & (1 << mask_data[i])))
              ? 0 : 255;
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

Bitmap SkyWaterSegmenter::GenerateMask(const Bitmap& /*bitmap*/) {
  LOG(ERROR) << "SkyWaterSegmentation requires ONNX support";
  return Bitmap();
}

}  // namespace colmap
#endif  // !COLMAP_ONNX_ENABLED

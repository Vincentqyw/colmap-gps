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

#pragma once

#include "colmap/feature/resources.h"
#include "colmap/sensor/bitmap.h"

#include <memory>
#include <string>

namespace colmap {

class ONNXModel;

// Sky/water/person segmentation using an ONNX model.
//
// The model performs 4-class segmentation:
//   0 = background, 1 = sky, 2 = water, 3 = person
//
// Input:  RGB image, any size (internally resized to 384x384)
// Output: Grayscale mask at original resolution, where:
//   - 255 (white) = keep (background class)
//   - 0   (black) = remove (selected classes: sky, water, person)

struct SkyWaterSegmentationOptions {
  bool enabled = false;
  std::string model_path = kDefaultSkyWaterSegmenterUri;
  // Bitmask: 2=sky, 4=water, 8=person. Default: 6 = sky|water
  int classes_to_mask = 6;
  int num_threads = -1;
  bool use_gpu = true;
  std::string gpu_index = "-1";

  bool Check() const;
};

class SkyWaterSegmenter {
 public:
  explicit SkyWaterSegmenter(const SkyWaterSegmentationOptions& options);
  ~SkyWaterSegmenter();

  Bitmap GenerateMask(const Bitmap& bitmap);
  bool IsValid() const;

 private:
  bool InitWithGPU(bool use_gpu);

  SkyWaterSegmentationOptions options_;
  std::unique_ptr<ONNXModel> model_;
  bool output_is_fp16_ = false;
  bool valid_ = false;
};

}  // namespace colmap

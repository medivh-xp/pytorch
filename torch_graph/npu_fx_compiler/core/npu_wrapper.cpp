/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2021. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <atomic>
#include <iostream>
#include <memory>
#include <map>
#include <unordered_map>
#include <unordered_set>

#include "Python.h"
#include "pybind11/chrono.h"
#include "pybind11/complex.h"
#include "pybind11/functional.h"
#include "pybind11/stl.h"

#include "ge/ge_api.h"
#include "ge_ir.pb.h"

namespace py = pybind11;

namespace npu {
PYBIND11_MODULE(_npu_fx_compiler, m) {
  (void)m.def("StupidRepeat", [](const char *device_name, int times) {
    for (int i = 0; i < times; i++) {
      std::cerr << device_name;
    }
  });

  (void)m.def("ParseSerializedGraph", [](const std::string &serialized_graph) {
    ge::proto::GraphDef graph_def;
    graph_def.ParseFromString(serialized_graph);

    std::cerr << "graph_def.op_size() = " << graph_def.op_size() << std::endl;
    for (const auto &op : graph_def.op()) {
      std::cerr << "op.name() = " << op.name() << ", op.type() = " << op.type() << std::endl;
    }
  });
};
}  // namespace npu

# Copyright 2023 InstaDeep Ltd. All rights reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from og_marl.offline_dataset import download_and_unzip_dataset

# download_and_unzip_dataset("voltage_control", "case33_3min_final", dataset_base_dir="datasets")

download_and_unzip_dataset("flatland", "3_trains", dataset_base_dir="datasets")
download_and_unzip_dataset("flatland", "5_trains", dataset_base_dir="datasets")

# download_and_unzip_dataset("smac_v1", "3s5z_vs_3s6z", dataset_base_dir="datasets")
# download_and_unzip_dataset("smac_v1", "2c_vs_64zg", dataset_base_dir="datasets")
# download_and_unzip_dataset("smac_v1", "27m_vs_30m", dataset_base_dir="datasets")

# download_and_unzip_dataset("smac_v2", "terran_5_vs_5", dataset_base_dir="datasets")
# download_and_unzip_dataset("smac_v2", "zerg_5_vs_5", dataset_base_dir="datasets")
# download_and_unzip_dataset("smac_v2", "terran_10_vs_10", dataset_base_dir="datasets")

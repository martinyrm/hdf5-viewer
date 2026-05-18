#include "imgui.h"
#include "imgui_internal.h"
#include "imgui_impl_opengl3.h"
#include "imgui_impl_sdl2.h"
#include "implot.h"

#include <SDL.h>
#include <SDL_opengl.h>
#include <hdf5.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cctype>
#include <ctime>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <functional>
#include <future>
#include <limits>
#include <memory>
#include <mutex>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <thread>
#include <utility>
#include <vector>

namespace {

std::mutex g_hdf5_mutex;
constexpr std::chrono::milliseconds kSwmrPollInterval(1000);
constexpr std::chrono::milliseconds kIdleFrameInterval(250);
constexpr std::chrono::milliseconds kActiveFrameInterval(16);

template <herr_t (*CloseFn)(hid_t)> class H5Object {
  public:
    H5Object() = default;
    explicit H5Object(hid_t id) : id_(id) {}
    H5Object(const H5Object &) = delete;
    H5Object &operator=(const H5Object &) = delete;

    H5Object(H5Object &&other) noexcept : id_(std::exchange(other.id_, -1)) {}

    H5Object &operator=(H5Object &&other) noexcept {
        if (this != &other) {
            close();
            id_ = std::exchange(other.id_, -1);
        }
        return *this;
    }

    ~H5Object() { close(); }

    [[nodiscard]] hid_t get() const { return id_; }
    [[nodiscard]] bool valid() const { return id_ >= 0; }

    void close() {
        if (id_ >= 0) {
            CloseFn(id_);
            id_ = -1;
        }
    }

  private:
    hid_t id_ = -1;
};

using H5File = H5Object<H5Fclose>;
using H5Dataset = H5Object<H5Dclose>;
using H5Space = H5Object<H5Sclose>;
using H5Type = H5Object<H5Tclose>;
using H5Prop = H5Object<H5Pclose>;

H5File open_readonly_file(const std::string &path, bool prefer_swmr = false, bool *opened_swmr = nullptr) {
    if (opened_swmr != nullptr) {
        *opened_swmr = false;
    }

#ifdef H5F_ACC_SWMR_READ
    const unsigned swmr_flags = H5F_ACC_RDONLY | H5F_ACC_SWMR_READ;
    if (prefer_swmr) {
        H5File file(H5Fopen(path.c_str(), swmr_flags, H5P_DEFAULT));
        if (file.valid()) {
            if (opened_swmr != nullptr) {
                *opened_swmr = true;
            }
            return file;
        }
    }
#else
    (void)prefer_swmr;
#endif

    H5File file(H5Fopen(path.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT));
    if (file.valid()) {
        return file;
    }

#ifdef H5F_ACC_SWMR_READ
    if (!prefer_swmr) {
        H5File swmr_file(H5Fopen(path.c_str(), swmr_flags, H5P_DEFAULT));
        if (swmr_file.valid()) {
            if (opened_swmr != nullptr) {
                *opened_swmr = true;
            }
            return swmr_file;
        }
    }
#endif

    return H5File();
}

struct DatasetInfo {
    std::string path;
    std::vector<hsize_t> dims;
    H5T_class_t type_class = H5T_NO_CLASS;
    size_t type_size = 0;
    bool numeric = false;
    bool scalar = false;
    hsize_t element_count = 0;
    H5D_layout_t layout = H5D_LAYOUT_ERROR;
    std::vector<hsize_t> chunk_dims;
    hsize_t storage_size = 0;
    std::vector<std::string> filters;
};

struct DatasetTreeNode {
    std::string name;
    std::string path;
    int dataset_index = -1;
    std::vector<DatasetTreeNode> children;
};

struct AxisSpec {
    std::string path;
    std::string label;
};

struct AxisGuess {
    AxisSpec x;
    AxisSpec y;
};

enum class LoadedKind { None, Scalar, Line1D, Heatmap2D };

struct LoadedDataset {
    DatasetInfo info;
    LoadedKind kind = LoadedKind::None;
    std::string scalar_text;
    std::string note;

    std::vector<float> line_values;
    std::vector<double> line_x_values;
    hsize_t source_count = 0;
    hsize_t source_stride = 1;
    bool x_values_increasing = true;
    bool x_values_decreasing = false;
    std::string x_source_path;

    std::vector<uint8_t> rgba;
    std::vector<float> heat_values;
    int texture_width = 0;
    int texture_height = 0;
    hsize_t source_rows = 0;
    hsize_t source_cols = 0;
    hsize_t row_stride = 1;
    hsize_t col_stride = 1;

    double x_min = 0.0;
    double x_max = 1.0;
    double y_min = 0.0;
    double y_max = 1.0;
    std::string x_label = "index";
    std::string y_label = "value";

    float value_min = 0.0f;
    float value_max = 1.0f;
    float applied_color_min = std::numeric_limits<float>::quiet_NaN();
    float applied_color_max = std::numeric_limits<float>::quiet_NaN();
};

struct LoadConfig {
    size_t max_line_values = 12000000;
    int max_texture_side = 4096;
    size_t max_texture_cells = 12000000;
    bool transpose_2d = false;
    hsize_t x_fallback_count = 0;
    hsize_t y_fallback_count = 0;
    std::string x_fallback_label = "column";
    std::string y_fallback_label = "row";
};

struct TexturePreviewPlan {
    bool valid = false;
    hsize_t row_stride = 1;
    hsize_t col_stride = 1;
    hsize_t rows = 0;
    hsize_t cols = 0;
    hsize_t cells = 0;
};

struct LoadResult {
    int token = 0;
    bool ok = false;
    std::string error;
    std::shared_ptr<LoadedDataset> data;
};

struct SwmrPollResult {
    int token = 0;
    bool ok = false;
    bool opened_swmr = false;
    bool selected_shape_changed = false;
    std::string error;
    std::string selected_path;
    std::vector<DatasetInfo> datasets;
    std::string caption;
};

struct LineCache {
    bool valid = false;
    const float *source_ptr = nullptr;
    const double *x_source_ptr = nullptr;
    size_t source_size = 0;
    double data_x_min = 0.0;
    double data_x_max = 0.0;
    double view_x_min = 0.0;
    double view_x_max = 0.0;
    int pixel_width = 0;
    std::vector<double> x;
    std::vector<double> y;
};

struct RangeControl {
    bool automatic = true;
    double min = 0.0;
    double max = 1.0;
};

struct ColorRangeControl {
    bool auto_min = true;
    bool auto_max = true;
    double min = 0.0;
    double max = 1.0;
};

struct PlotControls {
    RangeControl x;
    RangeControl y;
    ColorRangeControl color;
};

struct FileTab {
    int id = 0;
    std::array<char, 4096> file_path{};
    std::array<char, 256> filter{};
    std::string current_file;
    std::vector<DatasetInfo> datasets;
    DatasetTreeNode dataset_tree;
    int selected_index = -1;
    int x_axis_index = -2; // -2 auto, -1 indices, >=0 dataset index
    int y_axis_index = -2; // 2D only: -2 auto, -1 row/column indices, >=0 dataset index
    bool transpose_2d = false;
    bool x_datetime = false;
    bool x_datetime_utc = false;
    bool show_file_details = false;
    std::string status;
    std::string error;
    std::string caption;

    std::future<LoadResult> load_future;
    bool loading = false;
    bool loading_quiet = false;
    bool loading_preserve_controls = false;
    int load_token = 0;
    std::shared_ptr<LoadedDataset> loaded;
    LineCache line_cache;
    PlotControls controls;
    PlotControls preserved_controls;

    bool swmr_available = false;
    bool live_swmr_enabled = false;
    bool swmr_polling = false;
    int swmr_token = 0;
    std::future<SwmrPollResult> swmr_future;
    std::chrono::steady_clock::time_point next_swmr_poll{};
    std::string live_status;

    GLuint heat_texture = 0;
};

struct FilePicker {
    bool request_open = false;
    bool visible = false;
    std::filesystem::path directory;
    std::array<char, 4096> selected_path{};
    std::string error;
};

struct AppState {
    std::vector<std::unique_ptr<FileTab>> tabs;
    int next_tab_id = 1;
    int active_tab_id = 0;
    int plot_focus_request_id = 0;
    bool show_plot_captions = true;
    FilePicker picker;
    bool dock_layout_built = false;
    ImGuiID dockspace_id = 0;
    ImGuiID file_dock_id = 0;
    ImGuiID plot_dock_id = 0;
    int gl_max_texture_size = 4096;
    bool show_performance_hud = false;
    std::string gl_vendor;
    std::string gl_renderer;
    std::string gl_version;
    std::string glsl_version;
    float ui_framebuffer_scale = 1.0f;
    double last_frame_ms = 0.0;
    int last_event_count = 0;
    bool last_used_idle_wait = false;
    bool last_fast_frame = false;
    bool last_background_work = false;
    bool last_ui_active = false;
};

std::string trim(std::string s) {
    const auto is_space = [](unsigned char ch) { return std::isspace(ch) != 0; };
    s.erase(s.begin(), std::find_if_not(s.begin(), s.end(), is_space));
    s.erase(std::find_if_not(s.rbegin(), s.rend(), is_space).base(), s.end());
    return s;
}

std::string lower_ascii(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char ch) {
        return static_cast<char>(std::tolower(ch));
    });
    return s;
}

void copy_to_buffer(std::array<char, 4096> &buffer, const std::string &value) {
    std::fill(buffer.begin(), buffer.end(), '\0');
    std::strncpy(buffer.data(), value.c_str(), buffer.size() - 1);
}

std::string base_name(const std::string &path) {
    const size_t pos = path.find_last_of('/');
    return pos == std::string::npos ? path : path.substr(pos + 1);
}

std::string group_name(const std::string &path) {
    const size_t pos = path.find_last_of('/');
    if (pos == std::string::npos || pos == 0) {
        return "/";
    }
    return path.substr(0, pos);
}

std::string shape_label(const std::vector<hsize_t> &dims) {
    if (dims.empty()) {
        return "scalar";
    }
    std::ostringstream out;
    for (size_t i = 0; i < dims.size(); ++i) {
        if (i != 0) {
            out << " x ";
        }
        out << dims[i];
    }
    return out.str();
}

std::string type_label(H5T_class_t type_class) {
    switch (type_class) {
    case H5T_INTEGER:
        return "int";
    case H5T_FLOAT:
        return "float";
    case H5T_STRING:
        return "string";
    case H5T_ARRAY:
        return "array";
    case H5T_COMPOUND:
        return "compound";
    default:
        return "other";
    }
}

std::string layout_label(H5D_layout_t layout) {
    switch (layout) {
    case H5D_COMPACT:
        return "compact";
    case H5D_CONTIGUOUS:
        return "contiguous";
    case H5D_CHUNKED:
        return "chunked";
    case H5D_VIRTUAL:
        return "virtual";
    default:
        return "unknown";
    }
}

std::string filter_label(H5Z_filter_t filter_id, const char *name, const std::vector<unsigned int> &params) {
    std::string label;
    if (name != nullptr && name[0] != '\0') {
        label = name;
    } else {
        switch (filter_id) {
        case H5Z_FILTER_DEFLATE:
            label = "deflate";
            break;
        case H5Z_FILTER_SHUFFLE:
            label = "shuffle";
            break;
        case H5Z_FILTER_FLETCHER32:
            label = "fletcher32";
            break;
        case H5Z_FILTER_SZIP:
            label = "szip";
            break;
        case H5Z_FILTER_NBIT:
            label = "nbit";
            break;
        case H5Z_FILTER_SCALEOFFSET:
            label = "scaleoffset";
            break;
        default:
            label = "filter " + std::to_string(static_cast<int>(filter_id));
            break;
        }
    }

    if (!params.empty()) {
        label += " (";
        for (size_t i = 0; i < params.size(); ++i) {
            if (i != 0) {
                label += ", ";
            }
            label += std::to_string(params[i]);
        }
        label += ")";
    }
    return label;
}

std::string count_label(hsize_t count) {
    if (count < 1000) {
        return std::to_string(count);
    }
    std::ostringstream out;
    out.setf(std::ios::fixed);
    out.precision(count < 1000000 ? 1 : 2);
    if (count < 1000000) {
        out << (static_cast<double>(count) / 1000.0) << "k";
    } else {
        out << (static_cast<double>(count) / 1000000.0) << "M";
    }
    return out.str();
}

std::string bytes_label(long double bytes) {
    static const std::array<const char *, 5> units = {"B", "KiB", "MiB", "GiB", "TiB"};
    size_t unit = 0;
    while (bytes >= 1024.0L && unit + 1 < units.size()) {
        bytes /= 1024.0L;
        ++unit;
    }
    std::ostringstream out;
    out.setf(std::ios::fixed);
    out.precision(unit == 0 ? 0 : 2);
    out << static_cast<double>(bytes) << " " << units[unit];
    return out.str();
}

size_t checked_count(const std::vector<hsize_t> &dims) {
    size_t count = 1;
    for (hsize_t dim : dims) {
        if (dim > std::numeric_limits<size_t>::max() / count) {
            throw std::runtime_error("dataset is too large for this process");
        }
        count *= static_cast<size_t>(dim);
    }
    return count;
}

hsize_t ceil_div(hsize_t value, hsize_t divisor) {
    return (value + divisor - 1) / divisor;
}

TexturePreviewPlan plan_texture_preview(const DatasetInfo &info, int max_texture_side, size_t max_texture_cells) {
    TexturePreviewPlan plan;
    if (info.dims.size() != 2 || info.dims[0] == 0 || info.dims[1] == 0 || max_texture_side <= 0 || max_texture_cells == 0) {
        return plan;
    }

    plan.row_stride = std::max<hsize_t>(1, ceil_div(info.dims[0], static_cast<hsize_t>(max_texture_side)));
    plan.col_stride = std::max<hsize_t>(1, ceil_div(info.dims[1], static_cast<hsize_t>(max_texture_side)));
    const auto update_counts = [&]() {
        plan.rows = 1 + (info.dims[0] - 1) / plan.row_stride;
        plan.cols = 1 + (info.dims[1] - 1) / plan.col_stride;
        plan.cells = plan.rows * plan.cols;
    };
    update_counts();
    while (plan.cells > static_cast<hsize_t>(max_texture_cells)) {
        if (info.dims[0] / plan.row_stride > info.dims[1] / plan.col_stride) {
            ++plan.row_stride;
        } else {
            ++plan.col_stride;
        }
        update_counts();
    }
    plan.valid = true;
    return plan;
}

bool numeric_hdf5_type_class(H5T_class_t type_class) {
    return type_class == H5T_INTEGER || type_class == H5T_FLOAT;
}

bool array_type_info(hid_t type, std::vector<hsize_t> &dims, H5T_class_t &base_class) {
    dims.clear();
    base_class = H5T_NO_CLASS;
    if (H5Tget_class(type) != H5T_ARRAY) {
        return false;
    }

    const int rank = H5Tget_array_ndims(type);
    if (rank <= 0) {
        return false;
    }

    dims.resize(static_cast<size_t>(rank));
    if (H5Tget_array_dims2(type, dims.data()) < 0) {
        dims.clear();
        return false;
    }

    H5Type base_type(H5Tget_super(type));
    if (!base_type.valid()) {
        dims.clear();
        return false;
    }
    base_class = H5Tget_class(base_type.get());
    return true;
}

hsize_t vector_axis_length(const DatasetInfo &info) {
    if (!info.numeric) {
        return 0;
    }
    if (info.dims.size() == 1) {
        return info.dims[0];
    }
    if (info.dims.size() == 2) {
        if (info.dims[0] == 1) {
            return info.dims[1];
        }
        if (info.dims[1] == 1) {
            return info.dims[0];
        }
    }
    return 0;
}

bool contains_case_insensitive(const std::string &haystack, const std::string &needle) {
    if (needle.empty()) {
        return true;
    }
    auto it = std::search(
        haystack.begin(), haystack.end(), needle.begin(), needle.end(),
        [](char lhs, char rhs) {
            return std::tolower(static_cast<unsigned char>(lhs)) ==
                   std::tolower(static_cast<unsigned char>(rhs));
        });
    return it != haystack.end();
}

std::vector<std::string> split_hdf5_path(const std::string &path) {
    std::vector<std::string> parts;
    size_t start = 0;
    while (start < path.size()) {
        while (start < path.size() && path[start] == '/') {
            ++start;
        }
        if (start >= path.size()) {
            break;
        }
        const size_t end = path.find('/', start);
        parts.push_back(path.substr(start, end == std::string::npos ? std::string::npos : end - start));
        if (end == std::string::npos) {
            break;
        }
        start = end + 1;
    }
    return parts;
}

bool looks_like_unix_time_axis(const LoadedDataset &data) {
    const std::string label = data.x_label + " " + data.x_source_path;
    if (contains_case_insensitive(label, "timestamp") || contains_case_insensitive(label, "utime") ||
        contains_case_insensitive(label, "unix")) {
        return true;
    }
    const double low = std::min(data.x_min, data.x_max);
    const double high = std::max(data.x_min, data.x_max);
    return low > 946684800.0 && high < 4102444800.0;
}

int unix_time_formatter(double value, char *buffer, int size, void *user_data) {
    const bool utc = user_data != nullptr && *static_cast<bool *>(user_data);
    const std::time_t seconds = static_cast<std::time_t>(std::llround(value));
    std::tm tm_value{};
    if (utc) {
        gmtime_r(&seconds, &tm_value);
    } else {
        localtime_r(&seconds, &tm_value);
    }

    char time_buffer[64] = {};
    std::strftime(time_buffer, sizeof(time_buffer), "%Y-%m-%d %H:%M:%S", &tm_value);
    return std::snprintf(buffer, static_cast<size_t>(size), "%s", time_buffer);
}

bool preferred_name(const std::string &name, const std::vector<std::string> &preferred) {
    const std::string lowered = lower_ascii(name);
    return std::find(preferred.begin(), preferred.end(), lowered) != preferred.end();
}

DatasetInfo inspect_dataset(hid_t file, const std::string &open_path, const std::string &display_path) {
    H5Dataset dataset(H5Dopen2(file, open_path.c_str(), H5P_DEFAULT));
    if (!dataset.valid()) {
        throw std::runtime_error("failed to open dataset " + display_path);
    }

    H5Space space(H5Dget_space(dataset.get()));
    H5Type type(H5Dget_type(dataset.get()));
    if (!space.valid() || !type.valid()) {
        throw std::runtime_error("failed to inspect dataset " + display_path);
    }

    const int rank = H5Sget_simple_extent_ndims(space.get());
    DatasetInfo info;
    info.path = display_path;
    info.scalar = rank == 0;
    if (rank > 0) {
        info.dims.resize(static_cast<size_t>(rank));
        H5Sget_simple_extent_dims(space.get(), info.dims.data(), nullptr);
    }

    info.type_class = H5Tget_class(type.get());
    info.type_size = H5Tget_size(type.get());
    info.numeric = numeric_hdf5_type_class(info.type_class);
    if (rank == 0 && info.type_class == H5T_ARRAY) {
        std::vector<hsize_t> array_dims;
        H5T_class_t base_class = H5T_NO_CLASS;
        if (array_type_info(type.get(), array_dims, base_class) && array_dims.size() == 1) {
            info.dims = std::move(array_dims);
            info.scalar = false;
            info.numeric = numeric_hdf5_type_class(base_class);
        }
    }
    info.element_count = info.scalar ? 1 : static_cast<hsize_t>(checked_count(info.dims));
    info.storage_size = H5Dget_storage_size(dataset.get());

    H5Prop create_plist(H5Dget_create_plist(dataset.get()));
    if (create_plist.valid()) {
        info.layout = H5Pget_layout(create_plist.get());
        if (info.layout == H5D_CHUNKED && rank > 0) {
            info.chunk_dims.resize(static_cast<size_t>(rank));
            if (H5Pget_chunk(create_plist.get(), rank, info.chunk_dims.data()) < 0) {
                info.chunk_dims.clear();
            }
        }

        const int filter_count = H5Pget_nfilters(create_plist.get());
        for (int filter_index = 0; filter_index < filter_count; ++filter_index) {
            unsigned int flags = 0;
            size_t cd_count = 32;
            std::array<unsigned int, 32> cd_values{};
            char filter_name[128] = {};
            unsigned int filter_config = 0;
            const H5Z_filter_t filter_id =
                H5Pget_filter2(create_plist.get(), static_cast<unsigned>(filter_index), &flags, &cd_count, cd_values.data(),
                               sizeof(filter_name), filter_name, &filter_config);
            if (filter_id >= 0) {
                const size_t used_params = std::min(cd_count, cd_values.size());
                std::vector<unsigned int> params(cd_values.begin(), cd_values.begin() + static_cast<std::ptrdiff_t>(used_params));
                std::string label = filter_label(filter_id, filter_name, params);
                if ((flags & H5Z_FLAG_OPTIONAL) != 0) {
                    label += " optional";
                }
                info.filters.push_back(std::move(label));
            }
        }
    }
    return info;
}

struct VisitState {
    hid_t file = -1;
    std::vector<DatasetInfo> *datasets = nullptr;
};

#if H5_VERSION_GE(1, 12, 0)
herr_t visit_dataset(hid_t, const char *name, const H5O_info2_t *info, void *op_data) {
#else
herr_t visit_dataset(hid_t, const char *name, const H5O_info_t *info, void *op_data) {
#endif
    if (info == nullptr || info->type != H5O_TYPE_DATASET || name == nullptr || std::strcmp(name, ".") == 0) {
        return 0;
    }

    auto *state = static_cast<VisitState *>(op_data);
    const std::string open_path(name);
    const std::string display_path = open_path.empty() || open_path[0] == '/' ? open_path : "/" + open_path;

    try {
        state->datasets->push_back(inspect_dataset(state->file, open_path, display_path));
    } catch (...) {
        return 0;
    }
    return 0;
}

std::vector<DatasetInfo> collect_file_index(hid_t file) {
    std::vector<DatasetInfo> datasets;
    VisitState state{file, &datasets};
#if H5_VERSION_GE(1, 12, 0)
    if (H5Ovisit3(file, H5_INDEX_NAME, H5_ITER_NATIVE, visit_dataset, &state, H5O_INFO_BASIC) < 0) {
#else
    if (H5Ovisit(file, H5_INDEX_NAME, H5_ITER_NATIVE, visit_dataset, &state) < 0) {
#endif
        throw std::runtime_error("failed while scanning HDF5 datasets");
    }

    std::sort(datasets.begin(), datasets.end(), [](const DatasetInfo &lhs, const DatasetInfo &rhs) {
        return lhs.path < rhs.path;
    });
    return datasets;
}

std::vector<DatasetInfo> load_file_index(const std::string &path, bool *opened_swmr = nullptr) {
    std::lock_guard<std::mutex> lock(g_hdf5_mutex);
    H5File file(open_readonly_file(path, false, opened_swmr));
    if (!file.valid()) {
        throw std::runtime_error("could not open HDF5 file: " + path);
    }
    return collect_file_index(file.get());
}

bool read_single_double_selection(hid_t dataset, const std::vector<hsize_t> &start, double &out) {
    H5Space file_space(H5Dget_space(dataset));
    if (!file_space.valid()) {
        return false;
    }
    std::vector<hsize_t> count(start.size(), 1);
    if (H5Sselect_hyperslab(file_space.get(), H5S_SELECT_SET, start.data(), nullptr, count.data(), nullptr) < 0) {
        return false;
    }
    const hsize_t mem_count[1] = {1};
    H5Space mem_space(H5Screate_simple(1, mem_count, nullptr));
    if (!mem_space.valid()) {
        return false;
    }
    return H5Dread(dataset, H5T_NATIVE_DOUBLE, mem_space.get(), file_space.get(), H5P_DEFAULT, &out) >= 0;
}

bool read_one_double(hid_t dataset, hsize_t index, double &out) {
    return read_single_double_selection(dataset, {index}, out);
}

bool read_axis_endpoints(const std::string &file_path, const AxisSpec &axis, double &first, double &last) {
    if (axis.path.empty()) {
        return false;
    }

    H5File file(open_readonly_file(file_path));
    H5Dataset dataset(file.valid() ? H5Dopen2(file.get(), axis.path.c_str(), H5P_DEFAULT) : -1);
    if (!dataset.valid()) {
        return false;
    }
    H5Type type(H5Dget_type(dataset.get()));
    H5Space space(H5Dget_space(dataset.get()));
    if (!type.valid() || !space.valid()) {
        return false;
    }

    if (H5Sget_simple_extent_ndims(space.get()) == 0 && H5Tget_class(type.get()) == H5T_ARRAY) {
        std::vector<hsize_t> array_dims;
        H5T_class_t base_class = H5T_NO_CLASS;
        if (!array_type_info(type.get(), array_dims, base_class) || array_dims.size() != 1 || array_dims[0] == 0 ||
            !numeric_hdf5_type_class(base_class)) {
            return false;
        }
        const size_t count = static_cast<size_t>(array_dims[0]);
        std::vector<double> values(count);
        H5Type mem_type(H5Tarray_create2(H5T_NATIVE_DOUBLE, 1, array_dims.data()));
        if (!mem_type.valid() || H5Dread(dataset.get(), mem_type.get(), H5S_ALL, H5S_ALL, H5P_DEFAULT, values.data()) < 0) {
            return false;
        }
        first = values.front();
        last = values.back();
        return true;
    }

    const int rank = H5Sget_simple_extent_ndims(space.get());
    if (rank == 2) {
        hsize_t dims[2] = {0, 0};
        H5Sget_simple_extent_dims(space.get(), dims, nullptr);
        if (dims[0] == 1 && dims[1] > 0) {
            return read_single_double_selection(dataset.get(), {0, 0}, first) &&
                   read_single_double_selection(dataset.get(), {0, dims[1] - 1}, last);
        }
        if (dims[1] == 1 && dims[0] > 0) {
            return read_single_double_selection(dataset.get(), {0, 0}, first) &&
                   read_single_double_selection(dataset.get(), {dims[0] - 1, 0}, last);
        }
        return false;
    }

    if (rank != 1) {
        return false;
    }

    hsize_t dims[1] = {0};
    H5Sget_simple_extent_dims(space.get(), dims, nullptr);
    if (dims[0] == 0) {
        return false;
    }

    return read_one_double(dataset.get(), 0, first) && read_one_double(dataset.get(), dims[0] - 1, last);
}

std::string read_scalar_text(hid_t file, const DatasetInfo &info) {
    H5Dataset dataset(H5Dopen2(file, info.path.c_str(), H5P_DEFAULT));
    if (!dataset.valid()) {
        throw std::runtime_error("failed to open scalar dataset");
    }

    H5Type type(H5Dget_type(dataset.get()));
    if (!type.valid()) {
        throw std::runtime_error("failed to inspect scalar type");
    }

    if (info.type_class == H5T_STRING) {
        const bool variable = H5Tis_variable_str(type.get()) > 0;
        if (variable) {
            char *raw = nullptr;
            if (H5Dread(dataset.get(), type.get(), H5S_ALL, H5S_ALL, H5P_DEFAULT, &raw) < 0) {
                throw std::runtime_error("failed to read string scalar");
            }
            std::string value = raw != nullptr ? raw : "";
            H5free_memory(raw);
            return value;
        }

        const size_t size = H5Tget_size(type.get());
        std::vector<char> buffer(size + 1, '\0');
        if (H5Dread(dataset.get(), type.get(), H5S_ALL, H5S_ALL, H5P_DEFAULT, buffer.data()) < 0) {
            throw std::runtime_error("failed to read string scalar");
        }
        return std::string(buffer.data());
    }

    if (info.numeric) {
        double value = 0.0;
        if (H5Dread(dataset.get(), H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, &value) < 0) {
            throw std::runtime_error("failed to read numeric scalar");
        }
        std::ostringstream out;
        out.precision(12);
        out << value;
        return out.str();
    }

    return "unsupported scalar type";
}

std::string read_string_dataset_text(hid_t file, const DatasetInfo &info) {
    if (info.type_class != H5T_STRING) {
        return {};
    }

    H5Dataset dataset(H5Dopen2(file, info.path.c_str(), H5P_DEFAULT));
    if (!dataset.valid()) {
        return {};
    }
    H5Type type(H5Dget_type(dataset.get()));
    H5Space space(H5Dget_space(dataset.get()));
    if (!type.valid() || !space.valid()) {
        return {};
    }

    const size_t count = info.scalar ? 1 : static_cast<size_t>(std::min<hsize_t>(info.element_count, 64));
    if (count == 0 || (!info.scalar && info.element_count > 64)) {
        return {};
    }

    std::vector<std::string> parts;
    parts.reserve(count);
    if (H5Tis_variable_str(type.get()) > 0) {
        std::vector<char *> values(count, nullptr);
        if (H5Dread(dataset.get(), type.get(), H5S_ALL, H5S_ALL, H5P_DEFAULT, values.data()) < 0) {
            return {};
        }
        for (char *value : values) {
            if (value != nullptr) {
                std::string part = trim(value);
                if (!part.empty()) {
                    parts.push_back(std::move(part));
                }
            }
        }
        H5Dvlen_reclaim(type.get(), space.get(), H5P_DEFAULT, values.data());
    } else {
        const size_t item_size = H5Tget_size(type.get());
        if (item_size == 0 || item_size > 1'000'000) {
            return {};
        }
        std::vector<char> values(count * item_size, '\0');
        if (H5Dread(dataset.get(), type.get(), H5S_ALL, H5S_ALL, H5P_DEFAULT, values.data()) < 0) {
            return {};
        }
        for (size_t i = 0; i < count; ++i) {
            std::string part(values.data() + i * item_size, item_size);
            const size_t null_pos = part.find('\0');
            if (null_pos != std::string::npos) {
                part.erase(null_pos);
            }
            part = trim(std::move(part));
            if (!part.empty()) {
                parts.push_back(std::move(part));
            }
        }
    }

    std::ostringstream out;
    for (size_t i = 0; i < parts.size(); ++i) {
        if (i != 0) {
            out << "\n";
        }
        out << parts[i];
    }
    return out.str();
}

bool is_comment_dataset(const DatasetInfo &info) {
    const std::string name = lower_ascii(base_name(info.path));
    return name == "comment" || name == "comments";
}

std::string collect_comment_caption(hid_t file, const std::vector<DatasetInfo> &datasets) {
    std::vector<const DatasetInfo *> comments;
    for (const DatasetInfo &info : datasets) {
        if (is_comment_dataset(info)) {
            comments.push_back(&info);
        }
    }
    const auto path_depth = [](const std::string &path) {
        return static_cast<int>(std::count(path.begin(), path.end(), '/'));
    };
    std::sort(comments.begin(), comments.end(), [&](const DatasetInfo *lhs, const DatasetInfo *rhs) {
        const int lhs_depth = path_depth(lhs->path);
        const int rhs_depth = path_depth(rhs->path);
        if (lhs_depth != rhs_depth) {
            return lhs_depth < rhs_depth;
        }
        return lhs->path < rhs->path;
    });

    for (const DatasetInfo *info : comments) {
        if (info == nullptr) {
            continue;
        }
        std::string text = trim(read_string_dataset_text(file, *info));
        if (text.empty()) {
            continue;
        }
        return text;
    }
    return {};
}

std::string load_comment_caption(const std::string &path, const std::vector<DatasetInfo> &datasets) {
    std::lock_guard<std::mutex> lock(g_hdf5_mutex);
    H5File file(open_readonly_file(path));
    if (!file.valid()) {
        return {};
    }
    return collect_comment_caption(file.get(), datasets);
}

void rebuild_dataset_tree(FileTab &tab) {
    tab.dataset_tree = {};
    tab.dataset_tree.name = "/";
    tab.dataset_tree.path = "/";

    for (int index = 0; index < static_cast<int>(tab.datasets.size()); ++index) {
        const DatasetInfo &info = tab.datasets[static_cast<size_t>(index)];
        const std::vector<std::string> parts = split_hdf5_path(info.path);
        if (parts.empty()) {
            continue;
        }

        DatasetTreeNode *node = &tab.dataset_tree;
        std::string current_path;
        for (size_t part_index = 0; part_index < parts.size(); ++part_index) {
            const std::string &part = parts[part_index];
            current_path += "/" + part;
            auto child = std::find_if(node->children.begin(), node->children.end(), [&](const DatasetTreeNode &candidate) {
                return candidate.name == part;
            });
            if (child == node->children.end()) {
                DatasetTreeNode new_node;
                new_node.name = part;
                new_node.path = current_path;
                node->children.push_back(std::move(new_node));
                child = std::prev(node->children.end());
            }
            node = &*child;
        }
        node->dataset_index = index;
    }
}

unsigned cpu_worker_limit() {
    static const unsigned limit = [] {
        const unsigned hw = std::max(1u, std::thread::hardware_concurrency());
        const char *env = std::getenv("HDF5_PLOTTER_MAX_THREADS");
        if (env != nullptr && env[0] != '\0') {
            char *end = nullptr;
            const unsigned long requested = std::strtoul(env, &end, 10);
            if (end != env && requested > 0) {
                return static_cast<unsigned>(std::min<unsigned long>(requested, hw));
            }
        }
        return std::min(4u, hw);
    }();
    return limit;
}

void parallel_for(size_t count, const std::function<void(size_t, size_t)> &fn) {
    if (count == 0) {
        return;
    }
    const size_t worker_count = std::min<size_t>(cpu_worker_limit(), std::max<size_t>(1, count / 262144));
    if (worker_count <= 1) {
        fn(0, count);
        return;
    }

    std::vector<std::thread> workers;
    workers.reserve(worker_count);
    for (size_t worker = 0; worker < worker_count; ++worker) {
        const size_t begin = count * worker / worker_count;
        const size_t end = count * (worker + 1) / worker_count;
        workers.emplace_back([begin, end, &fn]() { fn(begin, end); });
    }
    for (std::thread &worker : workers) {
        worker.join();
    }
}

std::pair<float, float> finite_minmax(const std::vector<float> &values) {
    if (values.empty()) {
        return {0.0f, 1.0f};
    }

    const size_t worker_count = std::min<size_t>(cpu_worker_limit(), std::max<size_t>(1, values.size() / 524288));
    std::vector<float> mins(worker_count, std::numeric_limits<float>::infinity());
    std::vector<float> maxs(worker_count, -std::numeric_limits<float>::infinity());
    std::vector<std::thread> workers;
    workers.reserve(worker_count);

    for (size_t worker = 0; worker < worker_count; ++worker) {
        const size_t begin = values.size() * worker / worker_count;
        const size_t end = values.size() * (worker + 1) / worker_count;
        workers.emplace_back([&, worker, begin, end]() {
        float local_min = std::numeric_limits<float>::infinity();
        float local_max = -std::numeric_limits<float>::infinity();
        for (size_t i = begin; i < end; ++i) {
            const float value = values[i];
            if (std::isfinite(value)) {
                local_min = std::min(local_min, value);
                local_max = std::max(local_max, value);
            }
        }
        mins[worker] = std::min(mins[worker], local_min);
        maxs[worker] = std::max(maxs[worker], local_max);
        });
    }
    for (std::thread &worker : workers) {
        worker.join();
    }

    float min_value = *std::min_element(mins.begin(), mins.end());
    float max_value = *std::max_element(maxs.begin(), maxs.end());
    if (!std::isfinite(min_value) || !std::isfinite(max_value)) {
        return {0.0f, 1.0f};
    }
    if (min_value == max_value) {
        const float pad = std::abs(min_value) > 1.0f ? std::abs(min_value) * 0.01f : 1.0f;
        min_value -= pad;
        max_value += pad;
    }
    return {min_value, max_value};
}

std::pair<double, double> finite_minmax(const std::vector<double> &values) {
    double min_value = std::numeric_limits<double>::infinity();
    double max_value = -std::numeric_limits<double>::infinity();
    for (double value : values) {
        if (std::isfinite(value)) {
            min_value = std::min(min_value, value);
            max_value = std::max(max_value, value);
        }
    }
    if (!std::isfinite(min_value) || !std::isfinite(max_value)) {
        return {0.0, 1.0};
    }
    if (min_value == max_value) {
        const double pad = std::abs(min_value) > 1.0 ? std::abs(min_value) * 0.01 : 1.0;
        min_value -= pad;
        max_value += pad;
    }
    return {min_value, max_value};
}

bool is_monotonic_increasing(const std::vector<double> &values) {
    for (size_t i = 1; i < values.size(); ++i) {
        if (values[i] < values[i - 1]) {
            return false;
        }
    }
    return true;
}

bool is_monotonic_decreasing(const std::vector<double> &values) {
    for (size_t i = 1; i < values.size(); ++i) {
        if (values[i] > values[i - 1]) {
            return false;
        }
    }
    return true;
}

void normalize_range(double &min_value, double &max_value) {
    if (min_value > max_value) {
        std::swap(min_value, max_value);
    }
    if (min_value == max_value) {
        const double pad = std::abs(min_value) > 1.0 ? std::abs(min_value) * 0.01 : 1.0;
        min_value -= pad;
        max_value += pad;
    }
}

std::pair<double, double> effective_range(double min_value, double max_value) {
    if (!std::isfinite(min_value) || !std::isfinite(max_value)) {
        return {0.0, 1.0};
    }
    normalize_range(min_value, max_value);
    return {min_value, max_value};
}

uint8_t to_byte(float value) {
    value = std::clamp(value, 0.0f, 1.0f);
    return static_cast<uint8_t>(std::lround(value * 255.0f));
}

std::array<uint8_t, 4> turbo_rgba(float t) {
    t = std::clamp(t, 0.0f, 1.0f);
    const float t2 = t * t;
    const float t3 = t2 * t;
    const float t4 = t3 * t;
    const float t5 = t4 * t;

    const float r = 0.13572138f + 4.61539260f * t - 42.66032258f * t2 + 132.13108234f * t3 -
                    152.94239396f * t4 + 59.28637943f * t5;
    const float g = 0.09140261f + 2.19418839f * t + 4.84296658f * t2 - 14.18503333f * t3 +
                    4.27729857f * t4 + 2.82956604f * t5;
    const float b = 0.10667330f + 12.64194608f * t - 60.58204836f * t2 + 110.36276771f * t3 -
                    89.90310912f * t4 + 27.34824973f * t5;
    return {to_byte(r), to_byte(g), to_byte(b), 255};
}

std::vector<uint8_t> make_turbo_rgba(const std::vector<float> &values, float min_value, float max_value) {
    std::vector<uint8_t> rgba(values.size() * 4);
    const float denom = max_value - min_value;
    const float safe_denom = denom == 0.0f ? 1.0f : denom;
    parallel_for(values.size(), [&](size_t begin, size_t end) {
        for (size_t i = begin; i < end; ++i) {
            const float value = values[i];
            const float t = std::isfinite(value) ? (value - min_value) / safe_denom : 0.0f;
            const auto color = turbo_rgba(t);
            rgba[i * 4 + 0] = color[0];
            rgba[i * 4 + 1] = color[1];
            rgba[i * 4 + 2] = color[2];
            rgba[i * 4 + 3] = std::isfinite(value) ? color[3] : 0;
        }
    });
    return rgba;
}

template <typename T> std::vector<T> sample_stride(const std::vector<T> &values, hsize_t stride) {
    if (stride <= 1 || values.empty()) {
        return values;
    }
    const size_t count = static_cast<size_t>(1 + (static_cast<hsize_t>(values.size()) - 1) / stride);
    std::vector<T> sampled(count);
    for (size_t i = 0; i < count; ++i) {
        sampled[i] = values[static_cast<size_t>(static_cast<hsize_t>(i) * stride)];
    }
    return sampled;
}

std::vector<float> read_array_dataset_float(const std::string &file_path, const DatasetInfo &info, hsize_t stride) {
    if (info.dims.empty()) {
        return {};
    }

    H5File file(open_readonly_file(file_path));
    H5Dataset dataset(file.valid() ? H5Dopen2(file.get(), info.path.c_str(), H5P_DEFAULT) : -1);
    if (!file.valid() || !dataset.valid()) {
        throw std::runtime_error("failed to open array dataset " + info.path);
    }

    const size_t count = checked_count(info.dims);
    std::vector<float> values(count);
    H5Type mem_type(H5Tarray_create2(H5T_NATIVE_FLOAT, static_cast<unsigned>(info.dims.size()), info.dims.data()));
    if (!mem_type.valid() || H5Dread(dataset.get(), mem_type.get(), H5S_ALL, H5S_ALL, H5P_DEFAULT, values.data()) < 0) {
        throw std::runtime_error("failed to read array dataset as float");
    }
    return sample_stride(values, stride);
}

std::vector<double> read_array_dataset_double(const std::string &file_path, const DatasetInfo &info, hsize_t stride) {
    if (info.dims.empty()) {
        return {};
    }

    H5File file(open_readonly_file(file_path));
    H5Dataset dataset(file.valid() ? H5Dopen2(file.get(), info.path.c_str(), H5P_DEFAULT) : -1);
    if (!file.valid() || !dataset.valid()) {
        throw std::runtime_error("failed to open array dataset " + info.path);
    }

    const size_t count = checked_count(info.dims);
    std::vector<double> values(count);
    H5Type mem_type(H5Tarray_create2(H5T_NATIVE_DOUBLE, static_cast<unsigned>(info.dims.size()), info.dims.data()));
    if (!mem_type.valid() || H5Dread(dataset.get(), mem_type.get(), H5S_ALL, H5S_ALL, H5P_DEFAULT, values.data()) < 0) {
        throw std::runtime_error("failed to read array dataset as double");
    }
    return sample_stride(values, stride);
}

std::vector<double> read_vector_dataset_double(const std::string &file_path, const DatasetInfo &info, hsize_t stride) {
    H5File file(open_readonly_file(file_path));
    H5Dataset dataset(file.valid() ? H5Dopen2(file.get(), info.path.c_str(), H5P_DEFAULT) : -1);
    if (!file.valid() || !dataset.valid()) {
        throw std::runtime_error("failed to open vector axis dataset " + info.path);
    }

    H5Space file_space(H5Dget_space(dataset.get()));
    if (!file_space.valid()) {
        throw std::runtime_error("failed to inspect vector axis dataspace");
    }
    const int rank = H5Sget_simple_extent_ndims(file_space.get());
    if (rank != 2) {
        throw std::runtime_error("axis dataset is not vector-like");
    }

    hsize_t dims[2] = {0, 0};
    H5Sget_simple_extent_dims(file_space.get(), dims, nullptr);
    const bool row_vector = dims[0] == 1 && dims[1] > 0;
    const bool col_vector = dims[1] == 1 && dims[0] > 0;
    if (!row_vector && !col_vector) {
        throw std::runtime_error("axis dataset is not vector-like");
    }

    const hsize_t length = row_vector ? dims[1] : dims[0];
    const hsize_t count = 1 + (length - 1) / stride;
    const hsize_t start[2] = {0, 0};
    const hsize_t stride_v[2] = {col_vector ? stride : 1, row_vector ? stride : 1};
    const hsize_t count_v[2] = {col_vector ? count : 1, row_vector ? count : 1};
    const hsize_t block[2] = {1, 1};
    if (H5Sselect_hyperslab(file_space.get(), H5S_SELECT_SET, start, stride_v, count_v, block) < 0) {
        throw std::runtime_error("failed to select vector axis hyperslab");
    }

    H5Space mem_space(H5Screate_simple(1, &count, nullptr));
    std::vector<double> values(static_cast<size_t>(count));
    if (!mem_space.valid() ||
        H5Dread(dataset.get(), H5T_NATIVE_DOUBLE, mem_space.get(), file_space.get(), H5P_DEFAULT, values.data()) < 0) {
        throw std::runtime_error("failed to read vector axis dataset as double");
    }
    return values;
}

std::vector<float> read_1d_dataset(const std::string &file_path, const DatasetInfo &info, hsize_t stride) {
    if (info.type_class == H5T_ARRAY) {
        return read_array_dataset_float(file_path, info, stride);
    }

    H5File file(open_readonly_file(file_path));
    H5Dataset dataset(file.valid() ? H5Dopen2(file.get(), info.path.c_str(), H5P_DEFAULT) : -1);
    if (!file.valid() || !dataset.valid()) {
        throw std::runtime_error("failed to open dataset " + info.path);
    }

    const hsize_t count = 1 + (info.dims[0] - 1) / stride;
    if (count > static_cast<hsize_t>(std::numeric_limits<size_t>::max())) {
        throw std::runtime_error("1D selection is too large");
    }

    H5Space file_space(H5Dget_space(dataset.get()));
    const hsize_t start[1] = {0};
    const hsize_t stride_v[1] = {stride};
    const hsize_t count_v[1] = {count};
    const hsize_t block[1] = {1};
    if (!file_space.valid() ||
        H5Sselect_hyperslab(file_space.get(), H5S_SELECT_SET, start, stride_v, count_v, block) < 0) {
        throw std::runtime_error("failed to select 1D hyperslab");
    }

    H5Space mem_space(H5Screate_simple(1, count_v, nullptr));
    std::vector<float> values(static_cast<size_t>(count));
    if (!mem_space.valid() ||
        H5Dread(dataset.get(), H5T_NATIVE_FLOAT, mem_space.get(), file_space.get(), H5P_DEFAULT, values.data()) < 0) {
        throw std::runtime_error("failed to read 1D dataset as float");
    }
    return values;
}

std::vector<double> read_1d_dataset_double(const std::string &file_path, const DatasetInfo &info, hsize_t stride) {
    if (info.type_class == H5T_ARRAY) {
        return read_array_dataset_double(file_path, info, stride);
    }
    if (info.dims.size() == 2) {
        return read_vector_dataset_double(file_path, info, stride);
    }

    H5File file(open_readonly_file(file_path));
    H5Dataset dataset(file.valid() ? H5Dopen2(file.get(), info.path.c_str(), H5P_DEFAULT) : -1);
    if (!file.valid() || !dataset.valid()) {
        throw std::runtime_error("failed to open axis dataset " + info.path);
    }

    const hsize_t count = 1 + (info.dims[0] - 1) / stride;
    if (count > static_cast<hsize_t>(std::numeric_limits<size_t>::max())) {
        throw std::runtime_error("axis selection is too large");
    }

    H5Space file_space(H5Dget_space(dataset.get()));
    const hsize_t start[1] = {0};
    const hsize_t stride_v[1] = {stride};
    const hsize_t count_v[1] = {count};
    const hsize_t block[1] = {1};
    if (!file_space.valid() ||
        H5Sselect_hyperslab(file_space.get(), H5S_SELECT_SET, start, stride_v, count_v, block) < 0) {
        throw std::runtime_error("failed to select axis hyperslab");
    }

    H5Space mem_space(H5Screate_simple(1, count_v, nullptr));
    std::vector<double> values(static_cast<size_t>(count));
    if (!mem_space.valid() ||
        H5Dread(dataset.get(), H5T_NATIVE_DOUBLE, mem_space.get(), file_space.get(), H5P_DEFAULT, values.data()) < 0) {
        throw std::runtime_error("failed to read axis dataset as double");
    }
    return values;
}

std::vector<float> read_2d_dataset(const std::string &file_path, const DatasetInfo &info, hsize_t row_stride,
                                   hsize_t col_stride, hsize_t &rows_out, hsize_t &cols_out) {
    H5File file(open_readonly_file(file_path));
    H5Dataset dataset(file.valid() ? H5Dopen2(file.get(), info.path.c_str(), H5P_DEFAULT) : -1);
    if (!file.valid() || !dataset.valid()) {
        throw std::runtime_error("failed to open dataset " + info.path);
    }

    rows_out = 1 + (info.dims[0] - 1) / row_stride;
    cols_out = 1 + (info.dims[1] - 1) / col_stride;
    const size_t rows = static_cast<size_t>(rows_out);
    const size_t cols = static_cast<size_t>(cols_out);
    if (cols != 0 && rows > std::numeric_limits<size_t>::max() / cols) {
        throw std::runtime_error("2D selection is too large");
    }

    H5Space file_space(H5Dget_space(dataset.get()));
    const hsize_t start[2] = {0, 0};
    const hsize_t stride_v[2] = {row_stride, col_stride};
    const hsize_t count_v[2] = {rows_out, cols_out};
    const hsize_t block[2] = {1, 1};
    if (!file_space.valid() ||
        H5Sselect_hyperslab(file_space.get(), H5S_SELECT_SET, start, stride_v, count_v, block) < 0) {
        throw std::runtime_error("failed to select 2D hyperslab");
    }

    H5Space mem_space(H5Screate_simple(2, count_v, nullptr));
    std::vector<float> values(rows * cols);
    if (!mem_space.valid() ||
        H5Dread(dataset.get(), H5T_NATIVE_FLOAT, mem_space.get(), file_space.get(), H5P_DEFAULT, values.data()) < 0) {
        throw std::runtime_error("failed to read 2D dataset as float");
    }
    return values;
}

std::vector<float> transpose_2d_values(const std::vector<float> &values, hsize_t rows, hsize_t cols) {
    const size_t src_rows = static_cast<size_t>(rows);
    const size_t src_cols = static_cast<size_t>(cols);
    std::vector<float> transposed(src_rows * src_cols);
    parallel_for(src_rows, [&](size_t begin, size_t end) {
        for (size_t row = begin; row < end; ++row) {
            for (size_t col = 0; col < src_cols; ++col) {
                transposed[col * src_rows + row] = values[row * src_cols + col];
            }
        }
    });
    return transposed;
}

AxisSpec find_axis(const std::vector<DatasetInfo> &datasets, const DatasetInfo &target, hsize_t dim,
                   const std::vector<std::string> &preferred) {
    const std::string group = group_name(target.path);
    std::vector<const DatasetInfo *> matches;
    std::vector<const DatasetInfo *> preferred_anywhere;
    for (const DatasetInfo &candidate : datasets) {
        if (candidate.path == target.path || vector_axis_length(candidate) != dim) {
            continue;
        }
        if (preferred_name(base_name(candidate.path), preferred)) {
            preferred_anywhere.push_back(&candidate);
        }
        if (group_name(candidate.path) == group) {
            matches.push_back(&candidate);
        }
    }

    auto best = std::find_if(matches.begin(), matches.end(), [&](const DatasetInfo *candidate) {
        return preferred_name(base_name(candidate->path), preferred);
    });
    if (best != matches.end()) {
        return {(*best)->path, base_name((*best)->path)};
    }
    if (matches.size() == 1) {
        best = matches.begin();
        return {(*best)->path, base_name((*best)->path)};
    }

    if (matches.empty() && !preferred_anywhere.empty()) {
        const auto common_prefix_len = [](const std::string &lhs, const std::string &rhs) {
            size_t count = 0;
            const size_t limit = std::min(lhs.size(), rhs.size());
            while (count < limit && lhs[count] == rhs[count]) {
                ++count;
            }
            return count;
        };
        auto preferred_best = std::max_element(preferred_anywhere.begin(), preferred_anywhere.end(), [&](const auto *lhs, const auto *rhs) {
            return common_prefix_len(group_name(lhs->path), group) < common_prefix_len(group_name(rhs->path), group);
        });
        return {(*preferred_best)->path, base_name((*preferred_best)->path)};
    }

    if (matches.size() > 1) {
        return {};
    }
    return {};
}

AxisGuess guess_axes(const std::vector<DatasetInfo> &datasets, const DatasetInfo &target) {
    static const std::vector<std::string> x_preferred = {
        "timestamps", "selected_spectrum_indices", "time", "times", "delay_f", "fftfreqs"};
    static const std::vector<std::string> y_preferred = {
        "wavelength", "wavelengths", "freqs", "frequency", "frequencies", "fftfreqs", "delay_f"};

    AxisGuess axes;
    if (target.dims.size() == 1) {
        axes.x = find_axis(datasets, target, target.dims[0], x_preferred);
    } else if (target.dims.size() == 2) {
        axes.x = find_axis(datasets, target, target.dims[1], x_preferred);
        axes.y = find_axis(datasets, target, target.dims[0], y_preferred);
    }
    return axes;
}

std::vector<int> compatible_axis_indices(const std::vector<DatasetInfo> &datasets, const DatasetInfo &target,
                                         hsize_t required) {
    std::vector<int> indices;
    if (required == 0) {
        return indices;
    }

    for (int i = 0; i < static_cast<int>(datasets.size()); ++i) {
        const DatasetInfo &candidate = datasets[static_cast<size_t>(i)];
        if (candidate.path == target.path || vector_axis_length(candidate) != required) {
            continue;
        }
        indices.push_back(i);
    }
    return indices;
}

std::vector<int> compatible_x_axis_indices(const std::vector<DatasetInfo> &datasets, const DatasetInfo &target,
                                           bool transpose_2d) {
    if (target.dims.empty() || !target.numeric) {
        return {};
    }
    const hsize_t required =
        target.dims.size() == 1 ? target.dims[0] : target.dims.size() == 2 ? target.dims[transpose_2d ? 0 : 1] : 0;
    return compatible_axis_indices(datasets, target, required);
}

std::vector<int> compatible_y_axis_indices(const std::vector<DatasetInfo> &datasets, const DatasetInfo &target,
                                           bool transpose_2d) {
    if (!target.numeric || target.dims.size() != 2) {
        return {};
    }
    return compatible_axis_indices(datasets, target, target.dims[transpose_2d ? 1 : 0]);
}

int find_dataset_index_by_path(const std::vector<DatasetInfo> &datasets, const std::string &path) {
    const auto it = std::find_if(datasets.begin(), datasets.end(), [&](const DatasetInfo &info) {
        return info.path == path;
    });
    return it == datasets.end() ? -1 : static_cast<int>(std::distance(datasets.begin(), it));
}

AxisSpec selected_axis(const std::vector<DatasetInfo> &datasets, int axis_index, hsize_t required_dim, AxisSpec auto_axis) {
    if (axis_index >= 0 && axis_index < static_cast<int>(datasets.size())) {
        const DatasetInfo &axis = datasets[static_cast<size_t>(axis_index)];
        if (vector_axis_length(axis) == required_dim) {
            return {axis.path, base_name(axis.path)};
        }
    }
    if (axis_index == -2) {
        return auto_axis;
    }
    return {};
}

void apply_axis_range(const std::string &file_path, const AxisSpec &axis, double fallback_min, double fallback_max,
                      std::string fallback_label, double &out_min, double &out_max, std::string &out_label) {
    double first = 0.0;
    double last = 0.0;
    if (read_axis_endpoints(file_path, axis, first, last) && std::isfinite(first) && std::isfinite(last) &&
        first != last) {
        out_min = first;
        out_max = last;
        out_label = axis.label.empty() ? fallback_label : axis.label;
    } else {
        out_min = fallback_min;
        out_max = fallback_max;
        out_label = std::move(fallback_label);
    }
    normalize_range(out_min, out_max);
}

LoadResult load_dataset_worker(int token, std::string file_path, DatasetInfo info, AxisSpec x_axis, AxisSpec y_axis,
                               DatasetInfo x_axis_info, bool has_x_axis_info, LoadConfig config) {
    LoadResult result;
    result.token = token;
    result.data = std::make_shared<LoadedDataset>();
    result.data->info = info;

    try {
        std::lock_guard<std::mutex> lock(g_hdf5_mutex);
        if (info.scalar || info.dims.empty()) {
            H5File file(open_readonly_file(file_path));
            if (!file.valid()) {
                throw std::runtime_error("failed to open file");
            }
            result.data->kind = LoadedKind::Scalar;
            result.data->scalar_text = read_scalar_text(file.get(), info);
            result.ok = true;
            return result;
        }

        if (!info.numeric) {
            throw std::runtime_error("selected dataset is not numeric");
        }

        if (info.dims.size() == 1) {
            const hsize_t source_count = info.dims[0];
            hsize_t stride = 1;
            if (source_count > config.max_line_values) {
                stride = ceil_div(source_count, static_cast<hsize_t>(config.max_line_values));
            }

            result.data->kind = LoadedKind::Line1D;
            result.data->source_count = source_count;
            result.data->source_stride = stride;
            result.data->line_values = read_1d_dataset(file_path, info, stride);
            auto [min_value, max_value] = finite_minmax(result.data->line_values);
            result.data->value_min = min_value;
            result.data->value_max = max_value;
            result.data->y_label = base_name(info.path);
            if (has_x_axis_info && !x_axis.path.empty()) {
                result.data->line_x_values = read_1d_dataset_double(file_path, x_axis_info, stride);
                if (result.data->line_x_values.size() == result.data->line_values.size()) {
                    auto [x_min, x_max] = finite_minmax(result.data->line_x_values);
                    result.data->x_min = x_min;
                    result.data->x_max = x_max;
                    result.data->x_label = x_axis.label;
                    result.data->x_source_path = x_axis.path;
                    result.data->x_values_increasing = is_monotonic_increasing(result.data->line_x_values);
                    result.data->x_values_decreasing = is_monotonic_decreasing(result.data->line_x_values);
                } else {
                    result.data->line_x_values.clear();
                    apply_axis_range(file_path, x_axis, 0.0, static_cast<double>(source_count - 1), "index",
                                     result.data->x_min, result.data->x_max, result.data->x_label);
                }
            } else {
                apply_axis_range(file_path, x_axis, 0.0, static_cast<double>(source_count - 1), "index", result.data->x_min,
                                 result.data->x_max, result.data->x_label);
                result.data->x_source_path = x_axis.path;
            }
            if (stride > 1) {
                result.data->note = "Preview sampled every " + std::to_string(stride) + " values from " +
                                    count_label(source_count) + " source values.";
            }
            result.ok = true;
            return result;
        }

        if (info.dims.size() == 2) {
            const TexturePreviewPlan plan = plan_texture_preview(info, config.max_texture_side, config.max_texture_cells);
            if (!plan.valid) {
                throw std::runtime_error("could not plan 2D texture preview");
            }

            hsize_t rows = 0;
            hsize_t cols = 0;
            std::vector<float> values = read_2d_dataset(file_path, info, plan.row_stride, plan.col_stride, rows, cols);
            hsize_t texture_rows = rows;
            hsize_t texture_cols = cols;
            if (config.transpose_2d) {
                values = transpose_2d_values(values, rows, cols);
                texture_rows = cols;
                texture_cols = rows;
            }
            const auto minmax = finite_minmax(values);
            const float min_value = minmax.first;
            const float max_value = minmax.second;
            std::vector<uint8_t> rgba = make_turbo_rgba(values, min_value, max_value);

            result.data->kind = LoadedKind::Heatmap2D;
            result.data->source_rows = info.dims[0];
            result.data->source_cols = info.dims[1];
            result.data->row_stride = plan.row_stride;
            result.data->col_stride = plan.col_stride;
            result.data->texture_height = static_cast<int>(texture_rows);
            result.data->texture_width = static_cast<int>(texture_cols);
            result.data->rgba = std::move(rgba);
            result.data->heat_values = std::move(values);
            result.data->value_min = min_value;
            result.data->value_max = max_value;
            result.data->applied_color_min = min_value;
            result.data->applied_color_max = max_value;
            apply_axis_range(file_path, x_axis, 0.0, static_cast<double>(config.x_fallback_count - 1),
                             config.x_fallback_label, result.data->x_min, result.data->x_max, result.data->x_label);
            apply_axis_range(file_path, y_axis, 0.0, static_cast<double>(config.y_fallback_count - 1),
                             config.y_fallback_label, result.data->y_min, result.data->y_max, result.data->y_label);
            if (plan.row_stride > 1 || plan.col_stride > 1) {
                result.data->note = "Texture preview sampled every " + std::to_string(plan.row_stride) + " row(s) and " +
                                    std::to_string(plan.col_stride) + " column(s).";
            }
            result.ok = true;
            return result;
        }

        throw std::runtime_error("only scalar, 1D, and 2D datasets are supported");
    } catch (const std::exception &ex) {
        result.ok = false;
        result.error = ex.what();
        return result;
    }
}

void delete_heat_texture(FileTab &tab) {
    if (tab.heat_texture != 0) {
        glDeleteTextures(1, &tab.heat_texture);
        tab.heat_texture = 0;
    }
}

void upload_heat_texture(FileTab &tab, LoadedDataset &data) {
    delete_heat_texture(tab);
    if (data.rgba.empty() || data.texture_width <= 0 || data.texture_height <= 0) {
        return;
    }

    glGenTextures(1, &tab.heat_texture);
    glBindTexture(GL_TEXTURE_2D, tab.heat_texture);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, data.texture_width, data.texture_height, 0, GL_RGBA, GL_UNSIGNED_BYTE,
                 data.rgba.data());
    glBindTexture(GL_TEXTURE_2D, 0);
    data.rgba.clear();
    data.rgba.shrink_to_fit();
}

void update_heat_texture_colors(FileTab &tab, LoadedDataset &data, float min_value, float max_value) {
    if (data.kind != LoadedKind::Heatmap2D || data.heat_values.empty() || tab.heat_texture == 0) {
        return;
    }
    const auto nearly_same = [](float lhs, float rhs) {
        const float scale = std::max({1.0f, std::abs(lhs), std::abs(rhs)});
        return std::abs(lhs - rhs) <= scale * 1e-5f;
    };
    if (std::isfinite(data.applied_color_min) && std::isfinite(data.applied_color_max) &&
        nearly_same(data.applied_color_min, min_value) && nearly_same(data.applied_color_max, max_value)) {
        return;
    }

    std::vector<uint8_t> rgba = make_turbo_rgba(data.heat_values, min_value, max_value);
    glBindTexture(GL_TEXTURE_2D, tab.heat_texture);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, data.texture_width, data.texture_height, GL_RGBA, GL_UNSIGNED_BYTE, rgba.data());
    glBindTexture(GL_TEXTURE_2D, 0);
    data.applied_color_min = min_value;
    data.applied_color_max = max_value;
}

void invalidate_plot(FileTab &tab) {
    tab.loaded.reset();
    tab.line_cache = {};
    tab.loading_quiet = false;
    tab.loading_preserve_controls = false;
    delete_heat_texture(tab);
}

void reset_controls_from_data(FileTab &tab, const LoadedDataset &data) {
    tab.controls = {};
    tab.controls.x.min = data.x_min;
    tab.controls.x.max = data.x_max;
    tab.controls.x.automatic = false;
    tab.controls.y.min = data.kind == LoadedKind::Line1D ? data.value_min : data.y_min;
    tab.controls.y.max = data.kind == LoadedKind::Line1D ? data.value_max : data.y_max;
    tab.controls.color.min = data.value_min;
    tab.controls.color.max = data.value_max;
    tab.x_datetime = looks_like_unix_time_axis(data);
    tab.x_datetime_utc = false;
}

PlotControls preserved_controls_for_reload(const FileTab &tab) {
    PlotControls controls = tab.controls;
    if (!tab.loaded) {
        return controls;
    }

    const LoadedDataset &data = *tab.loaded;
    if (controls.x.automatic) {
        controls.x.min = data.x_min;
        controls.x.max = data.x_max;
    }
    if (controls.y.automatic) {
        controls.y.min = data.kind == LoadedKind::Line1D ? data.value_min : data.y_min;
        controls.y.max = data.kind == LoadedKind::Line1D ? data.value_max : data.y_max;
    }
    if (data.kind == LoadedKind::Heatmap2D) {
        if (controls.color.auto_min) {
            controls.color.min = data.value_min;
        }
        if (controls.color.auto_max) {
            controls.color.max = data.value_max;
        }
    }
    normalize_range(controls.x.min, controls.x.max);
    normalize_range(controls.y.min, controls.y.max);
    return controls;
}

std::pair<float, float> effective_color_range(const ColorRangeControl &control, const LoadedDataset &data) {
    const double raw_min = control.auto_min ? static_cast<double>(data.value_min) : control.min;
    const double raw_max = control.auto_max ? static_cast<double>(data.value_max) : control.max;
    const auto [min_value, max_value] = effective_range(raw_min, raw_max);
    return {static_cast<float>(min_value), static_cast<float>(max_value)};
}

void start_load(AppState &app, FileTab &tab, int index, bool keep_axis_choice = false, bool preserve_plot = false,
                bool preserve_controls = false) {
    if (index < 0 || index >= static_cast<int>(tab.datasets.size()) || tab.loading) {
        return;
    }

    if (tab.selected_index != index || !keep_axis_choice) {
        tab.x_axis_index = -2;
        tab.y_axis_index = -2;
    }
    tab.selected_index = index;
    if (preserve_controls) {
        tab.preserved_controls = preserved_controls_for_reload(tab);
    }
    if (!preserve_plot) {
        invalidate_plot(tab);
    }
    const DatasetInfo info = tab.datasets[static_cast<size_t>(index)];
    const AxisGuess guessed_axes = guess_axes(tab.datasets, info);
    AxisSpec x_axis;
    AxisSpec y_axis;
    hsize_t x_fallback_count = 1;
    hsize_t y_fallback_count = 1;
    std::string x_fallback_label = "index";
    std::string y_fallback_label = "value";
    if (info.dims.size() == 1) {
        x_axis = selected_axis(tab.datasets, tab.x_axis_index, info.dims[0], guessed_axes.x);
        x_fallback_count = info.dims[0];
    } else if (info.dims.size() == 2) {
        const bool transposed = tab.transpose_2d;
        x_axis = selected_axis(tab.datasets, tab.x_axis_index, info.dims[transposed ? 0 : 1],
                               transposed ? guessed_axes.y : guessed_axes.x);
        y_axis = selected_axis(tab.datasets, tab.y_axis_index, info.dims[transposed ? 1 : 0],
                               transposed ? guessed_axes.x : guessed_axes.y);
        x_fallback_count = info.dims[transposed ? 0 : 1];
        y_fallback_count = info.dims[transposed ? 1 : 0];
        x_fallback_label = transposed ? "row" : "column";
        y_fallback_label = transposed ? "column" : "row";
    }
    DatasetInfo x_axis_info;
    bool has_x_axis_info = false;
    if (info.dims.size() == 1 && tab.x_axis_index >= 0 && tab.x_axis_index < static_cast<int>(tab.datasets.size())) {
        x_axis_info = tab.datasets[static_cast<size_t>(tab.x_axis_index)];
        has_x_axis_info = true;
    } else if (info.dims.size() == 1 && tab.x_axis_index == -2 && !x_axis.path.empty()) {
        auto it = std::find_if(tab.datasets.begin(), tab.datasets.end(), [&](const DatasetInfo &candidate) {
            return candidate.path == x_axis.path;
        });
        if (it != tab.datasets.end()) {
            x_axis_info = *it;
            has_x_axis_info = true;
        }
    }

    const int token = ++tab.load_token;
    LoadConfig config;
    config.max_texture_side = std::max(256, std::min(app.gl_max_texture_size, 4096));
    config.transpose_2d = info.dims.size() == 2 && tab.transpose_2d;
    config.x_fallback_count = x_fallback_count;
    config.y_fallback_count = y_fallback_count;
    config.x_fallback_label = std::move(x_fallback_label);
    config.y_fallback_label = std::move(y_fallback_label);

    tab.loading = true;
    tab.loading_quiet = preserve_plot;
    tab.loading_preserve_controls = preserve_controls;
    if (!preserve_plot) {
        tab.status = "Loading " + info.path + "...";
    }
    tab.error.clear();
    tab.load_future = std::async(std::launch::async, load_dataset_worker, token, tab.current_file, info, x_axis, y_axis,
                                 x_axis_info, has_x_axis_info, config);
}

bool poll_load(FileTab &tab) {
    if (!tab.loading || !tab.load_future.valid()) {
        return false;
    }

    using namespace std::chrono_literals;
    if (tab.load_future.wait_for(0ms) != std::future_status::ready) {
        return false;
    }

    LoadResult result = tab.load_future.get();
    tab.loading = false;
    const bool quiet_load = tab.loading_quiet;
    tab.loading_quiet = false;
    const bool preserve_controls = tab.loading_preserve_controls;
    const PlotControls preserved_controls = tab.preserved_controls;
    tab.loading_preserve_controls = false;
    if (result.token != tab.load_token) {
        return true;
    }

    if (!result.ok) {
        if (!quiet_load) {
            tab.error = result.error;
            tab.status = "Load failed.";
        }
        return true;
    }

    tab.loaded = std::move(result.data);
    tab.line_cache = {};
    if (tab.loaded) {
        if (preserve_controls) {
            tab.controls = preserved_controls;
            tab.x_datetime = looks_like_unix_time_axis(*tab.loaded);
        } else {
            reset_controls_from_data(tab, *tab.loaded);
        }
    }
    if (tab.loaded && tab.loaded->kind == LoadedKind::Heatmap2D) {
        upload_heat_texture(tab, *tab.loaded);
        if (preserve_controls && (!tab.controls.color.auto_min || !tab.controls.color.auto_max)) {
            const auto [color_min, color_max] = effective_color_range(tab.controls.color, *tab.loaded);
            update_heat_texture_colors(tab, *tab.loaded, color_min, color_max);
        }
    }
    if (!quiet_load) {
        tab.status = tab.loaded ? "Loaded " + tab.loaded->info.path : "Loaded.";
    }
    return true;
}

void open_file_in_tab(AppState &app, FileTab &tab, const std::string &path) {
    if (tab.loading) {
        return;
    }
    if (tab.swmr_polling && tab.swmr_future.valid()) {
        tab.swmr_future.wait();
        tab.swmr_polling = false;
    }

    invalidate_plot(tab);
    tab.datasets.clear();
    rebuild_dataset_tree(tab);
    tab.selected_index = -1;
    tab.x_axis_index = -2;
    tab.y_axis_index = -2;
    tab.transpose_2d = false;
    tab.caption.clear();
    tab.swmr_available = false;
    tab.live_swmr_enabled = false;
    tab.live_status.clear();
    tab.next_swmr_poll = {};
    tab.current_file = path;
    tab.status = "Scanning " + path + "...";
    tab.error.clear();
    copy_to_buffer(tab.file_path, path);

    try {
        bool opened_swmr = false;
        tab.datasets = load_file_index(path, &opened_swmr);
        rebuild_dataset_tree(tab);
        tab.swmr_available = opened_swmr;
        tab.caption = load_comment_caption(path, tab.datasets);
        tab.status = "Found " + std::to_string(tab.datasets.size()) + " datasets.";
        if (opened_swmr) {
            tab.status += " Opened as SWMR reader.";
        }
        auto preferred = std::find_if(tab.datasets.begin(), tab.datasets.end(), [](const DatasetInfo &info) {
            return info.path == "/plot/spectra_image";
        });
        if (preferred == tab.datasets.end()) {
            preferred = std::find_if(tab.datasets.begin(), tab.datasets.end(), [](const DatasetInfo &info) {
                return info.numeric && (info.dims.size() == 2 || info.dims.size() == 1);
            });
        }
        if (preferred != tab.datasets.end()) {
            start_load(app, tab, static_cast<int>(std::distance(tab.datasets.begin(), preferred)));
        }
    } catch (const std::exception &ex) {
        tab.error = ex.what();
        tab.status = "Open failed.";
    }
}

bool can_open_swmr_reader(const std::string &path) {
    std::lock_guard<std::mutex> lock(g_hdf5_mutex);
    bool opened_swmr = false;
    H5File file(open_readonly_file(path, true, &opened_swmr));
    return file.valid() && opened_swmr;
}

void set_live_swmr_enabled(FileTab &tab, bool enabled) {
    if (!enabled) {
        tab.live_swmr_enabled = false;
        tab.live_status = "Live SWMR refresh disabled.";
        return;
    }
    if (tab.current_file.empty()) {
        tab.live_swmr_enabled = false;
        tab.live_status = "Open a file before enabling live SWMR refresh.";
        return;
    }
    if (!can_open_swmr_reader(tab.current_file)) {
        tab.live_swmr_enabled = false;
        tab.swmr_available = false;
        tab.live_status = "No active SWMR writer detected for this file.";
        return;
    }

    tab.swmr_available = true;
    tab.live_swmr_enabled = true;
    tab.next_swmr_poll = std::chrono::steady_clock::now();
    tab.live_status = "Live SWMR refresh enabled.";
}

SwmrPollResult swmr_poll_worker(int token, std::string path, std::string selected_path,
                                std::vector<hsize_t> previous_dims) {
    SwmrPollResult result;
    result.token = token;
    result.selected_path = std::move(selected_path);

    try {
        std::lock_guard<std::mutex> lock(g_hdf5_mutex);
        bool opened_swmr = false;
        H5File file(open_readonly_file(path, true, &opened_swmr));
        if (!file.valid() || !opened_swmr) {
            throw std::runtime_error("file is not currently available as an SWMR reader");
        }

        result.opened_swmr = true;
#if H5_VERSION_GE(1, 10, 0)
        H5Dataset selected_dataset(H5Dopen2(file.get(), result.selected_path.c_str(), H5P_DEFAULT));
        if (selected_dataset.valid()) {
            H5Drefresh(selected_dataset.get());
        }
#endif
        result.datasets = collect_file_index(file.get());
        result.caption = collect_comment_caption(file.get(), result.datasets);

        const int selected_index = find_dataset_index_by_path(result.datasets, result.selected_path);
        if (selected_index < 0) {
            throw std::runtime_error("selected dataset is no longer present");
        }
        result.selected_shape_changed = result.datasets[static_cast<size_t>(selected_index)].dims != previous_dims;
        result.ok = true;
    } catch (const std::exception &ex) {
        result.ok = false;
        result.error = ex.what();
    }
    return result;
}

bool poll_swmr_live(AppState &app, FileTab &tab) {
    using clock = std::chrono::steady_clock;

    if (tab.swmr_polling && tab.swmr_future.valid()) {
        using namespace std::chrono_literals;
        if (tab.swmr_future.wait_for(0ms) == std::future_status::ready) {
            SwmrPollResult result = tab.swmr_future.get();
            tab.swmr_polling = false;
            tab.next_swmr_poll = clock::now() + kSwmrPollInterval;
            if (!tab.live_swmr_enabled || result.token != tab.swmr_token) {
                return true;
            }
            if (!result.ok) {
                tab.live_status = "Live SWMR: " + result.error;
                return true;
            }

            tab.swmr_available = result.opened_swmr;
            if (tab.caption != result.caption) {
                tab.caption = std::move(result.caption);
            }

            if (!result.selected_shape_changed) {
                tab.live_status = "Live SWMR refresh enabled.";
                return true;
            }

            if (tab.loading) {
                tab.live_status = "Live SWMR: update detected while another load is running.";
                return true;
            }

            std::string axis_path;
            if (tab.x_axis_index >= 0 && tab.x_axis_index < static_cast<int>(tab.datasets.size())) {
                axis_path = tab.datasets[static_cast<size_t>(tab.x_axis_index)].path;
            }
            std::string y_axis_path;
            if (tab.y_axis_index >= 0 && tab.y_axis_index < static_cast<int>(tab.datasets.size())) {
                y_axis_path = tab.datasets[static_cast<size_t>(tab.y_axis_index)].path;
            }

            tab.datasets = std::move(result.datasets);
            rebuild_dataset_tree(tab);
            const int selected_index = find_dataset_index_by_path(tab.datasets, result.selected_path);
            if (selected_index < 0) {
                tab.live_status = "Live SWMR: selected dataset disappeared.";
                tab.live_swmr_enabled = false;
                return true;
            }
            if (!axis_path.empty()) {
                const int axis_index = find_dataset_index_by_path(tab.datasets, axis_path);
                tab.x_axis_index = axis_index >= 0 ? axis_index : -2;
            }
            if (!y_axis_path.empty()) {
                const int axis_index = find_dataset_index_by_path(tab.datasets, y_axis_path);
                tab.y_axis_index = axis_index >= 0 ? axis_index : -2;
            }
            tab.selected_index = selected_index;
            tab.live_status = "Live SWMR: update detected.";
            start_load(app, tab, selected_index, true, true, true);
            return true;
        }
    }

    if (!tab.live_swmr_enabled || tab.swmr_polling || tab.loading || tab.selected_index < 0 ||
        tab.selected_index >= static_cast<int>(tab.datasets.size()) || tab.current_file.empty()) {
        return false;
    }

    const clock::time_point now = clock::now();
    if (now < tab.next_swmr_poll) {
        return false;
    }

    const DatasetInfo &selected = tab.datasets[static_cast<size_t>(tab.selected_index)];
    const int token = ++tab.swmr_token;
    tab.swmr_polling = true;
    tab.live_status = "Live SWMR: polling...";
    tab.swmr_future = std::async(std::launch::async, swmr_poll_worker, token, tab.current_file, selected.path, selected.dims);
    return true;
}

double x_at_index(const LoadedDataset &data, size_t index, size_t source_size) {
    if (!data.line_x_values.empty() && index < data.line_x_values.size()) {
        return data.line_x_values[index];
    }
    if (source_size <= 1 || data.x_max == data.x_min) {
        return static_cast<double>(index);
    }
    const double t = static_cast<double>(index) / static_cast<double>(source_size - 1);
    return data.x_min + t * (data.x_max - data.x_min);
}

void rebuild_line_cache(const LoadedDataset &data, LineCache &cache, double view_min, double view_max, int pixel_width) {
    const std::vector<float> &values = data.line_values;
    const size_t n = values.size();
    if (n == 0) {
        cache = {};
        cache.valid = true;
        return;
    }

    if (view_min > view_max) {
        std::swap(view_min, view_max);
    }
    pixel_width = std::max(64, pixel_width);

    const double data_min = std::min(data.x_min, data.x_max);
    const double data_max = std::max(data.x_min, data.x_max);
    view_min = std::clamp(view_min, data_min, data_max);
    view_max = std::clamp(view_max, data_min, data_max);
    if (view_min > view_max) {
        std::swap(view_min, view_max);
    }

    const double tolerance_base = std::max({1.0, std::abs(view_max - view_min), std::abs(data_max - data_min)});
    const double cache_tolerance = tolerance_base * 1e-6;
    const auto nearly_same = [&](double lhs, double rhs) {
        return std::abs(lhs - rhs) <= cache_tolerance;
    };

    const bool cache_hit = cache.valid && cache.source_ptr == values.data() && cache.source_size == n &&
                           cache.x_source_ptr == (data.line_x_values.empty() ? nullptr : data.line_x_values.data()) &&
                           cache.data_x_min == data.x_min && cache.data_x_max == data.x_max &&
                           nearly_same(cache.view_x_min, view_min) && nearly_same(cache.view_x_max, view_max) &&
                           cache.pixel_width == pixel_width;
    if (cache_hit) {
        return;
    }

    cache = {};
    cache.valid = true;
    cache.source_ptr = values.data();
    cache.x_source_ptr = data.line_x_values.empty() ? nullptr : data.line_x_values.data();
    cache.source_size = n;
    cache.data_x_min = data.x_min;
    cache.data_x_max = data.x_max;
    cache.view_x_min = view_min;
    cache.view_x_max = view_max;
    cache.pixel_width = pixel_width;

    const auto index_for_x = [&](double x) -> size_t {
        if (!data.line_x_values.empty() && data.x_values_increasing) {
            auto it = std::lower_bound(data.line_x_values.begin(), data.line_x_values.end(), x);
            if (it == data.line_x_values.end()) {
                return n - 1;
            }
            return static_cast<size_t>(std::distance(data.line_x_values.begin(), it));
        }
        if (!data.line_x_values.empty() && data.x_values_decreasing) {
            auto it = std::lower_bound(data.line_x_values.begin(), data.line_x_values.end(), x, std::greater<double>());
            if (it == data.line_x_values.end()) {
                return n - 1;
            }
            return static_cast<size_t>(std::distance(data.line_x_values.begin(), it));
        }
        if (data.x_max == data.x_min || n <= 1) {
            return 0;
        }
        const double t = (x - data.x_min) / (data.x_max - data.x_min);
        const double clamped = std::clamp(t, 0.0, 1.0);
        return static_cast<size_t>(std::floor(clamped * static_cast<double>(n - 1)));
    };

    size_t first = index_for_x(view_min);
    size_t last = index_for_x(view_max);
    if (first > last) {
        std::swap(first, last);
    }
    last = std::min(last + 1, n - 1);
    const size_t visible_count = last - first + 1;
    const size_t raw_limit = static_cast<size_t>(pixel_width) * 4;

    if (visible_count <= raw_limit) {
        cache.x.resize(visible_count);
        cache.y.resize(visible_count);
        for (size_t i = 0; i < visible_count; ++i) {
            const size_t source_index = first + i;
            cache.x[i] = x_at_index(data, source_index, n);
            cache.y[i] = values[source_index];
        }
        return;
    }

    const size_t bins = std::min<size_t>(std::max(128, pixel_width * 2), 50000);
    cache.x.resize(bins * 2);
    cache.y.resize(bins * 2);

    parallel_for(bins, [&](size_t begin, size_t end) {
        for (size_t bin = begin; bin < end; ++bin) {
            const size_t a = first + visible_count * bin / bins;
            const size_t b = first + visible_count * (bin + 1) / bins;
            float min_value = std::numeric_limits<float>::infinity();
            float max_value = -std::numeric_limits<float>::infinity();
            for (size_t i = a; i < std::max(a + 1, b); ++i) {
                const float value = values[std::min(i, n - 1)];
                if (std::isfinite(value)) {
                    min_value = std::min(min_value, value);
                    max_value = std::max(max_value, value);
                }
            }
            if (!std::isfinite(min_value) || !std::isfinite(max_value)) {
                min_value = 0.0f;
                max_value = 0.0f;
            }
            const double x = x_at_index(data, std::min(a, n - 1), n);
            cache.x[bin * 2 + 0] = x;
            cache.y[bin * 2 + 0] = min_value;
            cache.x[bin * 2 + 1] = x;
            cache.y[bin * 2 + 1] = max_value;
        }
    });
}

ImVec4 high_contrast_line_color(int salt) {
    static const std::array<ImVec4, 8> colors = {
        ImVec4(0.10f, 0.95f, 1.00f, 1.0f), ImVec4(1.00f, 0.88f, 0.10f, 1.0f),
        ImVec4(1.00f, 0.22f, 0.78f, 1.0f), ImVec4(0.35f, 1.00f, 0.30f, 1.0f),
        ImVec4(1.00f, 0.52f, 0.12f, 1.0f), ImVec4(0.72f, 0.56f, 1.00f, 1.0f),
        ImVec4(0.96f, 0.96f, 0.96f, 1.0f), ImVec4(0.30f, 0.68f, 1.00f, 1.0f),
    };
    return colors[static_cast<size_t>(std::abs(salt)) % colors.size()];
}

void draw_color_strip(float min_value, float max_value) {
    const ImVec2 start = ImGui::GetCursorScreenPos();
    const ImVec2 size(std::max(160.0f, ImGui::GetContentRegionAvail().x), 14.0f);
    ImDrawList *draw_list = ImGui::GetWindowDrawList();
    const int steps = 128;
    for (int i = 0; i < steps; ++i) {
        const float t0 = static_cast<float>(i) / static_cast<float>(steps);
        const float t1 = static_cast<float>(i + 1) / static_cast<float>(steps);
        const auto c = turbo_rgba(t0);
        const ImU32 col = IM_COL32(c[0], c[1], c[2], 255);
        draw_list->AddRectFilled(ImVec2(start.x + size.x * t0, start.y), ImVec2(start.x + size.x * t1 + 1.0f, start.y + size.y),
                                 col);
    }
    ImGui::Dummy(size);
    ImGui::Text("%.6g", static_cast<double>(min_value));
    ImGui::SameLine();
    ImGui::SetCursorPosX(ImGui::GetCursorPosX() + std::max(0.0f, size.x - 150.0f));
    ImGui::Text("%.6g", static_cast<double>(max_value));
}

void draw_range_control(const char *label, RangeControl &range, double auto_min, double auto_max) {
    ImGui::PushID(label);
    ImGui::Checkbox(label, &range.automatic);
    ImGui::SameLine();
    const float input_width = ImGui::CalcTextSize("-8.8888e+88").x + ImGui::GetStyle().FramePadding.x * 2.0f;
    if (range.automatic) {
        ImGui::BeginDisabled();
        double min_value = auto_min;
        double max_value = auto_max;
        ImGui::SetNextItemWidth(input_width);
        ImGui::InputDouble("##min", &min_value, 0.0, 0.0, "%.6g");
        ImGui::SameLine();
        ImGui::SetNextItemWidth(input_width);
        ImGui::InputDouble("##max", &max_value, 0.0, 0.0, "%.6g");
        ImGui::EndDisabled();
        range.min = auto_min;
        range.max = auto_max;
    } else {
        bool changed = false;
        ImGui::SetNextItemWidth(input_width);
        changed |= ImGui::InputDouble("##min", &range.min, 0.0, 0.0, "%.6g");
        ImGui::SameLine();
        ImGui::SetNextItemWidth(input_width);
        changed |= ImGui::InputDouble("##max", &range.max, 0.0, 0.0, "%.6g");
        if (changed) {
            normalize_range(range.min, range.max);
        }
    }
    ImGui::PopID();
}

void draw_color_range_control(ColorRangeControl &range, double auto_min, double auto_max) {
    ImGui::PushID("ColorRange");
    const float input_width = ImGui::CalcTextSize("-8.8888e+88").x + ImGui::GetStyle().FramePadding.x * 2.0f;

    if (ImGui::Checkbox("Auto Min", &range.auto_min) && range.auto_min) {
        range.min = auto_min;
    }
    ImGui::SameLine();
    if (range.auto_min) {
        range.min = auto_min;
    }
    ImGui::BeginDisabled(range.auto_min);
    ImGui::SetNextItemWidth(input_width);
    ImGui::InputDouble("##color-min", &range.min, 0.0, 0.0, "%.6g");
    ImGui::EndDisabled();

    ImGui::SameLine();
    ImGui::Spacing();
    ImGui::SameLine();

    if (ImGui::Checkbox("Auto Max", &range.auto_max) && range.auto_max) {
        range.max = auto_max;
    }
    ImGui::SameLine();
    if (range.auto_max) {
        range.max = auto_max;
    }
    ImGui::BeginDisabled(range.auto_max);
    ImGui::SetNextItemWidth(input_width);
    ImGui::InputDouble("##color-max", &range.max, 0.0, 0.0, "%.6g");
    ImGui::EndDisabled();
    ImGui::PopID();
}

void draw_axis_combo(AppState &app, FileTab &tab, const char *label, int &axis_index,
                     const std::vector<int> &compatible, const char *index_label) {
    std::string preview = "Auto";
    if (axis_index == -1) {
        preview = index_label;
    } else if (axis_index >= 0 && axis_index < static_cast<int>(tab.datasets.size())) {
        preview = tab.datasets[static_cast<size_t>(axis_index)].path;
    }

    if (ImGui::BeginCombo(label, preview.c_str())) {
        if (ImGui::Selectable("Auto", axis_index == -2)) {
            axis_index = -2;
            start_load(app, tab, tab.selected_index, true, false, true);
        }
        if (ImGui::Selectable(index_label, axis_index == -1)) {
            axis_index = -1;
            start_load(app, tab, tab.selected_index, true, false, true);
        }
        ImGui::Separator();
        for (int candidate_index : compatible) {
            const DatasetInfo &axis = tab.datasets[static_cast<size_t>(candidate_index)];
            if (ImGui::Selectable(axis.path.c_str(), axis_index == candidate_index)) {
                axis_index = candidate_index;
                start_load(app, tab, tab.selected_index, true, false, true);
            }
        }
        ImGui::EndCombo();
    }
}

void draw_axis_selector(AppState &app, FileTab &tab) {
    if (tab.selected_index < 0 || tab.selected_index >= static_cast<int>(tab.datasets.size())) {
        return;
    }
    const DatasetInfo &target = tab.datasets[static_cast<size_t>(tab.selected_index)];
    if (!target.numeric || (target.dims.size() != 1 && target.dims.size() != 2)) {
        return;
    }

    const bool blocking_load = tab.loading && !tab.loading_quiet;
    ImGui::BeginDisabled(blocking_load);
    if (target.dims.size() == 2) {
        bool transpose = tab.transpose_2d;
        if (ImGui::Checkbox("Flip X/Y", &transpose)) {
            tab.transpose_2d = transpose;
            tab.x_axis_index = -2;
            tab.y_axis_index = -2;
            start_load(app, tab, tab.selected_index, true, false, true);
        }
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("Transpose 2D datasets so rows are shown on X and columns on Y.");
        }
    }
    const std::vector<int> compatible_x = compatible_x_axis_indices(tab.datasets, target, tab.transpose_2d);
    draw_axis_combo(app, tab, "X axis", tab.x_axis_index, compatible_x,
                    target.dims.size() == 1 ? "Index" : tab.transpose_2d ? "Row" : "Column");
    if (target.dims.size() == 2) {
        const std::vector<int> compatible_y = compatible_y_axis_indices(tab.datasets, target, tab.transpose_2d);
        draw_axis_combo(app, tab, "Y axis", tab.y_axis_index, compatible_y, tab.transpose_2d ? "Column" : "Row");
    }
    ImGui::EndDisabled();
}

std::string dataset_row_label(const DatasetInfo &info, bool full_path) {
    const std::string name = full_path ? info.path : base_name(info.path);
    return name + "  [" + shape_label(info.dims) + ", " + type_label(info.type_class) + "]";
}

void draw_dataset_tooltip(const DatasetInfo &info) {
    if (!ImGui::IsItemHovered()) {
        return;
    }

    ImGui::BeginTooltip();
    ImGui::TextUnformatted(info.path.c_str());
    ImGui::Text("shape: %s", shape_label(info.dims).c_str());
    ImGui::Text("type: %s, values: %s", type_label(info.type_class).c_str(), count_label(info.element_count).c_str());
    ImGui::EndTooltip();
}

void select_dataset_from_browser(AppState &app, FileTab &tab, int index) {
    start_load(app, tab, index);
    app.active_tab_id = tab.id;
    app.plot_focus_request_id = tab.id;
}

void draw_dataset_leaf(AppState &app, FileTab &tab, int index, bool full_path) {
    if (index < 0 || index >= static_cast<int>(tab.datasets.size())) {
        return;
    }

    const DatasetInfo &info = tab.datasets[static_cast<size_t>(index)];
    const std::string label = dataset_row_label(info, full_path);
    ImGui::PushID(index);
    ImGui::BeginDisabled(tab.loading && !tab.loading_quiet);
    if (ImGui::Selectable(label.c_str(), index == tab.selected_index, ImGuiSelectableFlags_SpanAvailWidth)) {
        select_dataset_from_browser(app, tab, index);
    }
    ImGui::EndDisabled();
    draw_dataset_tooltip(info);
    ImGui::PopID();
}

bool tree_contains_path(const DatasetTreeNode &node, const std::string &dataset_path) {
    if (dataset_path == node.path) {
        return true;
    }
    const std::string prefix = node.path == "/" ? "/" : node.path + "/";
    return dataset_path.size() > prefix.size() && dataset_path.compare(0, prefix.size(), prefix) == 0;
}

void draw_dataset_tree_node(AppState &app, FileTab &tab, const DatasetTreeNode &node) {
    if (node.children.empty()) {
        draw_dataset_leaf(app, tab, node.dataset_index, false);
        return;
    }

    std::string selected_path;
    if (tab.selected_index >= 0 && tab.selected_index < static_cast<int>(tab.datasets.size())) {
        selected_path = tab.datasets[static_cast<size_t>(tab.selected_index)].path;
    }
    const bool contains_selected = !selected_path.empty() && tree_contains_path(node, selected_path);
    ImGuiTreeNodeFlags flags = ImGuiTreeNodeFlags_OpenOnArrow | ImGuiTreeNodeFlags_OpenOnDoubleClick |
                               ImGuiTreeNodeFlags_SpanAvailWidth;
    if (contains_selected) {
        ImGui::SetNextItemOpen(true, ImGuiCond_Always);
    }

    ImGui::PushID(node.path.c_str());
    const bool open = ImGui::TreeNodeEx(node.name.c_str(), flags);
    if (open) {
        for (const DatasetTreeNode &child : node.children) {
            draw_dataset_tree_node(app, tab, child);
        }
        ImGui::TreePop();
    }
    ImGui::PopID();
}

void draw_dataset_panel(AppState &app, FileTab &tab) {
    ImGui::InputText("File", tab.file_path.data(), tab.file_path.size());
    ImGui::BeginDisabled(tab.loading && !tab.loading_quiet);
    if (ImGui::Button("Reload", ImVec2(-1, 0))) {
        const std::string path = trim(tab.file_path.data());
        if (!path.empty()) {
            open_file_in_tab(app, tab, path);
        }
    }
    ImGui::EndDisabled();
    if (ImGui::Button("File Details", ImVec2(-1, 0))) {
        tab.show_file_details = true;
    }

    ImGui::Separator();
    ImGui::SetNextItemWidth(-1.0f);
    ImGui::InputTextWithHint("##dataset-search", "Search datasets", tab.filter.data(), tab.filter.size());
    ImGui::TextUnformatted(tab.status.c_str());
    if (!tab.error.empty()) {
        ImGui::TextColored(ImVec4(1.0f, 0.35f, 0.25f, 1.0f), "%s", tab.error.c_str());
    }
    bool live_swmr = tab.live_swmr_enabled;
    if (ImGui::Checkbox("Live SWMR refresh", &live_swmr)) {
        set_live_swmr_enabled(tab, live_swmr);
    }
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("Poll an active SWMR writer and reload the selected plot when its dataset extent changes.");
    }
    draw_axis_selector(app, tab);

    ImGui::BeginChild("dataset-list", ImVec2(0, 0), true);
    const std::string filter = trim(tab.filter.data());
    if (filter.empty()) {
        for (const DatasetTreeNode &child : tab.dataset_tree.children) {
            draw_dataset_tree_node(app, tab, child);
        }
    } else {
        std::vector<int> visible_indices;
        visible_indices.reserve(tab.datasets.size());
        for (int i = 0; i < static_cast<int>(tab.datasets.size()); ++i) {
            const DatasetInfo &info = tab.datasets[static_cast<size_t>(i)];
            if (contains_case_insensitive(info.path, filter)) {
                visible_indices.push_back(i);
            }
        }
        ImGuiListClipper clipper;
        clipper.Begin(static_cast<int>(visible_indices.size()));
        while (clipper.Step()) {
            for (int visible = clipper.DisplayStart; visible < clipper.DisplayEnd; ++visible) {
                draw_dataset_leaf(app, tab, visible_indices[static_cast<size_t>(visible)], true);
            }
        }
    }
    ImGui::EndChild();
}

void draw_plot_controls(FileTab &tab, LoadedDataset &data) {
    if (data.kind == LoadedKind::Scalar) {
        return;
    }

    if (ImGui::CollapsingHeader("Scaling", ImGuiTreeNodeFlags_DefaultOpen)) {
        draw_range_control("Auto X", tab.controls.x, data.x_min, data.x_max);
        const double y_auto_min = data.kind == LoadedKind::Line1D ? data.value_min : data.y_min;
        const double y_auto_max = data.kind == LoadedKind::Line1D ? data.value_max : data.y_max;
        draw_range_control("Auto Y", tab.controls.y, y_auto_min, y_auto_max);
        if (data.kind == LoadedKind::Heatmap2D) {
            draw_color_range_control(tab.controls.color, data.value_min, data.value_max);
        }
        if (looks_like_unix_time_axis(data)) {
            ImGui::Checkbox("Format X as date/time", &tab.x_datetime);
            if (tab.x_datetime) {
                ImGui::SameLine();
                ImGui::Checkbox("UTC", &tab.x_datetime_utc);
            }
        }
    }
}

void draw_loaded_plot(const AppState &app, FileTab &tab) {
    if (app.show_plot_captions && !tab.caption.empty()) {
        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.82f, 0.86f, 0.90f, 1.0f));
        ImGui::TextWrapped("%s", tab.caption.c_str());
        ImGui::PopStyleColor();
        ImGui::Spacing();
    }

    if (tab.loading && !tab.loaded) {
        ImGui::TextUnformatted("Loading...");
        return;
    }

    if (!tab.loaded) {
        ImGui::TextUnformatted("Open an HDF5 file and select a numeric dataset.");
        return;
    }

    LoadedDataset &data = *tab.loaded;
    if (tab.loading && !tab.loading_quiet) {
        ImGui::TextDisabled("%s", tab.status.c_str());
    }
    ImGui::Text("%s  [%s, %s]", data.info.path.c_str(), shape_label(data.info.dims).c_str(),
                type_label(data.info.type_class).c_str());
    if (!data.note.empty()) {
        ImGui::TextColored(ImVec4(0.78f, 0.78f, 0.55f, 1.0f), "%s", data.note.c_str());
    }
    ImGui::Separator();

    if (data.kind == LoadedKind::Scalar) {
        ImGui::TextWrapped("%s", data.scalar_text.c_str());
        return;
    }

    draw_plot_controls(tab, data);
    ImGui::Separator();

    if (data.kind == LoadedKind::Line1D) {
        const int plot_width = static_cast<int>(std::max(64.0f, ImGui::GetContentRegionAvail().x));
        if (tab.controls.x.automatic) {
            ImPlot::SetNextAxisToFit(ImAxis_X1);
        } else {
            ImPlot::SetNextAxisLinks(ImAxis_X1, &tab.controls.x.min, &tab.controls.x.max);
        }
        if (tab.controls.y.automatic) {
            ImPlot::SetNextAxisToFit(ImAxis_Y1);
        } else {
            ImPlot::SetNextAxisLinks(ImAxis_Y1, &tab.controls.y.min, &tab.controls.y.max);
        }
        if (ImPlot::BeginPlot("##line", ImVec2(-1, -1))) {
            ImPlot::SetupAxes(data.x_label.c_str(), data.y_label.c_str());
            if (tab.x_datetime && looks_like_unix_time_axis(data)) {
                ImPlot::SetupAxisFormat(ImAxis_X1, unix_time_formatter, &tab.x_datetime_utc);
            }
            const ImPlotRect limits = ImPlot::GetPlotLimits();
            const double cache_min = tab.controls.x.automatic ? data.x_min : limits.X.Min;
            const double cache_max = tab.controls.x.automatic ? data.x_max : limits.X.Max;
            rebuild_line_cache(data, tab.line_cache, cache_min, cache_max, plot_width);
            if (!tab.line_cache.x.empty()) {
                ImPlot::SetNextLineStyle(high_contrast_line_color(tab.selected_index), 1.8f);
                ImPlot::PlotLine(base_name(data.info.path).c_str(), tab.line_cache.x.data(), tab.line_cache.y.data(),
                                 static_cast<int>(tab.line_cache.x.size()));
            }
            ImPlot::EndPlot();
        }
        return;
    }

    if (data.kind == LoadedKind::Heatmap2D) {
        const auto [color_min, color_max] = effective_color_range(tab.controls.color, data);
        update_heat_texture_colors(tab, data, color_min, color_max);
        ImGui::Text("Turbo texture: %d x %d from %s x %s cells, range %.6g to %.6g", data.texture_width,
                    data.texture_height, count_label(data.source_rows).c_str(), count_label(data.source_cols).c_str(),
                    static_cast<double>(color_min), static_cast<double>(color_max));
        draw_color_strip(color_min, color_max);

        if (tab.controls.x.automatic) {
            ImPlot::SetNextAxisToFit(ImAxis_X1);
        } else {
            ImPlot::SetNextAxisLinks(ImAxis_X1, &tab.controls.x.min, &tab.controls.x.max);
        }
        if (tab.controls.y.automatic) {
            ImPlot::SetNextAxisToFit(ImAxis_Y1);
        } else {
            ImPlot::SetNextAxisLinks(ImAxis_Y1, &tab.controls.y.min, &tab.controls.y.max);
        }

        if (tab.heat_texture != 0 && ImPlot::BeginPlot("##heatmap", ImVec2(-1, -1))) {
            ImPlot::SetupAxes(data.x_label.c_str(), data.y_label.c_str());
            if (tab.x_datetime && looks_like_unix_time_axis(data)) {
                ImPlot::SetupAxisFormat(ImAxis_X1, unix_time_formatter, &tab.x_datetime_utc);
            }
            ImPlot::PlotImage(base_name(data.info.path).c_str(), reinterpret_cast<ImTextureID>(static_cast<intptr_t>(tab.heat_texture)),
                              ImPlotPoint(data.x_min, data.y_min), ImPlotPoint(data.x_max, data.y_max), ImVec2(0, 0),
                              ImVec2(1, 1));
            ImPlot::EndPlot();
        }
    }
}

std::string tab_title(const FileTab &tab) {
    if (!tab.current_file.empty()) {
        return std::filesystem::path(tab.current_file).filename().string();
    }
    return "Untitled";
}

std::string plot_window_title(const FileTab &tab) {
    return "Plot: " + tab_title(tab) + "##plot-" + std::to_string(tab.id);
}

FileTab &create_tab(AppState &app) {
    auto tab = std::make_unique<FileTab>();
    tab->id = app.next_tab_id++;
    FileTab &ref = *tab;
    app.active_tab_id = ref.id;
    app.plot_focus_request_id = ref.id;
    app.tabs.push_back(std::move(tab));
    return ref;
}

FileTab *find_tab(AppState &app, int id) {
    for (auto &tab : app.tabs) {
        if (tab->id == id) {
            return tab.get();
        }
    }
    return nullptr;
}

void close_tab(AppState &app, int id) {
    auto it = std::find_if(app.tabs.begin(), app.tabs.end(), [&](const std::unique_ptr<FileTab> &tab) {
        return tab->id == id;
    });
    if (it == app.tabs.end()) {
        return;
    }
    FileTab &tab = **it;
    if (tab.loading && tab.load_future.valid()) {
        tab.load_future.wait();
    }
    if (tab.swmr_polling && tab.swmr_future.valid()) {
        tab.swmr_future.wait();
    }
    delete_heat_texture(tab);
    app.tabs.erase(it);
    if (app.active_tab_id == id) {
        app.active_tab_id = app.tabs.empty() ? 0 : app.tabs.front()->id;
        app.plot_focus_request_id = app.active_tab_id;
    }
    if (app.plot_focus_request_id == id) {
        app.plot_focus_request_id = 0;
    }
}

void open_path_as_tab(AppState &app, const std::string &path) {
    const std::string normalized = std::filesystem::absolute(path).lexically_normal().string();
    for (auto &tab : app.tabs) {
        if (!tab->current_file.empty() &&
            std::filesystem::absolute(tab->current_file).lexically_normal().string() == normalized) {
            app.active_tab_id = tab->id;
            app.plot_focus_request_id = tab->id;
            return;
        }
    }
    FileTab &tab = create_tab(app);
    copy_to_buffer(tab.file_path, normalized);
    open_file_in_tab(app, tab, normalized);
}

void request_file_picker(AppState &app) {
    FilePicker &picker = app.picker;
    picker.request_open = true;
    picker.error.clear();
    if (picker.directory.empty()) {
        if (FileTab *tab = find_tab(app, app.active_tab_id); tab != nullptr && !tab->current_file.empty()) {
            picker.directory = std::filesystem::path(tab->current_file).parent_path();
        } else {
            picker.directory = std::filesystem::current_path();
        }
    }
    copy_to_buffer(picker.selected_path, picker.directory.string());
}

bool toggle_axis_autoscale(FileTab &tab, bool x_axis) {
    if (!tab.loaded || tab.loaded->kind == LoadedKind::Scalar) {
        return false;
    }

    LoadedDataset &data = *tab.loaded;
    RangeControl &range = x_axis ? tab.controls.x : tab.controls.y;
    range.automatic = !range.automatic;
    if (range.automatic) {
        if (x_axis) {
            range.min = data.x_min;
            range.max = data.x_max;
        } else {
            range.min = data.kind == LoadedKind::Line1D ? data.value_min : data.y_min;
            range.max = data.kind == LoadedKind::Line1D ? data.value_max : data.y_max;
        }
    }
    return true;
}

bool flip_active_2d_axes(AppState &app, FileTab &tab) {
    if (tab.selected_index < 0 || tab.selected_index >= static_cast<int>(tab.datasets.size()) || tab.loading) {
        return false;
    }
    const DatasetInfo &target = tab.datasets[static_cast<size_t>(tab.selected_index)];
    if (!target.numeric || target.dims.size() != 2) {
        return false;
    }

    tab.transpose_2d = !tab.transpose_2d;
    tab.x_axis_index = -2;
    tab.y_axis_index = -2;
    start_load(app, tab, tab.selected_index, true, false, true);
    return true;
}

bool toggle_live_swmr(FileTab &tab) {
    set_live_swmr_enabled(tab, !tab.live_swmr_enabled);
    return true;
}

bool handle_keyboard_shortcuts(AppState &app) {
    ImGuiIO &io = ImGui::GetIO();
    const bool text_entry_active = io.WantTextInput || ImGui::IsAnyItemActive();
    const bool open_requested = ImGui::IsKeyPressed(ImGuiKey_O, false) && ((io.KeyCtrl || io.KeySuper) || !text_entry_active);
    if (open_requested && !app.picker.visible) {
        request_file_picker(app);
        return true;
    }

    if (text_entry_active || app.picker.visible) {
        return false;
    }

    FileTab *tab = find_tab(app, app.active_tab_id);
    if (tab == nullptr) {
        return false;
    }
    if (ImGui::IsKeyPressed(ImGuiKey_X, false)) {
        return toggle_axis_autoscale(*tab, true);
    }
    if (ImGui::IsKeyPressed(ImGuiKey_Y, false)) {
        return toggle_axis_autoscale(*tab, false);
    }
    if (ImGui::IsKeyPressed(ImGuiKey_F, false)) {
        return flip_active_2d_axes(app, *tab);
    }
    if (ImGui::IsKeyPressed(ImGuiKey_Space, false)) {
        return toggle_live_swmr(*tab);
    }
    return false;
}

void draw_file_picker(AppState &app) {
    FilePicker &picker = app.picker;
    if (picker.request_open) {
        ImGui::OpenPopup("Open HDF5 file");
        picker.visible = true;
        picker.request_open = false;
    }

    if (!ImGui::BeginPopupModal("Open HDF5 file", &picker.visible, ImGuiWindowFlags_AlwaysAutoResize)) {
        return;
    }

    ImGui::TextUnformatted(picker.directory.string().c_str());
    ImGui::InputText("Path", picker.selected_path.data(), picker.selected_path.size());
    if (ImGui::Button("Up")) {
        std::filesystem::path parent = picker.directory.parent_path();
        if (!parent.empty()) {
            picker.directory = parent;
            copy_to_buffer(picker.selected_path, picker.directory.string());
        }
    }
    ImGui::SameLine();
    if (ImGui::Button("Open Path")) {
        const std::string path = trim(picker.selected_path.data());
        if (!path.empty()) {
            open_path_as_tab(app, path);
            picker.visible = false;
            ImGui::CloseCurrentPopup();
        }
    }
    if (!picker.error.empty()) {
        ImGui::TextColored(ImVec4(1.0f, 0.35f, 0.25f, 1.0f), "%s", picker.error.c_str());
    }

    std::vector<std::filesystem::directory_entry> dirs;
    std::vector<std::filesystem::directory_entry> files;
    std::error_code ec;
    for (const auto &entry : std::filesystem::directory_iterator(picker.directory, ec)) {
        if (entry.is_directory()) {
            dirs.push_back(entry);
        } else if (entry.is_regular_file()) {
            const std::string ext = entry.path().extension().string();
            if (ext == ".h5" || ext == ".hdf5" || ext == ".H5" || ext == ".HDF5") {
                files.push_back(entry);
            }
        }
    }
    if (ec) {
        picker.error = ec.message();
    }
    const auto by_name = [](const auto &lhs, const auto &rhs) {
        return lhs.path().filename().string() < rhs.path().filename().string();
    };
    std::sort(dirs.begin(), dirs.end(), by_name);
    std::sort(files.begin(), files.end(), by_name);

    ImGui::BeginChild("files", ImVec2(720, 420), true);
    for (const auto &dir : dirs) {
        const std::string label = "[dir] " + dir.path().filename().string();
        if (ImGui::Selectable(label.c_str(), false, ImGuiSelectableFlags_AllowDoubleClick)) {
            picker.directory = dir.path();
            copy_to_buffer(picker.selected_path, picker.directory.string());
        }
        if (ImGui::IsItemHovered() && ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left)) {
            picker.directory = dir.path();
            copy_to_buffer(picker.selected_path, picker.directory.string());
        }
    }
    for (const auto &file : files) {
        const std::string label = file.path().filename().string();
        if (ImGui::Selectable(label.c_str(), trim(picker.selected_path.data()) == file.path().string(),
                              ImGuiSelectableFlags_AllowDoubleClick)) {
            copy_to_buffer(picker.selected_path, file.path().string());
        }
        if (ImGui::IsItemHovered() && ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left)) {
            open_path_as_tab(app, file.path().string());
            picker.visible = false;
            ImGui::CloseCurrentPopup();
        }
    }
    ImGui::EndChild();

    if (ImGui::Button("Open Selected")) {
        const std::string path = trim(picker.selected_path.data());
        if (!path.empty()) {
            open_path_as_tab(app, path);
            picker.visible = false;
            ImGui::CloseCurrentPopup();
        }
    }
    ImGui::SameLine();
    if (ImGui::Button("Cancel")) {
        picker.visible = false;
        ImGui::CloseCurrentPopup();
    }
    ImGui::EndPopup();
}

void draw_performance_hud(AppState &app) {
    if (!app.show_performance_hud) {
        return;
    }

    ImGui::SetNextWindowSize(ImVec2(520, 0), ImGuiCond_FirstUseEver);
    if (!ImGui::Begin("Performance")) {
        ImGui::End();
        return;
    }

    const ImGuiIO &io = ImGui::GetIO();
    ImGui::Text("Frame: %.2f ms (%.1f FPS)", app.last_frame_ms, io.Framerate);
    ImGui::Text("Idle wait: %s", app.last_used_idle_wait ? "yes" : "no");
    ImGui::Text("Fast frame: %s", app.last_fast_frame ? "yes" : "no");
    ImGui::Text("Background work: %s", app.last_background_work ? "yes" : "no");
    ImGui::Text("ImGui active item/mouse: %s", app.last_ui_active ? "yes" : "no");
    ImGui::Text("SDL events last frame: %d", app.last_event_count);
    ImGui::Separator();
    ImGui::TextWrapped("GL vendor: %s", app.gl_vendor.c_str());
    ImGui::TextWrapped("GL renderer: %s", app.gl_renderer.c_str());
    ImGui::TextWrapped("GL version: %s", app.gl_version.c_str());
    ImGui::TextWrapped("GLSL: %s", app.glsl_version.c_str());
    ImGui::Text("UI font framebuffer scale: %.2f", app.ui_framebuffer_scale);
    ImGui::Text("Max texture size: %d", app.gl_max_texture_size);
    ImGui::End();
}

std::string join_labels(const std::vector<std::string> &values, const char *empty_label = "none") {
    if (values.empty()) {
        return empty_label;
    }
    std::string joined;
    for (size_t i = 0; i < values.size(); ++i) {
        if (i != 0) {
            joined += ", ";
        }
        joined += values[i];
    }
    return joined;
}

long double logical_dataset_bytes(const DatasetInfo &info) {
    return static_cast<long double>(info.element_count) * static_cast<long double>(info.type_size);
}

hsize_t distinct_chunks_for_selection(hsize_t dim, hsize_t stride, hsize_t chunk_dim) {
    if (dim == 0 || stride == 0 || chunk_dim == 0) {
        return 0;
    }

    hsize_t chunks = 0;
    hsize_t previous_chunk = std::numeric_limits<hsize_t>::max();
    const hsize_t count = 1 + (dim - 1) / stride;
    for (hsize_t index = 0; index < count; ++index) {
        const hsize_t source_index = std::min(dim - 1, index * stride);
        const hsize_t chunk = source_index / chunk_dim;
        if (chunk != previous_chunk) {
            ++chunks;
            previous_chunk = chunk;
        }
    }
    return chunks;
}

void draw_dataset_diagnostics_table(const FileTab &tab, int max_texture_side, size_t max_texture_cells) {
    if (tab.datasets.empty()) {
        ImGui::TextUnformatted("No datasets indexed.");
        return;
    }

    if (!ImGui::BeginTable("dataset-diagnostics", 9,
                           ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg | ImGuiTableFlags_Resizable |
                               ImGuiTableFlags_ScrollY,
                           ImVec2(0, 360))) {
        return;
    }

    ImGui::TableSetupColumn("Dataset");
    ImGui::TableSetupColumn("Shape");
    ImGui::TableSetupColumn("Type");
    ImGui::TableSetupColumn("Layout");
    ImGui::TableSetupColumn("Chunk");
    ImGui::TableSetupColumn("Filters");
    ImGui::TableSetupColumn("Logical");
    ImGui::TableSetupColumn("Stored");
    ImGui::TableSetupColumn("Preview");
    ImGui::TableHeadersRow();

    for (const DatasetInfo &info : tab.datasets) {
        ImGui::TableNextRow();
        ImGui::TableNextColumn();
        ImGui::TextUnformatted(info.path.c_str());
        ImGui::TableNextColumn();
        ImGui::TextUnformatted(shape_label(info.dims).c_str());
        ImGui::TableNextColumn();
        ImGui::Text("%s/%zu", type_label(info.type_class).c_str(), info.type_size);
        ImGui::TableNextColumn();
        ImGui::TextUnformatted(layout_label(info.layout).c_str());
        ImGui::TableNextColumn();
        ImGui::TextUnformatted(info.chunk_dims.empty() ? "-" : shape_label(info.chunk_dims).c_str());
        ImGui::TableNextColumn();
        ImGui::TextWrapped("%s", join_labels(info.filters).c_str());
        ImGui::TableNextColumn();
        ImGui::TextUnformatted(bytes_label(logical_dataset_bytes(info)).c_str());
        ImGui::TableNextColumn();
        ImGui::TextUnformatted(bytes_label(static_cast<long double>(info.storage_size)).c_str());
        ImGui::TableNextColumn();
        if (info.dims.size() == 2 && info.numeric) {
            const TexturePreviewPlan plan = plan_texture_preview(info, max_texture_side, max_texture_cells);
            if (plan.valid) {
                ImGui::Text("%s x %s, stride %s/%s", count_label(plan.rows).c_str(), count_label(plan.cols).c_str(),
                            count_label(plan.row_stride).c_str(), count_label(plan.col_stride).c_str());
            } else {
                ImGui::TextUnformatted("-");
            }
        } else {
            ImGui::TextUnformatted("-");
        }
    }
    ImGui::EndTable();
}

void draw_selected_dataset_diagnostics(const FileTab &tab, int max_texture_side, size_t max_texture_cells) {
    if (tab.selected_index < 0 || tab.selected_index >= static_cast<int>(tab.datasets.size())) {
        ImGui::TextUnformatted("No dataset selected.");
        return;
    }

    const DatasetInfo &info = tab.datasets[static_cast<size_t>(tab.selected_index)];
    ImGui::TextWrapped("Selected: %s", info.path.c_str());
    ImGui::Text("Shape: %s", shape_label(info.dims).c_str());
    ImGui::Text("Type: %s, %zu byte%s/value", type_label(info.type_class).c_str(), info.type_size,
                info.type_size == 1 ? "" : "s");
    ImGui::Text("Layout: %s", layout_label(info.layout).c_str());
    ImGui::Text("Chunk: %s", info.chunk_dims.empty() ? "-" : shape_label(info.chunk_dims).c_str());
    ImGui::TextWrapped("Filters: %s", join_labels(info.filters).c_str());
    ImGui::Text("Logical payload: %s", bytes_label(logical_dataset_bytes(info)).c_str());
    ImGui::Text("Allocated storage: %s", bytes_label(static_cast<long double>(info.storage_size)).c_str());

    if (info.dims.size() == 2 && info.numeric) {
        const TexturePreviewPlan plan = plan_texture_preview(info, max_texture_side, max_texture_cells);
        if (plan.valid) {
            ImGui::SeparatorText("Current 2D Preview Plan");
            ImGui::Text("Texture preview: %s x %s cells", count_label(plan.rows).c_str(), count_label(plan.cols).c_str());
            ImGui::Text("Read stride: every %s row(s), every %s column(s)", count_label(plan.row_stride).c_str(),
                        count_label(plan.col_stride).c_str());
            ImGui::Text("Preview values: %s (%s float buffer)", count_label(plan.cells).c_str(),
                        bytes_label(static_cast<long double>(plan.cells) * sizeof(float)).c_str());
            ImGui::Text("RGBA texture upload: %s", bytes_label(static_cast<long double>(plan.cells) * 4.0L).c_str());
            if (info.layout == H5D_CHUNKED && info.chunk_dims.size() == 2) {
                const hsize_t row_chunks =
                    distinct_chunks_for_selection(info.dims[0], plan.row_stride, info.chunk_dims[0]);
                const hsize_t col_chunks =
                    distinct_chunks_for_selection(info.dims[1], plan.col_stride, info.chunk_dims[1]);
                const long double touched_chunks = static_cast<long double>(row_chunks) * static_cast<long double>(col_chunks);
                const long double chunk_values =
                    static_cast<long double>(info.chunk_dims[0]) * static_cast<long double>(info.chunk_dims[1]);
                ImGui::Text("Estimated chunks touched: %.0Lf (%s row chunks x %s column chunks)", touched_chunks,
                            count_label(row_chunks).c_str(), count_label(col_chunks).c_str());
                ImGui::Text("Logical bytes per chunk: %s", bytes_label(chunk_values * info.type_size).c_str());
                ImGui::TextWrapped("If this is close to the full chunk count, a strided preview can still decompress most "
                                   "of the dataset. Precomputed preview datasets or viewport-tile reads will be faster.");
            }
        }
    }
}

void draw_file_details_window(const AppState &app, FileTab &tab) {
    if (!tab.show_file_details) {
        return;
    }

    const std::string title = "File Details: " + tab_title(tab) + "##details-" + std::to_string(tab.id);
    ImGui::SetNextWindowSize(ImVec2(980, 620), ImGuiCond_FirstUseEver);
    if (!ImGui::Begin(title.c_str(), &tab.show_file_details)) {
        ImGui::End();
        return;
    }

    ImGui::TextWrapped("File: %s", tab.current_file.c_str());
    std::error_code ec;
    const auto file_size = tab.current_file.empty() ? 0 : std::filesystem::file_size(tab.current_file, ec);
    if (!ec && file_size > 0) {
        ImGui::Text("File size: %s", bytes_label(static_cast<long double>(file_size)).c_str());
    }
    ImGui::Text("Datasets: %d", static_cast<int>(tab.datasets.size()));
    ImGui::Text("OpenGL max texture side: %d", app.gl_max_texture_size);
    ImGui::Text("Preview limits: max side %d, max cells %s", std::max(256, std::min(app.gl_max_texture_size, 4096)),
                count_label(static_cast<hsize_t>(LoadConfig{}.max_texture_cells)).c_str());
    ImGui::Separator();

    if (ImGui::CollapsingHeader("Selected Dataset", ImGuiTreeNodeFlags_DefaultOpen)) {
        draw_selected_dataset_diagnostics(tab, std::max(256, std::min(app.gl_max_texture_size, 4096)),
                                          LoadConfig{}.max_texture_cells);
    }
    if (ImGui::CollapsingHeader("All Datasets")) {
        draw_dataset_diagnostics_table(tab, std::max(256, std::min(app.gl_max_texture_size, 4096)),
                                       LoadConfig{}.max_texture_cells);
    }
    ImGui::End();
}

void draw_file_tab_content(AppState &app, FileTab &tab) {
    ImGui::PushID(tab.id);
    draw_dataset_panel(app, tab);
    ImGui::PopID();
}

void build_initial_dock_layout(AppState &app, const ImGuiViewport *viewport, ImGuiID dockspace_id) {
    if (app.dock_layout_built) {
        return;
    }

    app.dock_layout_built = true;
    app.dockspace_id = dockspace_id;

    ImGui::DockBuilderRemoveNode(dockspace_id);
    ImGui::DockBuilderAddNode(dockspace_id, ImGuiDockNodeFlags_DockSpace);
    ImGui::DockBuilderSetNodePos(dockspace_id, viewport->WorkPos);
    ImGui::DockBuilderSetNodeSize(dockspace_id, viewport->WorkSize);

    ImGuiID left_id = 0;
    ImGuiID right_id = 0;
    ImGui::DockBuilderSplitNode(dockspace_id, ImGuiDir_Left, 0.30f, &left_id, &right_id);
    app.file_dock_id = left_id;
    app.plot_dock_id = right_id;

    ImGui::DockBuilderDockWindow("HDF5 Files", app.file_dock_id);
    for (const auto &tab : app.tabs) {
        const std::string title = plot_window_title(*tab);
        ImGui::DockBuilderDockWindow(title.c_str(), app.plot_dock_id);
    }
    ImGui::DockBuilderFinish(dockspace_id);
}

void layout_windows(AppState &app) {
    const ImGuiViewport *viewport = ImGui::GetMainViewport();
    const ImGuiID dockspace_id = ImHashStr("HDF5MainDockSpace");
    build_initial_dock_layout(app, viewport, dockspace_id);
    ImGui::DockSpaceOverViewport(dockspace_id, viewport, ImGuiDockNodeFlags_None);

    if (app.file_dock_id != 0) {
        ImGui::SetNextWindowDockID(app.file_dock_id, ImGuiCond_FirstUseEver);
    }
    ImGui::SetNextWindowSize(ImVec2(430, 760), ImGuiCond_FirstUseEver);
    ImGui::Begin("HDF5 Files");
    const bool browser_interacting =
        ImGui::IsWindowFocused(ImGuiFocusedFlags_RootAndChildWindows) ||
        ImGui::IsWindowHovered(ImGuiHoveredFlags_RootAndChildWindows);
    if (ImGui::Button("Open File")) {
        request_file_picker(app);
    }
    ImGui::SameLine();
    ImGui::TextDisabled("%d file%s open", static_cast<int>(app.tabs.size()), app.tabs.size() == 1 ? "" : "s");
    ImGui::Checkbox("Show plot captions", &app.show_plot_captions);
    ImGui::SameLine();
    ImGui::Checkbox("Performance HUD", &app.show_performance_hud);

    if (app.tabs.empty()) {
        ImGui::Separator();
        ImGui::TextUnformatted("Open an HDF5 file to begin.");
    } else if (ImGui::BeginTabBar("file-tabs", ImGuiTabBarFlags_Reorderable | ImGuiTabBarFlags_AutoSelectNewTabs)) {
        std::vector<int> to_close;
        for (auto &tab_ptr : app.tabs) {
            FileTab &tab = *tab_ptr;
            bool open = true;
            ImGuiTabItemFlags flags = tab.id == app.active_tab_id ? ImGuiTabItemFlags_SetSelected : 0;
            const bool tab_visible = ImGui::BeginTabItem(tab_title(tab).c_str(), &open, flags);
            if (ImGui::IsItemClicked(ImGuiMouseButton_Left)) {
                app.active_tab_id = tab.id;
                app.plot_focus_request_id = tab.id;
            }
            if (tab_visible) {
                draw_file_tab_content(app, tab);
                ImGui::EndTabItem();
            }
            if (!open) {
                to_close.push_back(tab.id);
            }
        }
        ImGui::EndTabBar();
        for (int id : to_close) {
            close_tab(app, id);
        }
    }
    ImGui::End();

    for (auto &tab_ptr : app.tabs) {
        FileTab &tab = *tab_ptr;
        const std::string title = plot_window_title(tab);
        const bool focus_requested = app.plot_focus_request_id == tab.id;
        if (app.plot_dock_id != 0) {
            ImGui::SetNextWindowDockID(app.plot_dock_id, ImGuiCond_FirstUseEver);
        }
        if (focus_requested) {
            ImGui::SetNextWindowFocus();
        }
        ImGui::SetNextWindowSize(ImVec2(860, 620), ImGuiCond_FirstUseEver);
        if (ImGui::Begin(title.c_str())) {
            if (focus_requested ||
                (!browser_interacting && ImGui::IsWindowFocused(ImGuiFocusedFlags_RootAndChildWindows))) {
                app.active_tab_id = tab.id;
            }
            draw_loaded_plot(app, tab);
        }
        ImGui::End();
        if (focus_requested) {
            app.plot_focus_request_id = 0;
        }
        draw_file_details_window(app, tab);
    }

    draw_file_picker(app);
    draw_performance_hud(app);
}

ImPlotColormap register_turbo_colormap() {
    std::array<ImVec4, 256> colors{};
    for (size_t i = 0; i < colors.size(); ++i) {
        const auto rgba = turbo_rgba(static_cast<float>(i) / static_cast<float>(colors.size() - 1));
        colors[i] = ImVec4(static_cast<float>(rgba[0]) / 255.0f, static_cast<float>(rgba[1]) / 255.0f,
                           static_cast<float>(rgba[2]) / 255.0f, 1.0f);
    }
    return ImPlot::AddColormap("Turbo", colors.data(), static_cast<int>(colors.size()), false);
}

float framebuffer_font_scale(SDL_Window *window) {
    int window_w = 0;
    int window_h = 0;
    int drawable_w = 0;
    int drawable_h = 0;
    SDL_GetWindowSize(window, &window_w, &window_h);
    SDL_GL_GetDrawableSize(window, &drawable_w, &drawable_h);
    if (window_w <= 0 || window_h <= 0 || drawable_w <= 0 || drawable_h <= 0) {
        return 1.0f;
    }
    const float scale_x = static_cast<float>(drawable_w) / static_cast<float>(window_w);
    const float scale_y = static_cast<float>(drawable_h) / static_cast<float>(window_h);
    return std::clamp(std::max(scale_x, scale_y), 1.0f, 4.0f);
}

void load_ui_font(ImGuiIO &io, float framebuffer_scale) {
    static const std::array<const char *, 6> candidates = {
        "/System/Library/Fonts/Supplemental/Verdana.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
    };

    const float baked_size = 16.0f * std::max(1.0f, framebuffer_scale);
    io.FontGlobalScale = 1.0f / std::max(1.0f, framebuffer_scale);

    for (const char *path : candidates) {
        if (std::filesystem::exists(path)) {
            ImFontConfig cfg;
            cfg.OversampleH = 3;
            cfg.OversampleV = 2;
            cfg.PixelSnapH = true;
            cfg.RasterizerMultiply = 1.05f;
            if (io.Fonts->AddFontFromFileTTF(path, baked_size, &cfg) != nullptr) {
                return;
            }
        }
    }
    io.Fonts->AddFontDefault();
}

std::string default_file_path(int argc, char **argv) {
    if (argc > 1) {
        return argv[1];
    }
    const std::string fixture = "stability_analyses/1777392253_delay_analysis.h5";
    if (std::filesystem::exists(fixture)) {
        return fixture;
    }
    return {};
}

bool app_has_background_work(const AppState &app) {
    return std::any_of(app.tabs.begin(), app.tabs.end(), [](const std::unique_ptr<FileTab> &tab) {
        return tab->loading || tab->swmr_polling;
    });
}

bool imgui_wants_fast_frames(const ImGuiIO &io) {
    for (bool down : io.MouseDown) {
        if (down) {
            return true;
        }
    }
    return ImGui::IsAnyItemActive();
}

std::string gl_string(GLenum name) {
    const GLubyte *value = glGetString(name);
    return value == nullptr ? "unavailable" : reinterpret_cast<const char *>(value);
}

} // namespace

int main(int argc, char **argv) {
    H5Eset_auto2(H5E_DEFAULT, nullptr, nullptr);

    if (SDL_Init(SDL_INIT_VIDEO | SDL_INIT_TIMER | SDL_INIT_GAMECONTROLLER) != 0) {
        SDL_Log("SDL_Init failed: %s", SDL_GetError());
        return 1;
    }

    const char *glsl_version = "#version 150";
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_FLAGS, 0);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 3);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 2);
    SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);
    SDL_GL_SetAttribute(SDL_GL_DEPTH_SIZE, 24);
    SDL_GL_SetAttribute(SDL_GL_STENCIL_SIZE, 8);

    SDL_Window *window = SDL_CreateWindow("HDF5 ImPlot Viewer", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, 1440, 900,
                                          SDL_WINDOW_OPENGL | SDL_WINDOW_RESIZABLE | SDL_WINDOW_ALLOW_HIGHDPI);
    if (window == nullptr) {
        SDL_Log("SDL_CreateWindow failed: %s", SDL_GetError());
        SDL_Quit();
        return 1;
    }

    SDL_GLContext gl_context = SDL_GL_CreateContext(window);
    if (gl_context == nullptr) {
        SDL_Log("SDL_GL_CreateContext failed: %s", SDL_GetError());
        SDL_DestroyWindow(window);
        SDL_Quit();
        return 1;
    }
    SDL_GL_MakeCurrent(window, gl_context);
    SDL_GL_SetSwapInterval(1);

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImPlot::CreateContext();
    ImGuiIO &io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
    io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;
    const float ui_framebuffer_scale = framebuffer_font_scale(window);
    load_ui_font(io, ui_framebuffer_scale);

    ImGui::StyleColorsDark();
    ImPlot::GetStyle().Colormap = register_turbo_colormap();

    ImGui_ImplSDL2_InitForOpenGL(window, gl_context);
    ImGui_ImplOpenGL3_Init(glsl_version);

    AppState app;
    app.ui_framebuffer_scale = ui_framebuffer_scale;
    app.gl_vendor = gl_string(GL_VENDOR);
    app.gl_renderer = gl_string(GL_RENDERER);
    app.gl_version = gl_string(GL_VERSION);
    app.glsl_version = gl_string(GL_SHADING_LANGUAGE_VERSION);
    glGetIntegerv(GL_MAX_TEXTURE_SIZE, &app.gl_max_texture_size);
    if (app.gl_max_texture_size <= 0) {
        app.gl_max_texture_size = 4096;
    }

    if (argc > 1) {
        for (int i = 1; i < argc; ++i) {
            open_path_as_tab(app, argv[i]);
        }
    } else {
        const std::string initial_path = default_file_path(argc, argv);
        if (!initial_path.empty()) {
            open_path_as_tab(app, initial_path);
        }
    }

    bool done = false;
    bool force_redraw = true;
    auto process_event = [&](const SDL_Event &event) {
        ImGui_ImplSDL2_ProcessEvent(&event);
        if (event.type == SDL_QUIT) {
            done = true;
        }
        if (event.type == SDL_WINDOWEVENT && event.window.event == SDL_WINDOWEVENT_CLOSE &&
            event.window.windowID == SDL_GetWindowID(window)) {
            done = true;
        }
    };

    while (!done) {
        const auto frame_start = std::chrono::steady_clock::now();
        SDL_Event event;
        int event_count = 0;
        bool used_idle_wait = false;

        if (!force_redraw && !app_has_background_work(app)) {
            used_idle_wait = true;
            if (SDL_WaitEventTimeout(&event, static_cast<int>(kIdleFrameInterval.count())) != 0) {
                ++event_count;
                process_event(event);
                while (SDL_PollEvent(&event) != 0) {
                    ++event_count;
                    process_event(event);
                }
            }
        } else {
            while (SDL_PollEvent(&event) != 0) {
                ++event_count;
                process_event(event);
            }
        }

        bool state_changed = false;
        for (auto &tab : app.tabs) {
            state_changed = poll_load(*tab) || state_changed;
            state_changed = poll_swmr_live(app, *tab) || state_changed;
        }

        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplSDL2_NewFrame();
        ImGui::NewFrame();

        layout_windows(app);
        state_changed = handle_keyboard_shortcuts(app) || state_changed;

        const bool ui_fast_frames = imgui_wants_fast_frames(io);
        ImGui::Render();
        int display_w = 0;
        int display_h = 0;
        SDL_GL_GetDrawableSize(window, &display_w, &display_h);
        glViewport(0, 0, display_w, display_h);
        glClearColor(0.07f, 0.08f, 0.09f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
        SDL_GL_SwapWindow(window);

        const bool background_work = app_has_background_work(app);
        const bool fast_frames = state_changed || background_work || ui_fast_frames;
        force_redraw = fast_frames;
        const auto frame_elapsed = std::chrono::steady_clock::now() - frame_start;
        app.last_frame_ms = std::chrono::duration<double, std::milli>(frame_elapsed).count();
        app.last_event_count = event_count;
        app.last_used_idle_wait = used_idle_wait;
        app.last_fast_frame = fast_frames;
        app.last_background_work = background_work;
        app.last_ui_active = ui_fast_frames;
        if (fast_frames && frame_elapsed < kActiveFrameInterval) {
            const auto delay_ms = std::chrono::duration_cast<std::chrono::milliseconds>(kActiveFrameInterval - frame_elapsed);
            if (delay_ms.count() > 0) {
                SDL_Delay(static_cast<Uint32>(delay_ms.count()));
            }
        }
    }

    for (auto &tab : app.tabs) {
        if (tab->loading && tab->load_future.valid()) {
            tab->load_future.wait();
        }
        if (tab->swmr_polling && tab->swmr_future.valid()) {
            tab->swmr_future.wait();
        }
        delete_heat_texture(*tab);
    }
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplSDL2_Shutdown();
    ImPlot::DestroyContext();
    ImGui::DestroyContext();
    SDL_GL_DeleteContext(gl_context);
    SDL_DestroyWindow(window);
    SDL_Quit();
    return 0;
}

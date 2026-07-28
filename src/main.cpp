#include "imgui.h"
#include "imgui_internal.h"
#include "imgui_impl_opengl3.h"
#include "imgui_impl_sdl2.h"
#include "implot.h"
#include "fft_transform.h"
#include "plot_axis_mapping.h"
#include "savitzky_golay.h"
#include "windowed_average.h"

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
    std::vector<double> x_axis_label_values;
    hsize_t source_count = 0;
    hsize_t source_stride = 1;
    bool x_axis_labels_from_values = false;
    bool x_values_increasing = true;
    bool x_values_decreasing = false;
    std::string x_source_path;

    std::vector<uint8_t> rgba;
    std::vector<float> heat_values;
    std::vector<double> heat_x_values;
    std::vector<double> heat_y_values;
    std::vector<double> y_axis_label_values;
    bool y_axis_labels_from_values = false;
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
    int applied_color_generation = -1;
};

struct LoadConfig {
    size_t max_line_values = 12000000;
    int max_texture_side = 4096;
    size_t max_texture_cells = 12000000;
    bool transpose_2d = false;
    bool sort_x_axis_increasing = false;
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

struct CommentPreviewResult {
    int token = 0;
    std::string path;
    std::string caption;
    std::string error;
};

struct CommentPreviewCacheEntry {
    std::string path;
    std::string caption;
    std::string error;
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

struct SliceState {
    bool enabled = false;
    bool row_window_open = true;
    bool column_window_open = true;
    int row_index = 0;
    int column_index = 0;
    bool heatmap_view_valid = false;
    double view_x_min = 0.0;
    double view_x_max = 1.0;
    double view_y_min = 0.0;
    double view_y_max = 1.0;
};

enum class PlotFilterKind { SavitzkyGolay, WindowedAverage, Fft };

struct PlotFilterStage {
    explicit PlotFilterStage(PlotFilterKind stage_kind = PlotFilterKind::SavitzkyGolay) : kind(stage_kind) {}

    PlotFilterKind kind = PlotFilterKind::SavitzkyGolay;
    bool enabled = false;
    hdf5_plotter::SavitzkyGolayParameters savitzky_golay;
    hdf5_plotter::WindowedAverageParameters windowed_average;
    hdf5_plotter::FftParameters fft;
};

struct FilterPipelineResult {
    int token = 0;
    bool ok = false;
    std::string error;
    std::shared_ptr<LoadedDataset> source;
    std::vector<float> values;
    float value_min = 0.0f;
    float value_max = 1.0f;
    bool x_axis_override = false;
    bool y_axis_override = false;
    std::vector<double> x_axis_values;
    std::vector<double> y_axis_values;
    double x_min = 0.0;
    double x_max = 1.0;
    double y_min = 0.0;
    double y_max = 1.0;
    std::string x_label;
    std::string y_label;
    std::string value_label;
};

struct FilterPipelineState {
    std::vector<PlotFilterStage> stages = {PlotFilterStage{PlotFilterKind::SavitzkyGolay},
                                           PlotFilterStage{PlotFilterKind::WindowedAverage},
                                           PlotFilterStage{PlotFilterKind::Fft}};
    bool settings_open = false;
    bool processing = false;
    bool recompute_pending = false;
    int requested_token = 0;
    int output_token = 0;
    std::future<FilterPipelineResult> future;
    std::shared_ptr<std::atomic<bool>> cancel_requested;
    std::shared_ptr<LoadedDataset> output_source;
    std::vector<float> output_values;
    float output_min = 0.0f;
    float output_max = 1.0f;
    bool x_axis_override = false;
    bool y_axis_override = false;
    std::vector<double> x_axis_values;
    std::vector<double> y_axis_values;
    double x_min = 0.0;
    double x_max = 1.0;
    double y_min = 0.0;
    double y_max = 1.0;
    std::string x_label;
    std::string y_label;
    std::string value_label;
    bool preserve_axis_controls_on_next_result = false;
    std::string error;
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
    bool sort_x_axis_increasing = false;
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
    SliceState slices;
    FilterPipelineState filters;

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
    int preview_token = 0;
    bool preview_loading = false;
    std::future<CommentPreviewResult> preview_future;
    std::string preview_path;
    std::string preview_caption;
    std::string preview_error;
    std::string pending_preview_path;
    std::vector<CommentPreviewCacheEntry> preview_cache;
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
    bool last_window_drawable = true;
    bool last_window_focused = true;
};

void clamp_slice_state(FileTab &tab, const LoadedDataset &data);

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
    double low = std::min(data.x_min, data.x_max);
    double high = std::max(data.x_min, data.x_max);
    if (!data.x_axis_label_values.empty()) {
        double label_min = std::numeric_limits<double>::infinity();
        double label_max = -std::numeric_limits<double>::infinity();
        for (double value : data.x_axis_label_values) {
            if (std::isfinite(value)) {
                label_min = std::min(label_min, value);
                label_max = std::max(label_max, value);
            }
        }
        if (std::isfinite(label_min) && std::isfinite(label_max)) {
            low = label_min;
            high = label_max;
        }
    }
    return low > 946684800.0 && high < 4102444800.0;
}

int unix_time_formatter(double value, char *buffer, int size, void *user_data) {
    const bool utc = user_data != nullptr && *static_cast<bool *>(user_data);
    const std::time_t seconds = static_cast<std::time_t>(std::llround(value));
    std::tm tm_value{};
    bool ok = false;
#if defined(_WIN32)
    ok = utc ? gmtime_s(&tm_value, &seconds) == 0 : localtime_s(&tm_value, &seconds) == 0;
#else
    ok = utc ? gmtime_r(&seconds, &tm_value) != nullptr : localtime_r(&seconds, &tm_value) != nullptr;
#endif
    if (!ok) {
        return std::snprintf(buffer, static_cast<size_t>(size), "%.0f", value);
    }

    char time_buffer[64] = {};
    std::strftime(time_buffer, sizeof(time_buffer), "%Y-%m-%d %H:%M:%S", &tm_value);
    return std::snprintf(buffer, static_cast<size_t>(size), "%s", time_buffer);
}

struct AxisValueFormatterData {
    const std::vector<double> *values = nullptr;
    double axis_min = 0.0;
    double axis_max = 1.0;
    bool datetime = false;
    bool utc = false;
};

int index_ordered_axis_formatter(double value, char *buffer, int size, void *user_data) {
    const auto *formatter = static_cast<const AxisValueFormatterData *>(user_data);
    if (formatter == nullptr || formatter->values == nullptr) {
        return std::snprintf(buffer, static_cast<size_t>(size), "%.6g", value);
    }

    const double label_value = hdf5_plotter::axis_label_value_at_coordinate(
        *formatter->values, value, formatter->axis_min, formatter->axis_max);
    if (formatter->datetime) {
        bool utc = formatter->utc;
        return unix_time_formatter(label_value, buffer, size, &utc);
    }
    return std::snprintf(buffer, static_cast<size_t>(size), "%.6g", label_value);
}

void setup_x_axis_format(FileTab &tab, const LoadedDataset &data, AxisValueFormatterData &formatter) {
    const bool datetime = tab.x_datetime && looks_like_unix_time_axis(data);
    if (data.x_axis_labels_from_values && !data.x_axis_label_values.empty()) {
        formatter.values = &data.x_axis_label_values;
        formatter.axis_min = data.x_min;
        formatter.axis_max = data.x_max;
        formatter.datetime = datetime;
        formatter.utc = tab.x_datetime_utc;
        ImPlot::SetupAxisFormat(ImAxis_X1, index_ordered_axis_formatter, &formatter);
    } else if (datetime) {
        ImPlot::SetupAxisFormat(ImAxis_X1, unix_time_formatter, &tab.x_datetime_utc);
    }
}

void setup_y_axis_format(const LoadedDataset &data, ImAxis axis, AxisValueFormatterData &formatter) {
    if (!data.y_axis_labels_from_values || data.y_axis_label_values.empty()) {
        return;
    }
    formatter.values = &data.y_axis_label_values;
    formatter.axis_min = data.y_min;
    formatter.axis_max = data.y_max;
    ImPlot::SetupAxisFormat(axis, index_ordered_axis_formatter, &formatter);
}

bool preferred_name(const std::string &name, const std::vector<std::string> &preferred) {
    const std::string lowered = lower_ascii(name);
    return std::find(preferred.begin(), preferred.end(), lowered) != preferred.end();
}

bool plausible_axis_name(const std::string &name) {
    const std::string lowered = lower_ascii(name);
    if (lowered == "x" || lowered == "y" || lowered == "row" || lowered == "rows" ||
        lowered == "column" || lowered == "columns") {
        return true;
    }
    static const std::array<const char *, 8> axis_terms = {
        "axis", "coord", "index", "time", "freq", "wave", "delay", "position"};
    return std::any_of(axis_terms.begin(), axis_terms.end(), [&](const char *term) {
        return lowered.find(term) != std::string::npos;
    });
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

struct CommentVisitState {
    hid_t file = -1;
    std::vector<DatasetInfo> *comments = nullptr;
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

#if H5_VERSION_GE(1, 12, 0)
herr_t visit_comment_dataset(hid_t, const char *name, const H5O_info2_t *info, void *op_data) {
#else
herr_t visit_comment_dataset(hid_t, const char *name, const H5O_info_t *info, void *op_data) {
#endif
    if (info == nullptr || info->type != H5O_TYPE_DATASET || name == nullptr || std::strcmp(name, ".") == 0) {
        return 0;
    }

    const std::string open_path(name);
    const std::string display_path = open_path.empty() || open_path[0] == '/' ? open_path : "/" + open_path;
    const std::string dataset_name = lower_ascii(base_name(display_path));
    if (dataset_name != "comment" && dataset_name != "comments") {
        return 0;
    }

    auto *state = static_cast<CommentVisitState *>(op_data);
    try {
        state->comments->push_back(inspect_dataset(state->file, open_path, display_path));
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

std::vector<DatasetInfo> collect_comment_index(hid_t file) {
    std::vector<DatasetInfo> comments;
    CommentVisitState state{file, &comments};
#if H5_VERSION_GE(1, 12, 0)
    H5Ovisit3(file, H5_INDEX_NAME, H5_ITER_NATIVE, visit_comment_dataset, &state, H5O_INFO_BASIC);
#else
    H5Ovisit(file, H5_INDEX_NAME, H5_ITER_NATIVE, visit_comment_dataset, &state);
#endif
    std::sort(comments.begin(), comments.end(), [](const DatasetInfo &lhs, const DatasetInfo &rhs) {
        return lhs.path < rhs.path;
    });
    return comments;
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

std::string load_file_comment_preview(const std::string &path) {
    std::lock_guard<std::mutex> lock(g_hdf5_mutex);
    H5File file(open_readonly_file(path));
    if (!file.valid()) {
        throw std::runtime_error("could not open HDF5 file");
    }

    std::vector<DatasetInfo> comments = collect_comment_index(file.get());
    return collect_comment_caption(file.get(), comments);
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

std::vector<size_t> ascending_axis_permutation(const std::vector<double> &values) {
    std::vector<size_t> order(values.size());
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [&](size_t lhs, size_t rhs) {
        const double lhs_value = values[lhs];
        const double rhs_value = values[rhs];
        const bool lhs_finite = std::isfinite(lhs_value);
        const bool rhs_finite = std::isfinite(rhs_value);
        if (lhs_finite != rhs_finite) {
            return lhs_finite;
        }
        if (!lhs_finite && !rhs_finite) {
            return lhs < rhs;
        }
        if (lhs_value == rhs_value) {
            return lhs < rhs;
        }
        return lhs_value < rhs_value;
    });
    return order;
}

template <typename T> std::vector<T> reordered_by_permutation(const std::vector<T> &values, const std::vector<size_t> &order) {
    std::vector<T> reordered(order.size());
    for (size_t i = 0; i < order.size(); ++i) {
        reordered[i] = values[order[i]];
    }
    return reordered;
}

std::vector<float> reorder_2d_columns(const std::vector<float> &values, hsize_t rows, hsize_t cols,
                                      const std::vector<size_t> &order) {
    const size_t row_count = static_cast<size_t>(rows);
    const size_t col_count = static_cast<size_t>(cols);
    if (order.size() != col_count || values.size() != row_count * col_count) {
        return values;
    }

    std::vector<float> reordered(values.size());
    parallel_for(row_count, [&](size_t begin, size_t end) {
        for (size_t row = begin; row < end; ++row) {
            for (size_t col = 0; col < col_count; ++col) {
                reordered[row * col_count + col] = values[row * col_count + order[col]];
            }
        }
    });
    return reordered;
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
    if (!preferred_anywhere.empty()) {
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
    if (matches.size() == 1 && plausible_axis_name(base_name(matches.front()->path))) {
        return {matches.front()->path, base_name(matches.front()->path)};
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
                               DatasetInfo x_axis_info, bool has_x_axis_info, DatasetInfo y_axis_info,
                               bool has_y_axis_info, LoadConfig config) {
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
                std::vector<double> x_values = read_1d_dataset_double(file_path, x_axis_info, stride);
                if (x_values.size() == result.data->line_values.size()) {
                    result.data->x_label = x_axis.label;
                    result.data->x_source_path = x_axis.path;
                    if (config.sort_x_axis_increasing) {
                        const std::vector<size_t> order = ascending_axis_permutation(x_values);
                        result.data->line_values = reordered_by_permutation(result.data->line_values, order);
                        x_values = reordered_by_permutation(x_values, order);
                    }
                    result.data->x_values_increasing = is_monotonic_increasing(x_values);
                    result.data->x_values_decreasing = is_monotonic_decreasing(x_values);
                    if (config.sort_x_axis_increasing || result.data->x_values_increasing) {
                        auto [x_min, x_max] = finite_minmax(x_values);
                        result.data->x_min = x_min;
                        result.data->x_max = x_max;
                        result.data->line_x_values = std::move(x_values);
                    } else {
                        result.data->x_min = 0.0;
                        result.data->x_max = source_count > 1 ? static_cast<double>(source_count - 1) : 0.0;
                        result.data->x_axis_labels_from_values = true;
                        result.data->x_axis_label_values = std::move(x_values);
                    }
                } else {
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
            const hsize_t x_axis_stride = config.transpose_2d ? plan.row_stride : plan.col_stride;
            const hsize_t y_axis_stride = config.transpose_2d ? plan.col_stride : plan.row_stride;
            bool applied_x_axis_values = false;
            if (has_x_axis_info && !x_axis.path.empty()) {
                std::vector<double> x_values = read_1d_dataset_double(file_path, x_axis_info, x_axis_stride);
                if (x_values.size() == static_cast<size_t>(texture_cols)) {
                    result.data->x_label = x_axis.label;
                    result.data->x_source_path = x_axis.path;
                    if (config.sort_x_axis_increasing) {
                        const std::vector<size_t> order = ascending_axis_permutation(x_values);
                        values = reorder_2d_columns(values, texture_rows, texture_cols, order);
                        x_values = reordered_by_permutation(x_values, order);
                    }
                    const hdf5_plotter::PreviewAxisMapping mapping =
                        hdf5_plotter::make_preview_axis_mapping(
                            x_values, static_cast<size_t>(texture_cols),
                            static_cast<size_t>(x_axis_stride),
                            static_cast<size_t>(config.x_fallback_count));
                    result.data->x_min = mapping.min;
                    result.data->x_max = mapping.max;
                    result.data->x_axis_labels_from_values = mapping.labels_from_values;
                    if (mapping.labels_from_values) {
                        result.data->x_axis_label_values = x_values;
                    }
                    result.data->heat_x_values = std::move(x_values);
                    applied_x_axis_values = true;
                }
            }
            bool applied_y_axis_values = false;
            if (has_y_axis_info && !y_axis.path.empty()) {
                std::vector<double> y_values = read_1d_dataset_double(file_path, y_axis_info, y_axis_stride);
                if (y_values.size() == static_cast<size_t>(texture_rows)) {
                    result.data->y_label = y_axis.label;
                    const hdf5_plotter::PreviewAxisMapping mapping =
                        hdf5_plotter::make_preview_axis_mapping(
                            y_values, static_cast<size_t>(texture_rows),
                            static_cast<size_t>(y_axis_stride),
                            static_cast<size_t>(config.y_fallback_count));
                    result.data->y_min = mapping.min;
                    result.data->y_max = mapping.max;
                    result.data->y_axis_labels_from_values = mapping.labels_from_values;
                    if (mapping.labels_from_values) {
                        result.data->y_axis_label_values = y_values;
                    }
                    result.data->heat_y_values = std::move(y_values);
                    applied_y_axis_values = true;
                }
            }
            if (!applied_x_axis_values) {
                const hdf5_plotter::PreviewAxisMapping mapping =
                    hdf5_plotter::make_preview_axis_mapping(
                        {}, static_cast<size_t>(texture_cols),
                        static_cast<size_t>(x_axis_stride),
                        static_cast<size_t>(config.x_fallback_count));
                result.data->x_min = mapping.min;
                result.data->x_max = mapping.max;
                result.data->x_label = x_axis.label.empty() ? config.x_fallback_label : x_axis.label;
                result.data->x_source_path = x_axis.path;
            }
            if (!applied_y_axis_values) {
                const hdf5_plotter::PreviewAxisMapping mapping =
                    hdf5_plotter::make_preview_axis_mapping(
                        {}, static_cast<size_t>(texture_rows),
                        static_cast<size_t>(y_axis_stride),
                        static_cast<size_t>(config.y_fallback_count));
                result.data->y_min = mapping.min;
                result.data->y_max = mapping.max;
                result.data->y_label = y_axis.label.empty() ? config.y_fallback_label : y_axis.label;
            }
            normalize_range(result.data->x_min, result.data->x_max);
            normalize_range(result.data->y_min, result.data->y_max);
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
            result.data->applied_color_generation = 0;
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

PlotFilterStage &savitzky_golay_stage(FilterPipelineState &filters) {
    auto found = std::find_if(filters.stages.begin(), filters.stages.end(), [](const PlotFilterStage &stage) {
        return stage.kind == PlotFilterKind::SavitzkyGolay;
    });
    if (found == filters.stages.end()) {
        filters.stages.push_back(PlotFilterStage{});
        return filters.stages.back();
    }
    return *found;
}

PlotFilterStage &windowed_average_stage(FilterPipelineState &filters) {
    auto found = std::find_if(filters.stages.begin(), filters.stages.end(), [](const PlotFilterStage &stage) {
        return stage.kind == PlotFilterKind::WindowedAverage;
    });
    if (found == filters.stages.end()) {
        filters.stages.push_back(PlotFilterStage{PlotFilterKind::WindowedAverage});
        return filters.stages.back();
    }
    return *found;
}

PlotFilterStage &fft_stage(FilterPipelineState &filters) {
    auto found = std::find_if(filters.stages.begin(), filters.stages.end(), [](const PlotFilterStage &stage) {
        return stage.kind == PlotFilterKind::Fft;
    });
    if (found == filters.stages.end()) {
        filters.stages.push_back(PlotFilterStage{PlotFilterKind::Fft});
        return filters.stages.back();
    }
    return *found;
}

bool filter_pipeline_enabled(const FilterPipelineState &filters) {
    return std::any_of(filters.stages.begin(), filters.stages.end(),
                       [](const PlotFilterStage &stage) { return stage.enabled; });
}

const std::vector<float> &source_plot_values(const LoadedDataset &data) {
    return data.kind == LoadedKind::Heatmap2D ? data.heat_values : data.line_values;
}

std::pair<size_t, size_t> plot_value_dimensions(const LoadedDataset &data) {
    if (data.kind == LoadedKind::Heatmap2D) {
        return {static_cast<size_t>(std::max(0, data.texture_width)),
                static_cast<size_t>(std::max(0, data.texture_height))};
    }
    return {data.line_values.size(), 1};
}

struct AxisSampling {
    double spacing = 1.0;
    bool physical_units = false;
    std::string frequency_label = "frequency (cycles/sample)";
};

bool generic_index_axis_label(const std::string &label) {
    const std::string lowered = lower_ascii(label);
    return lowered.empty() || lowered == "index" || lowered == "indices" || lowered == "row" ||
           lowered == "rows" || lowered == "column" || lowered == "columns";
}

AxisSampling fft_axis_sampling(const LoadedDataset &data, hdf5_plotter::FilterAxis axis) {
    const auto [width, height] = plot_value_dimensions(data);
    const size_t count = axis == hdf5_plotter::FilterAxis::X ? width : height;
    const std::vector<double> *axis_values = nullptr;
    const std::string &axis_label = axis == hdf5_plotter::FilterAxis::X ? data.x_label : data.y_label;
    double axis_min = axis == hdf5_plotter::FilterAxis::X ? data.x_min : data.y_min;
    double axis_max = axis == hdf5_plotter::FilterAxis::X ? data.x_max : data.y_max;
    if (data.kind == LoadedKind::Line1D) {
        axis_values = &data.line_x_values;
    } else if (axis == hdf5_plotter::FilterAxis::X) {
        axis_values = &data.heat_x_values;
    } else {
        axis_values = &data.heat_y_values;
    }

    AxisSampling sampling;
    bool uniform_values = false;
    if (axis_values != nullptr && axis_values->size() == count && count >= 2) {
        const double mean_step = (axis_values->back() - axis_values->front()) / static_cast<double>(count - 1);
        if (std::isfinite(mean_step) && mean_step != 0.0) {
            const double tolerance = std::max(1e-12, std::abs(mean_step) * 1e-4);
            uniform_values = true;
            for (size_t i = 1; i < count; ++i) {
                const double step = (*axis_values)[i] - (*axis_values)[i - 1];
                if (!std::isfinite(step) || std::abs(step - mean_step) > tolerance) {
                    uniform_values = false;
                    break;
                }
            }
            if (uniform_values) {
                sampling.spacing = std::abs(mean_step);
            }
        }
    } else if (count >= 2 && std::isfinite(axis_min) && std::isfinite(axis_max) && axis_min != axis_max) {
        sampling.spacing = std::abs(axis_max - axis_min) / static_cast<double>(count - 1);
        uniform_values = std::isfinite(sampling.spacing) && sampling.spacing > 0.0;
    }

    if (!uniform_values) {
        sampling.spacing = 1.0;
    }
    sampling.physical_units = uniform_values && !generic_index_axis_label(axis_label);
    if (sampling.physical_units) {
        sampling.frequency_label = "frequency (1/" + axis_label + ")";
    }
    return sampling;
}

std::string fft_value_label(const hdf5_plotter::FftParameters &parameters) {
    switch (parameters.value_mode) {
    case hdf5_plotter::FftValueMode::Magnitude:
        return parameters.decibels ? "FFT magnitude (dB)" : "FFT magnitude";
    case hdf5_plotter::FftValueMode::Power:
        return parameters.decibels ? "FFT power (dB)" : "FFT power";
    case hdf5_plotter::FftValueMode::Phase:
        return "FFT phase (rad)";
    case hdf5_plotter::FftValueMode::Real:
        return "FFT real";
    case hdf5_plotter::FftValueMode::Imaginary:
        return "FFT imaginary";
    }
    return "FFT";
}

size_t savitzky_golay_axis_length(const LoadedDataset &data,
                                  hdf5_plotter::FilterAxis axis) {
    if (data.kind == LoadedKind::Line1D) {
        return data.line_values.size();
    }
    const auto [width, height] = plot_value_dimensions(data);
    return hdf5_plotter::savitzky_golay_axis_length(width, height, axis);
}

size_t windowed_average_axis_length(const LoadedDataset &data, hdf5_plotter::FilterAxis axis) {
    if (data.kind == LoadedKind::Line1D) {
        return data.line_values.size();
    }
    const auto [width, height] = plot_value_dimensions(data);
    return hdf5_plotter::windowed_average_axis_length(width, height, axis);
}

bool sanitize_savitzky_golay_parameters(const LoadedDataset &data,
                                        hdf5_plotter::SavitzkyGolayParameters &parameters) {
    const hdf5_plotter::SavitzkyGolayParameters original = parameters;
    if (data.kind == LoadedKind::Line1D) {
        parameters.axis = hdf5_plotter::FilterAxis::X;
    } else if (savitzky_golay_axis_length(data, parameters.axis) < 3) {
        const hdf5_plotter::FilterAxis other_axis =
            parameters.axis == hdf5_plotter::FilterAxis::X ? hdf5_plotter::FilterAxis::Y
                                                            : hdf5_plotter::FilterAxis::X;
        if (savitzky_golay_axis_length(data, other_axis) >= 3) {
            parameters.axis = other_axis;
        }
    }

    const size_t axis_length = savitzky_golay_axis_length(data, parameters.axis);
    int max_window = static_cast<int>(std::min<size_t>(axis_length, 501));
    if (max_window % 2 == 0) {
        --max_window;
    }
    max_window = std::max(3, max_window);
    parameters.window_size = std::clamp(parameters.window_size, 3, max_window);
    if (parameters.window_size % 2 == 0) {
        parameters.window_size =
            parameters.window_size < max_window ? parameters.window_size + 1 : parameters.window_size - 1;
    }
    parameters.polynomial_order =
        std::clamp(parameters.polynomial_order, 0, std::min(10, parameters.window_size - 1));
    parameters.derivative_order =
        std::clamp(parameters.derivative_order, 0, parameters.polynomial_order);
    return parameters.window_size != original.window_size ||
           parameters.polynomial_order != original.polynomial_order ||
           parameters.derivative_order != original.derivative_order || parameters.axis != original.axis;
}

bool sanitize_windowed_average_parameters(const LoadedDataset &data,
                                          hdf5_plotter::WindowedAverageParameters &parameters) {
    const hdf5_plotter::WindowedAverageParameters original = parameters;
    if (data.kind == LoadedKind::Line1D) {
        parameters.axis = hdf5_plotter::FilterAxis::X;
    } else if (windowed_average_axis_length(data, parameters.axis) < 3) {
        const hdf5_plotter::FilterAxis other_axis =
            parameters.axis == hdf5_plotter::FilterAxis::X ? hdf5_plotter::FilterAxis::Y
                                                            : hdf5_plotter::FilterAxis::X;
        if (windowed_average_axis_length(data, other_axis) >= 3) {
            parameters.axis = other_axis;
        }
    }

    const size_t axis_length = windowed_average_axis_length(data, parameters.axis);
    if (axis_length < 3) {
        parameters.window_size = 3;
    } else {
        int max_window =
            static_cast<int>(std::min<size_t>(axis_length, static_cast<size_t>(std::numeric_limits<int>::max())));
        if (max_window % 2 == 0) {
            --max_window;
        }
        parameters.window_size = std::clamp(parameters.window_size, 3, max_window);
        if (parameters.window_size % 2 == 0) {
            parameters.window_size =
                parameters.window_size < max_window ? parameters.window_size + 1 : parameters.window_size - 1;
        }
    }
    return parameters.window_size != original.window_size || parameters.axis != original.axis;
}

void sanitize_fft_parameters(const LoadedDataset &data, hdf5_plotter::FftParameters &parameters) {
    if (data.kind == LoadedKind::Line1D) {
        parameters.axis = hdf5_plotter::FilterAxis::X;
        return;
    }
    const auto [width, height] = plot_value_dimensions(data);
    const size_t current_length = parameters.axis == hdf5_plotter::FilterAxis::X ? width : height;
    if (current_length >= 2) {
        return;
    }
    parameters.axis = width >= 2 ? hdf5_plotter::FilterAxis::X : hdf5_plotter::FilterAxis::Y;
}

FilterPipelineResult filter_pipeline_worker(int token, std::shared_ptr<LoadedDataset> source,
                                            std::vector<PlotFilterStage> stages,
                                            std::shared_ptr<std::atomic<bool>> cancel_requested) {
    FilterPipelineResult result;
    result.token = token;
    result.source = std::move(source);
    try {
        if (!result.source) {
            throw std::runtime_error("no dataset is loaded");
        }
        const auto [width, height] = plot_value_dimensions(*result.source);
        const std::vector<float> *current = &source_plot_values(*result.source);
        std::vector<float> filtered;
        float value_min = result.source->value_min;
        float value_max = result.source->value_max;
        bool applied_stage = false;

        for (const PlotFilterStage &stage : stages) {
            if (!stage.enabled) {
                continue;
            }
            if (cancel_requested->load(std::memory_order_relaxed)) {
                throw std::runtime_error("filter update cancelled");
            }
            switch (stage.kind) {
            case PlotFilterKind::SavitzkyGolay: {
                hdf5_plotter::SavitzkyGolayParameters parameters = stage.savitzky_golay;
                if (result.source->kind == LoadedKind::Line1D) {
                    parameters.axis = hdf5_plotter::FilterAxis::X;
                }
                hdf5_plotter::SavitzkyGolayOutput output =
                    hdf5_plotter::apply_savitzky_golay(*current, width, height, parameters, cpu_worker_limit(),
                                                      cancel_requested.get());
                filtered = std::move(output.values);
                value_min = output.value_min;
                value_max = output.value_max;
                current = &filtered;
                applied_stage = true;
                break;
            }
            case PlotFilterKind::WindowedAverage: {
                hdf5_plotter::WindowedAverageParameters parameters = stage.windowed_average;
                if (result.source->kind == LoadedKind::Line1D) {
                    parameters.axis = hdf5_plotter::FilterAxis::X;
                }
                hdf5_plotter::WindowedAverageOutput output =
                    hdf5_plotter::apply_windowed_average(*current, width, height, parameters,
                                                         cpu_worker_limit(), cancel_requested.get());
                filtered = std::move(output.values);
                value_min = output.value_min;
                value_max = output.value_max;
                current = &filtered;
                applied_stage = true;
                break;
            }
            case PlotFilterKind::Fft: {
                hdf5_plotter::FftParameters parameters = stage.fft;
                if (result.source->kind == LoadedKind::Line1D) {
                    parameters.axis = hdf5_plotter::FilterAxis::X;
                }
                AxisSampling sampling = fft_axis_sampling(*result.source, parameters.axis);
                if (!parameters.automatic_spacing && std::isfinite(parameters.sample_spacing) &&
                    parameters.sample_spacing > 0.0) {
                    sampling.spacing = parameters.sample_spacing;
                    const std::string &axis_label = parameters.axis == hdf5_plotter::FilterAxis::X
                                                        ? result.source->x_label
                                                        : result.source->y_label;
                    sampling.physical_units = !generic_index_axis_label(axis_label);
                    sampling.frequency_label = sampling.physical_units
                                                   ? "frequency (1/" + axis_label + ")"
                                                   : "frequency (cycles/sample)";
                }
                hdf5_plotter::FftOutput output =
                    hdf5_plotter::apply_fft_transform(*current, width, height, parameters, sampling.spacing,
                                                      cpu_worker_limit(), cancel_requested.get());
                filtered = std::move(output.values);
                value_min = output.value_min;
                value_max = output.value_max;
                current = &filtered;
                applied_stage = true;
                if (parameters.axis == hdf5_plotter::FilterAxis::X) {
                    result.x_axis_override = true;
                    result.x_axis_values = std::move(output.frequency_values);
                    result.x_min = result.x_axis_values.front();
                    result.x_max = result.x_axis_values.back();
                    result.x_label = sampling.frequency_label;
                } else {
                    result.y_axis_override = true;
                    result.y_axis_values = std::move(output.frequency_values);
                    result.y_min = result.y_axis_values.front();
                    result.y_max = result.y_axis_values.back();
                    result.y_label = sampling.frequency_label;
                }
                result.value_label = fft_value_label(parameters);
                break;
            }
            }
        }

        if (!applied_stage) {
            filtered = *current;
        }
        result.values = std::move(filtered);
        result.value_min = value_min;
        result.value_max = value_max;
        result.ok = true;
    } catch (const std::exception &ex) {
        result.error = ex.what();
    }
    return result;
}

bool filter_output_ready(const FileTab &tab, const LoadedDataset &data) {
    const FilterPipelineState &filters = tab.filters;
    return filter_pipeline_enabled(filters) && filters.output_source.get() == &data &&
           filters.output_values.size() == source_plot_values(data).size();
}

const std::vector<float> &displayed_plot_values(const FileTab &tab, const LoadedDataset &data) {
    return filter_output_ready(tab, data) ? tab.filters.output_values : source_plot_values(data);
}

std::pair<float, float> displayed_value_range(const FileTab &tab, const LoadedDataset &data) {
    return filter_output_ready(tab, data) ? std::pair<float, float>{tab.filters.output_min, tab.filters.output_max}
                                          : std::pair<float, float>{data.value_min, data.value_max};
}

int displayed_value_generation(const FileTab &tab, const LoadedDataset &data) {
    return filter_output_ready(tab, data) ? tab.filters.output_token : 0;
}

bool displayed_x_axis_overridden(const FileTab &tab, const LoadedDataset &data) {
    return filter_output_ready(tab, data) && tab.filters.x_axis_override;
}

bool displayed_y_axis_overridden(const FileTab &tab, const LoadedDataset &data) {
    return filter_output_ready(tab, data) && tab.filters.y_axis_override;
}

double displayed_x_min(const FileTab &tab, const LoadedDataset &data) {
    return displayed_x_axis_overridden(tab, data) ? tab.filters.x_min : data.x_min;
}

double displayed_x_max(const FileTab &tab, const LoadedDataset &data) {
    return displayed_x_axis_overridden(tab, data) ? tab.filters.x_max : data.x_max;
}

double displayed_y_min(const FileTab &tab, const LoadedDataset &data) {
    return displayed_y_axis_overridden(tab, data) ? tab.filters.y_min : data.y_min;
}

double displayed_y_max(const FileTab &tab, const LoadedDataset &data) {
    return displayed_y_axis_overridden(tab, data) ? tab.filters.y_max : data.y_max;
}

const std::string &displayed_x_label(const FileTab &tab, const LoadedDataset &data) {
    return displayed_x_axis_overridden(tab, data) ? tab.filters.x_label : data.x_label;
}

const std::string &displayed_y_label(const FileTab &tab, const LoadedDataset &data) {
    return displayed_y_axis_overridden(tab, data) ? tab.filters.y_label : data.y_label;
}

std::string displayed_value_label(const FileTab &tab, const LoadedDataset &data) {
    if (filter_output_ready(tab, data) && !tab.filters.value_label.empty()) {
        return tab.filters.value_label;
    }
    return data.kind == LoadedKind::Line1D ? data.y_label : base_name(data.info.path);
}

const std::vector<double> &displayed_line_x_values(const FileTab &tab, const LoadedDataset &data) {
    return displayed_x_axis_overridden(tab, data) ? tab.filters.x_axis_values : data.line_x_values;
}

const std::vector<double> &displayed_heat_x_values(const FileTab &tab, const LoadedDataset &data) {
    return displayed_x_axis_overridden(tab, data) ? tab.filters.x_axis_values : data.heat_x_values;
}

const std::vector<double> &displayed_heat_y_values(const FileTab &tab, const LoadedDataset &data) {
    return displayed_y_axis_overridden(tab, data) ? tab.filters.y_axis_values : data.heat_y_values;
}

void start_filter_pipeline(FileTab &tab) {
    FilterPipelineState &filters = tab.filters;
    if (filters.processing || !filters.recompute_pending || !filter_pipeline_enabled(filters) || !tab.loaded) {
        return;
    }
    if (tab.loaded->kind != LoadedKind::Line1D && tab.loaded->kind != LoadedKind::Heatmap2D) {
        filters.recompute_pending = false;
        filters.error = "filters can only be applied to 1D or 2D numeric datasets";
        return;
    }

    filters.processing = true;
    filters.recompute_pending = false;
    filters.error.clear();
    const int token = filters.requested_token;
    filters.cancel_requested = std::make_shared<std::atomic<bool>>(false);
    filters.future = std::async(std::launch::async, filter_pipeline_worker, token, tab.loaded, filters.stages,
                                filters.cancel_requested);
}

void request_filter_recompute(FileTab &tab) {
    FilterPipelineState &filters = tab.filters;
    if (filters.cancel_requested) {
        filters.cancel_requested->store(true, std::memory_order_relaxed);
    }
    ++filters.requested_token;
    filters.recompute_pending = filter_pipeline_enabled(filters) && tab.loaded != nullptr;
    filters.error.clear();
    start_filter_pipeline(tab);
}

void clear_filter_output_for_new_data(FileTab &tab, bool preserve_axis_controls = false) {
    FilterPipelineState &filters = tab.filters;
    const bool restore_x_axis = filters.x_axis_override;
    const bool restore_y_axis = filters.y_axis_override;
    bool keep_previous_output =
        preserve_axis_controls && tab.loaded && filter_pipeline_enabled(filters) &&
        filters.output_source && filters.output_values.size() == source_plot_values(*tab.loaded).size();
    if (keep_previous_output) {
        const auto old_dimensions = plot_value_dimensions(*filters.output_source);
        const auto new_dimensions = plot_value_dimensions(*tab.loaded);
        keep_previous_output =
            filters.output_source->kind == tab.loaded->kind && old_dimensions == new_dimensions;
    }
    if (filters.cancel_requested) {
        filters.cancel_requested->store(true, std::memory_order_relaxed);
    }
    ++filters.requested_token;
    filters.preserve_axis_controls_on_next_result =
        preserve_axis_controls && (restore_x_axis || restore_y_axis);
    if (keep_previous_output) {
        filters.output_source = tab.loaded;
    } else {
        filters.output_source.reset();
        filters.output_values.clear();
        filters.output_token = 0;
        filters.x_axis_override = false;
        filters.y_axis_override = false;
        filters.x_axis_values.clear();
        filters.y_axis_values.clear();
        filters.x_label.clear();
        filters.y_label.clear();
        filters.value_label.clear();
    }
    if (!preserve_axis_controls && tab.loaded && restore_x_axis) {
        tab.controls.x.min = tab.loaded->x_min;
        tab.controls.x.max = tab.loaded->x_max;
        tab.controls.x.automatic = false;
    }
    if (!preserve_axis_controls && tab.loaded && restore_y_axis &&
        tab.loaded->kind == LoadedKind::Heatmap2D) {
        tab.controls.y.min = tab.loaded->y_min;
        tab.controls.y.max = tab.loaded->y_max;
        tab.controls.y.automatic = false;
    }
    filters.error.clear();
    if (tab.loaded) {
        for (PlotFilterStage &stage : filters.stages) {
            switch (stage.kind) {
            case PlotFilterKind::SavitzkyGolay:
                sanitize_savitzky_golay_parameters(*tab.loaded, stage.savitzky_golay);
                break;
            case PlotFilterKind::WindowedAverage:
                sanitize_windowed_average_parameters(*tab.loaded, stage.windowed_average);
                break;
            case PlotFilterKind::Fft:
                sanitize_fft_parameters(*tab.loaded, stage.fft);
                break;
            }
        }
    }
    filters.recompute_pending = filter_pipeline_enabled(filters) && tab.loaded != nullptr;
    tab.line_cache = {};
    start_filter_pipeline(tab);
}

bool poll_filter_pipeline(FileTab &tab) {
    FilterPipelineState &filters = tab.filters;
    bool changed = false;
    if (filters.processing && filters.future.valid()) {
        using namespace std::chrono_literals;
        if (filters.future.wait_for(0ms) == std::future_status::ready) {
            FilterPipelineResult result = filters.future.get();
            filters.processing = false;
            changed = true;
            if (result.token == filters.requested_token && result.source == tab.loaded &&
                filter_pipeline_enabled(filters)) {
                const bool preserve_axis_controls = filters.preserve_axis_controls_on_next_result;
                filters.preserve_axis_controls_on_next_result = false;
                if (result.ok) {
                    const bool x_axis_changed =
                        filters.x_axis_override != result.x_axis_override ||
                        (result.x_axis_override &&
                         (filters.x_min != result.x_min || filters.x_max != result.x_max ||
                          filters.x_label != result.x_label));
                    const bool y_axis_changed =
                        filters.y_axis_override != result.y_axis_override ||
                        (result.y_axis_override &&
                         (filters.y_min != result.y_min || filters.y_max != result.y_max ||
                          filters.y_label != result.y_label));
                    filters.output_source = std::move(result.source);
                    filters.output_values = std::move(result.values);
                    filters.output_min = result.value_min;
                    filters.output_max = result.value_max;
                    filters.output_token = result.token;
                    filters.x_axis_override = result.x_axis_override;
                    filters.y_axis_override = result.y_axis_override;
                    filters.x_axis_values = std::move(result.x_axis_values);
                    filters.y_axis_values = std::move(result.y_axis_values);
                    filters.x_min = result.x_min;
                    filters.x_max = result.x_max;
                    filters.y_min = result.y_min;
                    filters.y_max = result.y_max;
                    filters.x_label = std::move(result.x_label);
                    filters.y_label = std::move(result.y_label);
                    filters.value_label = std::move(result.value_label);
                    filters.error.clear();
                    tab.line_cache = {};
                    if (x_axis_changed && tab.loaded && !preserve_axis_controls) {
                        tab.controls.x.min = filters.x_axis_override ? filters.x_min : tab.loaded->x_min;
                        tab.controls.x.max = filters.x_axis_override ? filters.x_max : tab.loaded->x_max;
                        tab.controls.x.automatic = false;
                        tab.slices.heatmap_view_valid = false;
                    }
                    if (y_axis_changed && tab.loaded && tab.loaded->kind == LoadedKind::Heatmap2D &&
                        !preserve_axis_controls) {
                        tab.controls.y.min = filters.y_axis_override ? filters.y_min : tab.loaded->y_min;
                        tab.controls.y.max = filters.y_axis_override ? filters.y_max : tab.loaded->y_max;
                        tab.controls.y.automatic = false;
                        tab.slices.heatmap_view_valid = false;
                    }
                } else {
                    filters.error = std::move(result.error);
                }
            }
        }
    }
    if (!filters.processing && filters.recompute_pending) {
        start_filter_pipeline(tab);
        changed = true;
    }
    return changed;
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

void update_heat_texture_colors(FileTab &tab, LoadedDataset &data, const std::vector<float> &values,
                                int value_generation, float min_value, float max_value) {
    if (data.kind != LoadedKind::Heatmap2D || values.empty() || tab.heat_texture == 0) {
        return;
    }
    const auto nearly_same = [](float lhs, float rhs) {
        const float scale = std::max({1.0f, std::abs(lhs), std::abs(rhs)});
        return std::abs(lhs - rhs) <= scale * 1e-5f;
    };
    if (data.applied_color_generation == value_generation && std::isfinite(data.applied_color_min) &&
        std::isfinite(data.applied_color_max) &&
        nearly_same(data.applied_color_min, min_value) && nearly_same(data.applied_color_max, max_value)) {
        return;
    }

    std::vector<uint8_t> rgba = make_turbo_rgba(values, min_value, max_value);
    glBindTexture(GL_TEXTURE_2D, tab.heat_texture);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, data.texture_width, data.texture_height, GL_RGBA, GL_UNSIGNED_BYTE, rgba.data());
    glBindTexture(GL_TEXTURE_2D, 0);
    data.applied_color_min = min_value;
    data.applied_color_max = max_value;
    data.applied_color_generation = value_generation;
}

void invalidate_plot(FileTab &tab) {
    tab.loaded.reset();
    clear_filter_output_for_new_data(tab);
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
    const auto [display_min, display_max] = displayed_value_range(tab, data);
    if (controls.x.automatic) {
        controls.x.min = displayed_x_min(tab, data);
        controls.x.max = displayed_x_max(tab, data);
    }
    if (controls.y.automatic) {
        controls.y.min = data.kind == LoadedKind::Line1D ? display_min : displayed_y_min(tab, data);
        controls.y.max = data.kind == LoadedKind::Line1D ? display_max : displayed_y_max(tab, data);
    }
    if (data.kind == LoadedKind::Heatmap2D) {
        if (controls.color.auto_min) {
            controls.color.min = display_min;
        }
        if (controls.color.auto_max) {
            controls.color.max = display_max;
        }
    }
    normalize_range(controls.x.min, controls.x.max);
    normalize_range(controls.y.min, controls.y.max);
    return controls;
}

std::pair<float, float> effective_color_range(const ColorRangeControl &control, float auto_min, float auto_max) {
    const double raw_min = control.auto_min ? static_cast<double>(auto_min) : control.min;
    const double raw_max = control.auto_max ? static_cast<double>(auto_max) : control.max;
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
        tab.sort_x_axis_increasing = false;
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
    if (!x_axis.path.empty()) {
        auto it = std::find_if(tab.datasets.begin(), tab.datasets.end(), [&](const DatasetInfo &candidate) {
            return candidate.path == x_axis.path;
        });
        if (it != tab.datasets.end() && vector_axis_length(*it) == x_fallback_count) {
            x_axis_info = *it;
            has_x_axis_info = true;
        }
    }
    DatasetInfo y_axis_info;
    bool has_y_axis_info = false;
    if (info.dims.size() == 2 && !y_axis.path.empty()) {
        auto it = std::find_if(tab.datasets.begin(), tab.datasets.end(), [&](const DatasetInfo &candidate) {
            return candidate.path == y_axis.path;
        });
        if (it != tab.datasets.end() && vector_axis_length(*it) == y_fallback_count) {
            y_axis_info = *it;
            has_y_axis_info = true;
        }
    }

    const int token = ++tab.load_token;
    LoadConfig config;
    config.max_texture_side = std::max(256, std::min(app.gl_max_texture_size, 4096));
    config.transpose_2d = info.dims.size() == 2 && tab.transpose_2d;
    config.sort_x_axis_increasing = tab.sort_x_axis_increasing;
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
                                 x_axis_info, has_x_axis_info, y_axis_info, has_y_axis_info, config);
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
        clear_filter_output_for_new_data(tab, preserve_controls);
    }
    if (tab.loaded && tab.loaded->kind == LoadedKind::Heatmap2D) {
        clamp_slice_state(tab, *tab.loaded);
        upload_heat_texture(tab, *tab.loaded);
        if (preserve_controls && (!tab.controls.color.auto_min || !tab.controls.color.auto_max)) {
            const auto [display_min, display_max] = displayed_value_range(tab, *tab.loaded);
            const auto [color_min, color_max] =
                effective_color_range(tab.controls.color, display_min, display_max);
            update_heat_texture_colors(tab, *tab.loaded, displayed_plot_values(tab, *tab.loaded),
                                       displayed_value_generation(tab, *tab.loaded), color_min, color_max);
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
    tab.slices = {};
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

double x_at_index(const std::vector<double> &x_values, size_t index, size_t source_size, double x_min, double x_max) {
    if (!x_values.empty() && index < x_values.size()) {
        return x_values[index];
    }
    if (source_size <= 1 || x_max == x_min) {
        return static_cast<double>(index);
    }
    const double t = static_cast<double>(index) / static_cast<double>(source_size - 1);
    return x_min + t * (x_max - x_min);
}

void rebuild_line_cache(const std::vector<float> &values, const std::vector<double> &x_values, bool x_increasing,
                        bool x_decreasing, double x_min, double x_max, LineCache &cache, double view_min,
                        double view_max, int pixel_width) {
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

    const double data_min = std::min(x_min, x_max);
    const double data_max = std::max(x_min, x_max);
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
                           cache.x_source_ptr == (x_values.empty() ? nullptr : x_values.data()) &&
                           cache.data_x_min == x_min && cache.data_x_max == x_max &&
                           nearly_same(cache.view_x_min, view_min) && nearly_same(cache.view_x_max, view_max) &&
                           cache.pixel_width == pixel_width;
    if (cache_hit) {
        return;
    }

    cache = {};
    cache.valid = true;
    cache.source_ptr = values.data();
    cache.x_source_ptr = x_values.empty() ? nullptr : x_values.data();
    cache.source_size = n;
    cache.data_x_min = x_min;
    cache.data_x_max = x_max;
    cache.view_x_min = view_min;
    cache.view_x_max = view_max;
    cache.pixel_width = pixel_width;

    const auto index_for_x = [&](double x) -> size_t {
        if (!x_values.empty() && x_increasing) {
            auto it = std::lower_bound(x_values.begin(), x_values.end(), x);
            if (it == x_values.end()) {
                return n - 1;
            }
            return static_cast<size_t>(std::distance(x_values.begin(), it));
        }
        if (!x_values.empty() && x_decreasing) {
            auto it = std::lower_bound(x_values.begin(), x_values.end(), x, std::greater<double>());
            if (it == x_values.end()) {
                return n - 1;
            }
            return static_cast<size_t>(std::distance(x_values.begin(), it));
        }
        if (x_max == x_min || n <= 1) {
            return 0;
        }
        const double t = (x - x_min) / (x_max - x_min);
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
            cache.x[i] = x_at_index(x_values, source_index, n, x_min, x_max);
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
            const double x = x_at_index(x_values, std::min(a, n - 1), n, x_min, x_max);
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

bool heatmap_slice_ready(const LoadedDataset &data) {
    return data.kind == LoadedKind::Heatmap2D && data.texture_width > 0 && data.texture_height > 0 &&
           data.heat_values.size() == static_cast<size_t>(data.texture_width) * static_cast<size_t>(data.texture_height);
}

void clamp_slice_state(FileTab &tab, const LoadedDataset &data) {
    if (!heatmap_slice_ready(data)) {
        tab.slices.row_index = 0;
        tab.slices.column_index = 0;
        return;
    }
    tab.slices.row_index = std::clamp(tab.slices.row_index, 0, data.texture_height - 1);
    tab.slices.column_index = std::clamp(tab.slices.column_index, 0, data.texture_width - 1);
}

double coordinate_for_index(int index, double axis_min, double axis_max, int count) {
    if (count <= 1 || axis_min == axis_max) {
        return axis_min;
    }
    const double t = static_cast<double>(std::clamp(index, 0, count - 1)) / static_cast<double>(count - 1);
    return axis_min + t * (axis_max - axis_min);
}

int index_for_coordinate(double coordinate, double axis_min, double axis_max, int count) {
    if (count <= 1 || axis_min == axis_max || !std::isfinite(coordinate)) {
        return 0;
    }
    const double t = (coordinate - axis_min) / (axis_max - axis_min);
    const double clamped = std::clamp(std::isfinite(t) ? t : 0.0, 0.0, 1.0);
    return std::clamp(static_cast<int>(std::llround(clamped * static_cast<double>(count - 1))), 0, count - 1);
}

bool can_plot_axis_values(const std::vector<double> &values, int count) {
    return values.size() == static_cast<size_t>(count) && is_monotonic_increasing(values);
}

void remember_heatmap_view(FileTab &tab, const ImPlotRect &limits) {
    tab.slices.view_x_min = limits.X.Min;
    tab.slices.view_x_max = limits.X.Max;
    tab.slices.view_y_min = limits.Y.Min;
    tab.slices.view_y_max = limits.Y.Max;
    tab.slices.heatmap_view_valid = std::isfinite(tab.slices.view_x_min) && std::isfinite(tab.slices.view_x_max) &&
                                    std::isfinite(tab.slices.view_y_min) && std::isfinite(tab.slices.view_y_max);
}

std::pair<double, double> slice_axis_view_range(const SliceState &slices, double axis_min, double axis_max,
                                                bool row_slice) {
    double min_value = axis_min;
    double max_value = axis_max;
    if (slices.heatmap_view_valid) {
        min_value = row_slice ? slices.view_x_min : slices.view_y_min;
        max_value = row_slice ? slices.view_x_max : slices.view_y_max;
    }
    return effective_range(min_value, max_value);
}

std::pair<double, double> visible_slice_value_range(const std::vector<double> &x, const std::vector<double> &y,
                                                    double view_min, double view_max) {
    if (view_min > view_max) {
        std::swap(view_min, view_max);
    }

    double min_value = std::numeric_limits<double>::infinity();
    double max_value = -std::numeric_limits<double>::infinity();
    double closest_distance = std::numeric_limits<double>::infinity();
    double closest_value = std::numeric_limits<double>::quiet_NaN();
    const double view_center = 0.5 * (view_min + view_max);
    for (size_t i = 0; i < x.size() && i < y.size(); ++i) {
        if (!std::isfinite(x[i]) || !std::isfinite(y[i])) {
            continue;
        }
        const double distance = std::abs(x[i] - view_center);
        if (distance < closest_distance) {
            closest_distance = distance;
            closest_value = y[i];
        }
        if (x[i] < view_min || x[i] > view_max) {
            continue;
        }
        min_value = std::min(min_value, y[i]);
        max_value = std::max(max_value, y[i]);
    }

    if (!std::isfinite(min_value) || !std::isfinite(max_value)) {
        if (!std::isfinite(closest_value)) {
            return finite_minmax(y);
        }
        min_value = closest_value;
        max_value = closest_value;
    }
    normalize_range(min_value, max_value);
    return {min_value, max_value};
}

hsize_t display_row_source_index(const FileTab &tab, const LoadedDataset &data) {
    if (tab.transpose_2d) {
        return static_cast<hsize_t>(hdf5_plotter::preview_source_index(
            static_cast<size_t>(tab.slices.row_index), static_cast<size_t>(data.col_stride),
            static_cast<size_t>(data.source_cols)));
    }
    return static_cast<hsize_t>(hdf5_plotter::preview_source_index(
        static_cast<size_t>(tab.slices.row_index), static_cast<size_t>(data.row_stride),
        static_cast<size_t>(data.source_rows)));
}

hsize_t display_column_source_index(const FileTab &tab, const LoadedDataset &data) {
    if (tab.transpose_2d) {
        return static_cast<hsize_t>(hdf5_plotter::preview_source_index(
            static_cast<size_t>(tab.slices.column_index), static_cast<size_t>(data.row_stride),
            static_cast<size_t>(data.source_rows)));
    }
    return static_cast<hsize_t>(hdf5_plotter::preview_source_index(
        static_cast<size_t>(tab.slices.column_index), static_cast<size_t>(data.col_stride),
        static_cast<size_t>(data.source_cols)));
}

void build_row_slice(const LoadedDataset &data, const std::vector<float> &values,
                     const std::vector<double> &axis_values, double axis_min, double axis_max, int row_index,
                     std::vector<double> &x, std::vector<double> &y) {
    const int width = std::max(0, data.texture_width);
    x.resize(static_cast<size_t>(width));
    y.resize(static_cast<size_t>(width));
    const int row = std::clamp(row_index, 0, std::max(0, data.texture_height - 1));
    const bool use_axis_values = can_plot_axis_values(axis_values, width);
    for (int col = 0; col < width; ++col) {
        x[static_cast<size_t>(col)] = use_axis_values ? axis_values[static_cast<size_t>(col)]
                                                     : coordinate_for_index(col, axis_min, axis_max, width);
        y[static_cast<size_t>(col)] =
            static_cast<double>(values[static_cast<size_t>(row) * static_cast<size_t>(width) + static_cast<size_t>(col)]);
    }
}

void build_column_slice(const LoadedDataset &data, const std::vector<float> &values,
                        const std::vector<double> &axis_values, double axis_min, double axis_max, int column_index,
                        std::vector<double> &x, std::vector<double> &y) {
    const int height = std::max(0, data.texture_height);
    const int width = std::max(0, data.texture_width);
    x.resize(static_cast<size_t>(height));
    y.resize(static_cast<size_t>(height));
    const int column = std::clamp(column_index, 0, std::max(0, width - 1));
    const bool use_axis_values = can_plot_axis_values(axis_values, height);
    for (int row = 0; row < height; ++row) {
        x[static_cast<size_t>(row)] = use_axis_values ? axis_values[static_cast<size_t>(row)]
                                                     : coordinate_for_index(row, axis_min, axis_max, height);
        y[static_cast<size_t>(row)] =
            static_cast<double>(values[static_cast<size_t>(row) * static_cast<size_t>(width) + static_cast<size_t>(column)]);
    }
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

void draw_slice_controls(FileTab &tab, LoadedDataset &data) {
    if (!heatmap_slice_ready(data)) {
        return;
    }

    bool enabled = tab.slices.enabled;
    if (ImGui::Checkbox("Slice plots", &enabled)) {
        tab.slices.enabled = enabled;
        if (enabled) {
            tab.slices.row_window_open = true;
            tab.slices.column_window_open = true;
            clamp_slice_state(tab, data);
        }
    }
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("Open dockable row/column slice plots. Click or drag visible guides to choose indices.");
    }

    if (!tab.slices.enabled) {
        return;
    }

    clamp_slice_state(tab, data);
    ImGui::SameLine();
    ImGui::TextDisabled("click visible guides or type indices");

    const float input_width = ImGui::CalcTextSize("88888888").x + ImGui::GetStyle().FramePadding.x * 2.0f;
    int row = tab.slices.row_index;
    ImGui::SetNextItemWidth(input_width);
    if (ImGui::InputInt("Row", &row, 1, 10)) {
        tab.slices.row_index = std::clamp(row, 0, data.texture_height - 1);
    }
    ImGui::SameLine();
    int column = tab.slices.column_index;
    ImGui::SetNextItemWidth(input_width);
    if (ImGui::InputInt("Column", &column, 1, 10)) {
        tab.slices.column_index = std::clamp(column, 0, data.texture_width - 1);
    }

    ImGui::Checkbox("Row window", &tab.slices.row_window_open);
    ImGui::SameLine();
    ImGui::Checkbox("Column window", &tab.slices.column_window_open);
    if (displayed_x_axis_overridden(tab, data) || displayed_y_axis_overridden(tab, data)) {
        ImGui::TextDisabled("Row %.6g; column %.6g",
                            coordinate_for_index(tab.slices.row_index, displayed_y_min(tab, data),
                                                 displayed_y_max(tab, data), data.texture_height),
                            coordinate_for_index(tab.slices.column_index, displayed_x_min(tab, data),
                                                 displayed_x_max(tab, data), data.texture_width));
    } else {
        ImGui::TextDisabled("Preview row %d -> source %s %s; preview column %d -> source %s %s",
                            tab.slices.row_index, tab.transpose_2d ? "column" : "row",
                            count_label(display_row_source_index(tab, data)).c_str(), tab.slices.column_index,
                            tab.transpose_2d ? "row" : "column",
                            count_label(display_column_source_index(tab, data)).c_str());
    }
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
            start_load(app, tab, tab.selected_index, true, true, true);
        }
        if (ImGui::Selectable(index_label, axis_index == -1)) {
            axis_index = -1;
            start_load(app, tab, tab.selected_index, true, true, true);
        }
        ImGui::Separator();
        for (int candidate_index : compatible) {
            const DatasetInfo &axis = tab.datasets[static_cast<size_t>(candidate_index)];
            if (ImGui::Selectable(axis.path.c_str(), axis_index == candidate_index)) {
                axis_index = candidate_index;
                start_load(app, tab, tab.selected_index, true, true, true);
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
            start_load(app, tab, tab.selected_index, true, true, true);
        }
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("Transpose 2D datasets so rows are shown on X and columns on Y.");
        }
    }
    const AxisGuess guessed_axes = guess_axes(tab.datasets, target);
    const std::vector<int> compatible_x = compatible_x_axis_indices(tab.datasets, target, tab.transpose_2d);
    draw_axis_combo(app, tab, "X axis", tab.x_axis_index, compatible_x,
                    target.dims.size() == 1 ? "Index" : tab.transpose_2d ? "Row" : "Column");
    AxisSpec active_x_axis;
    if (target.dims.size() == 1) {
        active_x_axis = selected_axis(tab.datasets, tab.x_axis_index, target.dims[0], guessed_axes.x);
    } else if (target.dims.size() == 2) {
        const bool transposed = tab.transpose_2d;
        active_x_axis = selected_axis(tab.datasets, tab.x_axis_index, target.dims[transposed ? 0 : 1],
                                      transposed ? guessed_axes.y : guessed_axes.x);
    }
    const bool has_x_axis_values = !active_x_axis.path.empty();
    ImGui::BeginDisabled(!has_x_axis_values);
    bool sort_x_axis = tab.sort_x_axis_increasing;
    if (ImGui::Checkbox("Sort X ascending", &sort_x_axis)) {
        tab.sort_x_axis_increasing = sort_x_axis;
        start_load(app, tab, tab.selected_index, true, true, true);
    }
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("Sort the selected X-axis values into increasing order and reorder plotted samples or columns to match. Leave off to preserve file index order.");
    }
    ImGui::EndDisabled();
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
        const auto [display_min, display_max] = displayed_value_range(tab, data);
        draw_range_control("Auto X", tab.controls.x, displayed_x_min(tab, data), displayed_x_max(tab, data));
        const double y_auto_min = data.kind == LoadedKind::Line1D ? display_min : displayed_y_min(tab, data);
        const double y_auto_max = data.kind == LoadedKind::Line1D ? display_max : displayed_y_max(tab, data);
        draw_range_control("Auto Y", tab.controls.y, y_auto_min, y_auto_max);
        if (data.kind == LoadedKind::Heatmap2D) {
            draw_color_range_control(tab.controls.color, display_min, display_max);
            draw_slice_controls(tab, data);
        }
        if (!displayed_x_axis_overridden(tab, data) && looks_like_unix_time_axis(data)) {
            ImGui::Checkbox("Format X as date/time", &tab.x_datetime);
            if (tab.x_datetime) {
                ImGui::SameLine();
                ImGui::Checkbox("UTC", &tab.x_datetime_utc);
            }
        }
    }
}

void draw_heatmap_slice_overlay(FileTab &tab, LoadedDataset &data) {
    if (!tab.slices.enabled || !heatmap_slice_ready(data)) {
        return;
    }

    clamp_slice_state(tab, data);
    const double x_min = displayed_x_min(tab, data);
    const double x_max = displayed_x_max(tab, data);
    const double y_min = displayed_y_min(tab, data);
    const double y_max = displayed_y_max(tab, data);
    const bool show_row_slice = tab.slices.row_window_open;
    const bool show_column_slice = tab.slices.column_window_open;
    if (ImPlot::IsPlotHovered() && ImGui::IsMouseClicked(ImGuiMouseButton_Left)) {
        const ImPlotPoint mouse = ImPlot::GetPlotMousePos();
        if (show_column_slice) {
            tab.slices.column_index = index_for_coordinate(mouse.x, x_min, x_max, data.texture_width);
        }
        if (show_row_slice) {
            tab.slices.row_index = index_for_coordinate(mouse.y, y_min, y_max, data.texture_height);
        }
    }

    double slice_x = coordinate_for_index(tab.slices.column_index, x_min, x_max, data.texture_width);
    double slice_y = coordinate_for_index(tab.slices.row_index, y_min, y_max, data.texture_height);
    const ImVec4 column_color(1.0f, 0.88f, 0.10f, 1.0f);
    const ImVec4 row_color(0.10f, 0.95f, 1.0f, 1.0f);
    const ImPlotDragToolFlags flags = ImPlotDragToolFlags_NoFit;

    if (show_column_slice && ImPlot::DragLineX(10000 + tab.id * 2, &slice_x, column_color, 1.5f, flags)) {
        tab.slices.column_index = index_for_coordinate(slice_x, x_min, x_max, data.texture_width);
    }
    if (show_row_slice && ImPlot::DragLineY(10001 + tab.id * 2, &slice_y, row_color, 1.5f, flags)) {
        tab.slices.row_index = index_for_coordinate(slice_y, y_min, y_max, data.texture_height);
    }

    if (show_column_slice) {
        const std::string source_index = count_label(display_column_source_index(tab, data));
        ImPlot::TagX(coordinate_for_index(tab.slices.column_index, x_min, x_max, data.texture_width), column_color,
                     "%s %s", tab.transpose_2d ? "row" : "column", source_index.c_str());
    }
    if (show_row_slice) {
        const std::string source_index = count_label(display_row_source_index(tab, data));
        ImPlot::TagY(coordinate_for_index(tab.slices.row_index, y_min, y_max, data.texture_height), row_color,
                     "%s %s", tab.transpose_2d ? "column" : "row", source_index.c_str());
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
        const std::vector<float> &plot_values = displayed_plot_values(tab, data);
        const std::vector<double> &x_values = displayed_line_x_values(tab, data);
        const double x_min = displayed_x_min(tab, data);
        const double x_max = displayed_x_max(tab, data);
        const bool x_overridden = displayed_x_axis_overridden(tab, data);
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
            ImPlot::SetupAxes(displayed_x_label(tab, data).c_str(), displayed_value_label(tab, data).c_str());
            AxisValueFormatterData x_formatter;
            if (!x_overridden) {
                setup_x_axis_format(tab, data, x_formatter);
            }
            const ImPlotRect limits = ImPlot::GetPlotLimits();
            const double cache_min = tab.controls.x.automatic ? x_min : limits.X.Min;
            const double cache_max = tab.controls.x.automatic ? x_max : limits.X.Max;
            rebuild_line_cache(plot_values, x_values, x_overridden || data.x_values_increasing,
                               !x_overridden && data.x_values_decreasing, x_min, x_max, tab.line_cache, cache_min,
                               cache_max, plot_width);
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
        const std::vector<float> &plot_values = displayed_plot_values(tab, data);
        const double x_min = displayed_x_min(tab, data);
        const double x_max = displayed_x_max(tab, data);
        const double y_min = displayed_y_min(tab, data);
        const double y_max = displayed_y_max(tab, data);
        const auto [display_min, display_max] = displayed_value_range(tab, data);
        const auto [color_min, color_max] = effective_color_range(tab.controls.color, display_min, display_max);
        update_heat_texture_colors(tab, data, plot_values, displayed_value_generation(tab, data), color_min, color_max);
        if (filter_output_ready(tab, data) && !tab.filters.value_label.empty()) {
            ImGui::TextUnformatted(tab.filters.value_label.c_str());
        }
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
            ImPlot::SetupAxes(displayed_x_label(tab, data).c_str(), displayed_y_label(tab, data).c_str());
            AxisValueFormatterData x_formatter;
            AxisValueFormatterData y_formatter;
            if (!displayed_x_axis_overridden(tab, data)) {
                setup_x_axis_format(tab, data, x_formatter);
            }
            if (!displayed_y_axis_overridden(tab, data)) {
                setup_y_axis_format(data, ImAxis_Y1, y_formatter);
            }
            ImPlot::PlotImage(base_name(data.info.path).c_str(), reinterpret_cast<ImTextureID>(static_cast<intptr_t>(tab.heat_texture)),
                              ImPlotPoint(x_min, y_min), ImPlotPoint(x_max, y_max), ImVec2(0, 1), ImVec2(1, 0));
            remember_heatmap_view(tab, ImPlot::GetPlotLimits());
            draw_heatmap_slice_overlay(tab, data);
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

std::string row_slice_window_title(const FileTab &tab) {
    return "Row Slice: " + tab_title(tab) + "##row-slice-" + std::to_string(tab.id);
}

std::string column_slice_window_title(const FileTab &tab) {
    return "Column Slice: " + tab_title(tab) + "##column-slice-" + std::to_string(tab.id);
}

void draw_slice_line_plot(FileTab &tab, LoadedDataset &data, bool row_slice) {
    if (!heatmap_slice_ready(data)) {
        ImGui::TextUnformatted("No 2D preview is loaded.");
        return;
    }

    clamp_slice_state(tab, data);
    const float input_width = ImGui::CalcTextSize("88888888").x + ImGui::GetStyle().FramePadding.x * 2.0f;
    if (row_slice) {
        int row = tab.slices.row_index;
        ImGui::SetNextItemWidth(input_width);
        if (ImGui::InputInt("Row", &row, 1, 10)) {
            tab.slices.row_index = std::clamp(row, 0, data.texture_height - 1);
        }
        ImGui::SameLine();
        if (displayed_y_axis_overridden(tab, data)) {
            ImGui::TextDisabled("%.6g",
                                coordinate_for_index(tab.slices.row_index, displayed_y_min(tab, data),
                                                     displayed_y_max(tab, data), data.texture_height));
        } else {
            ImGui::TextDisabled("source %s %s", tab.transpose_2d ? "column" : "row",
                                count_label(display_row_source_index(tab, data)).c_str());
        }
    } else {
        int column = tab.slices.column_index;
        ImGui::SetNextItemWidth(input_width);
        if (ImGui::InputInt("Column", &column, 1, 10)) {
            tab.slices.column_index = std::clamp(column, 0, data.texture_width - 1);
        }
        ImGui::SameLine();
        if (displayed_x_axis_overridden(tab, data)) {
            ImGui::TextDisabled("%.6g",
                                coordinate_for_index(tab.slices.column_index, displayed_x_min(tab, data),
                                                     displayed_x_max(tab, data), data.texture_width));
        } else {
            ImGui::TextDisabled("source %s %s", tab.transpose_2d ? "row" : "column",
                                count_label(display_column_source_index(tab, data)).c_str());
        }
    }

    std::vector<double> x;
    std::vector<double> y;
    const std::vector<float> &plot_values = displayed_plot_values(tab, data);
    const std::vector<double> &axis_values =
        row_slice ? displayed_heat_x_values(tab, data) : displayed_heat_y_values(tab, data);
    const double axis_min = row_slice ? displayed_x_min(tab, data) : displayed_y_min(tab, data);
    const double axis_max = row_slice ? displayed_x_max(tab, data) : displayed_y_max(tab, data);
    if (row_slice) {
        build_row_slice(data, plot_values, axis_values, axis_min, axis_max, tab.slices.row_index, x, y);
    } else {
        build_column_slice(data, plot_values, axis_values, axis_min, axis_max, tab.slices.column_index, x, y);
    }
    if (x.empty() || y.empty()) {
        ImGui::TextUnformatted("Slice is empty.");
        return;
    }

    const auto [view_min, view_max] = slice_axis_view_range(tab.slices, axis_min, axis_max, row_slice);
    const auto [value_min, value_max] = visible_slice_value_range(x, y, view_min, view_max);
    ImPlot::SetNextAxisLimits(ImAxis_X1, view_min, view_max, ImPlotCond_Always);
    ImPlot::SetNextAxisLimits(ImAxis_Y1, value_min, value_max, ImPlotCond_Always);
    const char *plot_id = row_slice ? "##row-slice-plot" : "##column-slice-plot";
    if (ImPlot::BeginPlot(plot_id, ImVec2(-1, -1))) {
        const std::string &x_label = row_slice ? displayed_x_label(tab, data) : displayed_y_label(tab, data);
        ImPlot::SetupAxes(x_label.c_str(), displayed_value_label(tab, data).c_str());
        AxisValueFormatterData x_formatter;
        const bool row_uses_axis_values = row_slice && can_plot_axis_values(axis_values, data.texture_width);
        const bool column_uses_axis_values =
            !row_slice && can_plot_axis_values(axis_values, data.texture_height);
        if (row_slice && !displayed_x_axis_overridden(tab, data) && !row_uses_axis_values) {
            setup_x_axis_format(tab, data, x_formatter);
        } else if (!row_slice && !displayed_y_axis_overridden(tab, data) &&
                   !column_uses_axis_values) {
            setup_y_axis_format(data, ImAxis_X1, x_formatter);
        } else if (row_slice && !displayed_x_axis_overridden(tab, data) && tab.x_datetime &&
                   looks_like_unix_time_axis(data)) {
            ImPlot::SetupAxisFormat(ImAxis_X1, unix_time_formatter, &tab.x_datetime_utc);
        }
        ImPlot::SetNextLineStyle(row_slice ? ImVec4(0.10f, 0.95f, 1.0f, 1.0f) : ImVec4(1.0f, 0.88f, 0.10f, 1.0f),
                                 1.8f);
        ImPlot::PlotLine(row_slice ? "row" : "column", x.data(), y.data(), static_cast<int>(y.size()),
                         ImPlotLineFlags_SkipNaN);
        ImPlot::EndPlot();
    }
}

void draw_slice_windows(AppState &app, FileTab &tab) {
    if (!tab.slices.enabled || !tab.loaded || !heatmap_slice_ready(*tab.loaded)) {
        return;
    }

    LoadedDataset &data = *tab.loaded;
    constexpr ImGuiWindowFlags slice_window_flags = ImGuiWindowFlags_NoFocusOnAppearing;
    bool slice_window_appeared = false;
    if (app.plot_dock_id != 0) {
        ImGui::SetNextWindowDockID(app.plot_dock_id, ImGuiCond_FirstUseEver);
    }
    ImGui::SetNextWindowSize(ImVec2(620, 300), ImGuiCond_FirstUseEver);
    if (tab.slices.row_window_open) {
        bool open = tab.slices.row_window_open;
        const std::string title = row_slice_window_title(tab);
        const bool visible = ImGui::Begin(title.c_str(), &open, slice_window_flags);
        slice_window_appeared |= ImGui::IsWindowAppearing();
        if (visible) {
            if (ImGui::IsWindowFocused(ImGuiFocusedFlags_RootAndChildWindows)) {
                app.active_tab_id = tab.id;
            }
            draw_slice_line_plot(tab, data, true);
        }
        ImGui::End();
        tab.slices.row_window_open = open;
    }

    if (app.plot_dock_id != 0) {
        ImGui::SetNextWindowDockID(app.plot_dock_id, ImGuiCond_FirstUseEver);
    }
    ImGui::SetNextWindowSize(ImVec2(620, 300), ImGuiCond_FirstUseEver);
    if (tab.slices.column_window_open) {
        bool open = tab.slices.column_window_open;
        const std::string title = column_slice_window_title(tab);
        const bool visible = ImGui::Begin(title.c_str(), &open, slice_window_flags);
        slice_window_appeared |= ImGui::IsWindowAppearing();
        if (visible) {
            if (ImGui::IsWindowFocused(ImGuiFocusedFlags_RootAndChildWindows)) {
                app.active_tab_id = tab.id;
            }
            draw_slice_line_plot(tab, data, false);
        }
        ImGui::End();
        tab.slices.column_window_open = open;
    }
    if (slice_window_appeared) {
        const std::string title = plot_window_title(tab);
        ImGui::SetWindowFocus(title.c_str());
    }
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

bool supports_savitzky_golay(const FileTab &tab) {
    if (!tab.loaded) {
        return false;
    }
    const LoadedDataset &data = *tab.loaded;
    if (data.kind == LoadedKind::Line1D) {
        return data.line_values.size() >= 3;
    }
    return data.kind == LoadedKind::Heatmap2D && (data.texture_width >= 3 || data.texture_height >= 3);
}

bool supports_windowed_average(const FileTab &tab) {
    return supports_savitzky_golay(tab);
}

bool supports_fft(const FileTab &tab) {
    if (!tab.loaded) {
        return false;
    }
    const LoadedDataset &data = *tab.loaded;
    if (data.kind == LoadedKind::Line1D) {
        return data.line_values.size() >= 2;
    }
    return data.kind == LoadedKind::Heatmap2D && (data.texture_width >= 2 || data.texture_height >= 2);
}

void restore_source_after_filters_disabled(FileTab &tab) {
    if (!tab.loaded) {
        return;
    }
    FilterPipelineState &filters = tab.filters;
    const bool restore_x = filters.x_axis_override;
    const bool restore_y = filters.y_axis_override;
    filters.output_source.reset();
    filters.output_values.clear();
    filters.output_token = 0;
    filters.x_axis_override = false;
    filters.y_axis_override = false;
    filters.x_axis_values.clear();
    filters.y_axis_values.clear();
    filters.x_label.clear();
    filters.y_label.clear();
    filters.value_label.clear();
    filters.preserve_axis_controls_on_next_result = false;
    if (restore_x) {
        tab.controls.x.min = tab.loaded->x_min;
        tab.controls.x.max = tab.loaded->x_max;
        tab.controls.x.automatic = false;
    }
    if (restore_y && tab.loaded->kind == LoadedKind::Heatmap2D) {
        tab.controls.y.min = tab.loaded->y_min;
        tab.controls.y.max = tab.loaded->y_max;
        tab.controls.y.automatic = false;
    }
    tab.slices.heatmap_view_valid = false;
    tab.line_cache = {};
}

void set_filter_stage_enabled(FileTab &tab, PlotFilterStage &stage, bool enabled) {
    if (stage.enabled == enabled) {
        return;
    }
    stage.enabled = enabled;
    if (enabled && tab.loaded) {
        switch (stage.kind) {
        case PlotFilterKind::SavitzkyGolay:
            sanitize_savitzky_golay_parameters(*tab.loaded, stage.savitzky_golay);
            break;
        case PlotFilterKind::WindowedAverage:
            sanitize_windowed_average_parameters(*tab.loaded, stage.windowed_average);
            break;
        case PlotFilterKind::Fft:
            sanitize_fft_parameters(*tab.loaded, stage.fft);
            break;
        }
    }
    tab.line_cache = {};
    request_filter_recompute(tab);
    if (!filter_pipeline_enabled(tab.filters)) {
        restore_source_after_filters_disabled(tab);
    }
}

const char *filter_stage_name(PlotFilterKind kind) {
    switch (kind) {
    case PlotFilterKind::SavitzkyGolay:
        return "Savitzky-Golay";
    case PlotFilterKind::WindowedAverage:
        return "Windowed Average";
    case PlotFilterKind::Fft:
        return "FFT";
    }
    return "Filter";
}

bool filter_stage_available(const FileTab &tab, PlotFilterKind kind) {
    switch (kind) {
    case PlotFilterKind::SavitzkyGolay:
        return supports_savitzky_golay(tab);
    case PlotFilterKind::WindowedAverage:
        return supports_windowed_average(tab);
    case PlotFilterKind::Fft:
        return supports_fft(tab);
    }
    return false;
}

void draw_filter_pipeline_order(FileTab &tab) {
    ImGui::SeparatorText("Pipeline order");

    bool moved = false;
    size_t move_from = 0;
    size_t move_to = 0;
    const float controls_width = ImGui::GetFrameHeight() * 2.0f + ImGui::GetStyle().ItemSpacing.x;
    if (ImGui::BeginTable("##filter-pipeline-order", 2, ImGuiTableFlags_SizingStretchProp)) {
        ImGui::TableSetupColumn("Stage", ImGuiTableColumnFlags_WidthStretch);
        ImGui::TableSetupColumn("Order", ImGuiTableColumnFlags_WidthFixed, controls_width);
        for (size_t index = 0; index < tab.filters.stages.size(); ++index) {
            PlotFilterStage &stage = tab.filters.stages[index];
            ImGui::PushID(static_cast<int>(stage.kind));
            ImGui::TableNextRow();
            ImGui::TableSetColumnIndex(0);

            bool enabled = stage.enabled;
            const bool available = filter_stage_available(tab, stage.kind);
            ImGui::BeginDisabled(!available && !stage.enabled);
            if (ImGui::Checkbox(filter_stage_name(stage.kind), &enabled)) {
                set_filter_stage_enabled(tab, stage, enabled);
            }
            ImGui::EndDisabled();

            ImGui::TableSetColumnIndex(1);
            ImGui::BeginDisabled(index == 0);
            if (ImGui::ArrowButton("##earlier", ImGuiDir_Up)) {
                move_from = index;
                move_to = index - 1;
                moved = true;
            }
            ImGui::EndDisabled();
            if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled)) {
                ImGui::SetTooltip("Move this stage earlier.");
            }
            ImGui::SameLine();

            ImGui::BeginDisabled(index + 1 >= tab.filters.stages.size());
            if (ImGui::ArrowButton("##later", ImGuiDir_Down)) {
                move_from = index;
                move_to = index + 1;
                moved = true;
            }
            ImGui::EndDisabled();
            if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled)) {
                ImGui::SetTooltip("Move this stage later.");
            }

            ImGui::PopID();
        }
        ImGui::EndTable();
    }

    if (moved) {
        std::swap(tab.filters.stages[move_from], tab.filters.stages[move_to]);
        tab.line_cache = {};
        request_filter_recompute(tab);
    }
}

void draw_main_menu_bar(AppState &app) {
    if (!ImGui::BeginMainMenuBar()) {
        return;
    }
    if (ImGui::BeginMenu("Filters")) {
        FileTab *tab = find_tab(app, app.active_tab_id);
        const bool available = tab != nullptr && supports_savitzky_golay(*tab);
        if (ImGui::BeginMenu("Savitzky-Golay", available)) {
            PlotFilterStage &stage = savitzky_golay_stage(tab->filters);
            const bool enabled = stage.enabled;
            if (ImGui::MenuItem("Enabled", nullptr, enabled)) {
                set_filter_stage_enabled(*tab, stage, !enabled);
                tab->filters.settings_open = true;
            }
            if (ImGui::MenuItem("Settings...")) {
                tab->filters.settings_open = true;
            }
            ImGui::EndMenu();
        }
        if (!available && ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled)) {
            ImGui::SetTooltip("Select a 1D or 2D numeric dataset first.");
        }

        const bool average_available = tab != nullptr && supports_windowed_average(*tab);
        if (ImGui::BeginMenu("Windowed Average", average_available)) {
            PlotFilterStage &stage = windowed_average_stage(tab->filters);
            const bool enabled = stage.enabled;
            if (ImGui::MenuItem("Enabled", nullptr, enabled)) {
                set_filter_stage_enabled(*tab, stage, !enabled);
                tab->filters.settings_open = true;
            }
            if (ImGui::MenuItem("Settings...")) {
                tab->filters.settings_open = true;
            }
            ImGui::EndMenu();
        }
        if (!average_available && ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled)) {
            ImGui::SetTooltip("Select a 1D or 2D numeric dataset first.");
        }

        const bool fft_available = tab != nullptr && supports_fft(*tab);
        if (ImGui::BeginMenu("FFT", fft_available)) {
            PlotFilterStage &stage = fft_stage(tab->filters);
            const bool enabled = stage.enabled;
            if (ImGui::MenuItem("Enabled", nullptr, enabled)) {
                set_filter_stage_enabled(*tab, stage, !enabled);
                tab->filters.settings_open = true;
            }
            if (ImGui::MenuItem("Settings...")) {
                tab->filters.settings_open = true;
            }
            ImGui::EndMenu();
        }
        if (!fft_available && ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled)) {
            ImGui::SetTooltip("Select a 1D or 2D numeric dataset first.");
        }
        ImGui::EndMenu();
    }
    ImGui::EndMainMenuBar();
}

std::string filter_window_title(const FileTab &tab) {
    return "Filters: " + tab_title(tab) + "##filters-" + std::to_string(tab.id);
}

void draw_filter_settings_window(AppState &app, FileTab &tab) {
    if (!tab.filters.settings_open) {
        return;
    }

    ImGui::SetNextWindowSize(ImVec2(430, 700), ImGuiCond_FirstUseEver);
    bool open = tab.filters.settings_open;
    const std::string title = filter_window_title(tab);
    if (ImGui::Begin(title.c_str(), &open)) {
        if (ImGui::IsWindowFocused(ImGuiFocusedFlags_RootAndChildWindows)) {
            app.active_tab_id = tab.id;
        }

        savitzky_golay_stage(tab.filters);
        windowed_average_stage(tab.filters);
        fft_stage(tab.filters);
        draw_filter_pipeline_order(tab);

        PlotFilterStage &stage = savitzky_golay_stage(tab.filters);
        const bool available = supports_savitzky_golay(tab);
        ImGui::PushID("savitzky-golay");
        ImGui::SeparatorText("Savitzky-Golay settings");

        if (!available || !tab.loaded) {
            ImGui::TextDisabled("Select a 1D or 2D numeric dataset.");
        } else {
            hdf5_plotter::SavitzkyGolayParameters &parameters = stage.savitzky_golay;
            sanitize_savitzky_golay_parameters(*tab.loaded, parameters);
            bool changed = false;
            const float input_width =
                ImGui::CalcTextSize("8888888888").x + ImGui::GetStyle().FramePadding.x * 2.0f;

            ImGui::TextUnformatted("Axis");
            ImGui::SameLine();
            int axis = parameters.axis == hdf5_plotter::FilterAxis::X ? 0 : 1;
            const bool x_available =
                savitzky_golay_axis_length(*tab.loaded, hdf5_plotter::FilterAxis::X) >= 3;
            const bool y_available = tab.loaded->kind == LoadedKind::Heatmap2D &&
                                     savitzky_golay_axis_length(*tab.loaded,
                                                               hdf5_plotter::FilterAxis::Y) >= 3;
            ImGui::BeginDisabled(!x_available);
            if (ImGui::RadioButton("X", &axis, 0)) {
                changed = true;
            }
            ImGui::EndDisabled();
            ImGui::SameLine();
            ImGui::BeginDisabled(!y_available);
            if (ImGui::RadioButton("Y", &axis, 1)) {
                changed = true;
            }
            ImGui::EndDisabled();
            parameters.axis = axis == 0 ? hdf5_plotter::FilterAxis::X : hdf5_plotter::FilterAxis::Y;

            ImGui::SetNextItemWidth(input_width);
            changed |= ImGui::InputInt("Window size", &parameters.window_size, 2, 10);
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip("Odd values from 3 to 501, limited by the selected axis length.");
            }
            ImGui::SetNextItemWidth(input_width);
            changed |= ImGui::InputInt("Polynomial order", &parameters.polynomial_order, 1, 1);
            ImGui::SetNextItemWidth(input_width);
            changed |= ImGui::InputInt("Derivative order", &parameters.derivative_order, 1, 1);
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip("Derivatives are calculated in sample-index units.");
            }

            if (sanitize_savitzky_golay_parameters(*tab.loaded, parameters)) {
                changed = true;
            }
            if (changed && stage.enabled) {
                request_filter_recompute(tab);
            }
        }

        ImGui::PopID();

        PlotFilterStage &average = windowed_average_stage(tab.filters);
        const bool average_available = supports_windowed_average(tab);
        ImGui::PushID("windowed-average");
        ImGui::SeparatorText("Windowed Average settings");

        if (!average_available || !tab.loaded) {
            ImGui::TextDisabled("Select a 1D or 2D numeric dataset.");
        } else {
            hdf5_plotter::WindowedAverageParameters &parameters = average.windowed_average;
            sanitize_windowed_average_parameters(*tab.loaded, parameters);
            bool changed = false;
            const float input_width =
                ImGui::CalcTextSize("8888888888").x + ImGui::GetStyle().FramePadding.x * 2.0f;

            ImGui::TextUnformatted("Axis");
            ImGui::SameLine();
            int axis = parameters.axis == hdf5_plotter::FilterAxis::X ? 0 : 1;
            const bool x_available =
                windowed_average_axis_length(*tab.loaded, hdf5_plotter::FilterAxis::X) >= 3;
            const bool y_available =
                tab.loaded->kind == LoadedKind::Heatmap2D &&
                windowed_average_axis_length(*tab.loaded, hdf5_plotter::FilterAxis::Y) >= 3;
            ImGui::BeginDisabled(!x_available);
            changed |= ImGui::RadioButton("X", &axis, 0);
            ImGui::EndDisabled();
            ImGui::SameLine();
            ImGui::BeginDisabled(!y_available);
            changed |= ImGui::RadioButton("Y", &axis, 1);
            ImGui::EndDisabled();
            parameters.axis = axis == 0 ? hdf5_plotter::FilterAxis::X : hdf5_plotter::FilterAxis::Y;

            ImGui::SetNextItemWidth(input_width);
            changed |= ImGui::InputInt("Window size", &parameters.window_size, 2, 10);
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip(
                    "Centered odd window, limited by the selected axis. Edges use available samples.");
            }

            if (sanitize_windowed_average_parameters(*tab.loaded, parameters)) {
                changed = true;
            }
            if (changed && average.enabled) {
                request_filter_recompute(tab);
            }
        }
        ImGui::PopID();

        PlotFilterStage &fft = fft_stage(tab.filters);
        const bool fft_available = supports_fft(tab);
        ImGui::PushID("fft");
        ImGui::SeparatorText("FFT settings");

        if (!fft_available || !tab.loaded) {
            ImGui::TextDisabled("Select a 1D or 2D numeric dataset.");
        } else {
            hdf5_plotter::FftParameters &parameters = fft.fft;
            sanitize_fft_parameters(*tab.loaded, parameters);
            bool changed = false;

            ImGui::TextUnformatted("Axis");
            ImGui::SameLine();
            int axis = parameters.axis == hdf5_plotter::FilterAxis::X ? 0 : 1;
            const auto [width, height] = plot_value_dimensions(*tab.loaded);
            ImGui::BeginDisabled(width < 2);
            changed |= ImGui::RadioButton("X", &axis, 0);
            ImGui::EndDisabled();
            ImGui::SameLine();
            ImGui::BeginDisabled(tab.loaded->kind == LoadedKind::Line1D || height < 2);
            changed |= ImGui::RadioButton("Y", &axis, 1);
            ImGui::EndDisabled();
            parameters.axis = axis == 0 ? hdf5_plotter::FilterAxis::X : hdf5_plotter::FilterAxis::Y;

            static const char *value_modes[] = {"Magnitude", "Power", "Phase", "Real", "Imaginary"};
            int value_mode = static_cast<int>(parameters.value_mode);
            ImGui::SetNextItemWidth(150.0f);
            if (ImGui::Combo("Output", &value_mode, value_modes, IM_ARRAYSIZE(value_modes))) {
                parameters.value_mode = static_cast<hdf5_plotter::FftValueMode>(value_mode);
                changed = true;
            }

            static const char *windows[] = {"Rectangular", "Hann", "Hamming", "Blackman"};
            int window = static_cast<int>(parameters.window);
            ImGui::SetNextItemWidth(150.0f);
            if (ImGui::Combo("Window", &window, windows, IM_ARRAYSIZE(windows))) {
                parameters.window = static_cast<hdf5_plotter::FftWindow>(window);
                changed = true;
            }
            changed |= ImGui::Checkbox("Remove mean", &parameters.remove_mean);
            const bool supports_decibels = parameters.value_mode == hdf5_plotter::FftValueMode::Magnitude ||
                                           parameters.value_mode == hdf5_plotter::FftValueMode::Power;
            ImGui::BeginDisabled(!supports_decibels);
            changed |= ImGui::Checkbox("Decibels", &parameters.decibels);
            ImGui::EndDisabled();
            if (!supports_decibels && parameters.decibels) {
                parameters.decibels = false;
                changed = true;
            }
            const AxisSampling automatic_sampling = fft_axis_sampling(*tab.loaded, parameters.axis);
            const bool was_automatic_spacing = parameters.automatic_spacing;
            if (ImGui::Checkbox("Auto sample spacing", &parameters.automatic_spacing)) {
                if (was_automatic_spacing && !parameters.automatic_spacing) {
                    parameters.sample_spacing = automatic_sampling.spacing;
                }
                changed = true;
            }
            double spacing = parameters.automatic_spacing ? automatic_sampling.spacing : parameters.sample_spacing;
            ImGui::BeginDisabled(parameters.automatic_spacing);
            ImGui::SetNextItemWidth(150.0f);
            if (ImGui::InputDouble("Sample spacing", &spacing, 0.0, 0.0, "%.8g")) {
                if (std::isfinite(spacing) && spacing > 0.0) {
                    parameters.sample_spacing = spacing;
                    changed = true;
                }
            }
            ImGui::EndDisabled();
            if (changed && fft.enabled) {
                request_filter_recompute(tab);
            }
        }
        ImGui::PopID();
        if (tab.filters.processing) {
            ImGui::TextDisabled("Updating...");
        }
        if (!tab.filters.error.empty()) {
            ImGui::TextColored(ImVec4(1.0f, 0.35f, 0.25f, 1.0f), "%s", tab.filters.error.c_str());
        }
    }
    ImGui::End();
    tab.filters.settings_open = open;
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
    if (tab.filters.processing && tab.filters.future.valid()) {
        if (tab.filters.cancel_requested) {
            tab.filters.cancel_requested->store(true, std::memory_order_relaxed);
        }
        tab.filters.future.wait();
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
            range.min = displayed_x_min(tab, data);
            range.max = displayed_x_max(tab, data);
        } else {
            const auto [display_min, display_max] = displayed_value_range(tab, data);
            range.min = data.kind == LoadedKind::Line1D ? display_min : displayed_y_min(tab, data);
            range.max = data.kind == LoadedKind::Line1D ? display_max : displayed_y_max(tab, data);
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

CommentPreviewResult comment_preview_worker(int token, std::string path) {
    CommentPreviewResult result;
    result.token = token;
    result.path = std::move(path);
    try {
        result.caption = trim(load_file_comment_preview(result.path));
    } catch (const std::exception &ex) {
        result.error = ex.what();
    }
    return result;
}

CommentPreviewCacheEntry *find_comment_preview(FilePicker &picker, const std::string &path) {
    auto found = std::find_if(picker.preview_cache.begin(), picker.preview_cache.end(), [&](const CommentPreviewCacheEntry &entry) {
        return entry.path == path;
    });
    return found == picker.preview_cache.end() ? nullptr : &(*found);
}

const CommentPreviewCacheEntry *find_comment_preview(const FilePicker &picker, const std::string &path) {
    auto found = std::find_if(picker.preview_cache.begin(), picker.preview_cache.end(), [&](const CommentPreviewCacheEntry &entry) {
        return entry.path == path;
    });
    return found == picker.preview_cache.end() ? nullptr : &(*found);
}

void start_comment_preview(FilePicker &picker, const std::string &path) {
    const std::string ext = std::filesystem::path(path).extension().string();
    if (path.empty() || (ext != ".h5" && ext != ".hdf5" && ext != ".H5" && ext != ".HDF5")) {
        return;
    }
    if (find_comment_preview(picker, path) != nullptr) {
        return;
    }
    if (path == picker.preview_path || path == picker.pending_preview_path) {
        return;
    }
    if (picker.preview_loading) {
        picker.pending_preview_path = path;
        return;
    }

    picker.preview_path = path;
    picker.preview_caption.clear();
    picker.preview_error.clear();
    picker.preview_loading = true;
    const int token = ++picker.preview_token;
    picker.preview_future = std::async(std::launch::async, comment_preview_worker, token, path);
}

void poll_comment_preview(FilePicker &picker) {
    if (!picker.preview_loading || !picker.preview_future.valid()) {
        if (!picker.pending_preview_path.empty()) {
            const std::string pending = std::move(picker.pending_preview_path);
            picker.pending_preview_path.clear();
            start_comment_preview(picker, pending);
        }
        return;
    }

    using namespace std::chrono_literals;
    if (picker.preview_future.wait_for(0ms) != std::future_status::ready) {
        return;
    }

    CommentPreviewResult result = picker.preview_future.get();
    picker.preview_loading = false;
    if (result.token == picker.preview_token) {
        picker.preview_path = std::move(result.path);
        picker.preview_caption = std::move(result.caption);
        picker.preview_error = std::move(result.error);
        CommentPreviewCacheEntry *entry = find_comment_preview(picker, picker.preview_path);
        if (entry == nullptr) {
            picker.preview_cache.push_back(CommentPreviewCacheEntry{picker.preview_path, picker.preview_caption, picker.preview_error});
            if (picker.preview_cache.size() > 64) {
                picker.preview_cache.erase(picker.preview_cache.begin());
            }
        } else {
            entry->caption = picker.preview_caption;
            entry->error = picker.preview_error;
        }
    }
    if (!picker.pending_preview_path.empty() && picker.pending_preview_path != picker.preview_path) {
        const std::string pending = std::move(picker.pending_preview_path);
        picker.pending_preview_path.clear();
        start_comment_preview(picker, pending);
    } else {
        picker.pending_preview_path.clear();
    }
}

void draw_comment_preview(const FilePicker &picker, const std::string &path, bool compact) {
    if (path.empty()) {
        return;
    }

    const CommentPreviewCacheEntry *entry = find_comment_preview(picker, path);
    if (entry != nullptr) {
        if (!entry->caption.empty()) {
            if (compact) {
                ImGui::TextWrapped("%s", entry->caption.c_str());
            } else {
                ImGui::TextWrapped("Comment: %s", entry->caption.c_str());
            }
        } else if (!entry->error.empty()) {
            ImGui::TextDisabled("Comment preview unavailable: %s", entry->error.c_str());
        } else {
            ImGui::TextDisabled("No comment field found.");
        }
        return;
    }

    if (picker.preview_loading && (picker.preview_path == path || picker.pending_preview_path == path)) {
        ImGui::TextDisabled("Comment preview loading...");
        return;
    }
    if (picker.preview_path != path) {
        return;
    }
    if (!picker.preview_caption.empty()) {
        if (compact) {
            ImGui::TextWrapped("%s", picker.preview_caption.c_str());
        } else {
            ImGui::TextWrapped("Comment: %s", picker.preview_caption.c_str());
        }
    } else if (!picker.preview_error.empty()) {
        ImGui::TextDisabled("Comment preview unavailable: %s", picker.preview_error.c_str());
    } else {
        ImGui::TextDisabled("No comment field found.");
    }
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
    poll_comment_preview(picker);

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
        const std::string file_path = file.path().string();
        if (ImGui::Selectable(label.c_str(), trim(picker.selected_path.data()) == file.path().string(),
                              ImGuiSelectableFlags_AllowDoubleClick)) {
            copy_to_buffer(picker.selected_path, file_path);
            start_comment_preview(picker, file_path);
        }
        if (ImGui::IsItemHovered()) {
            start_comment_preview(picker, file_path);
            ImGui::BeginTooltip();
            ImGui::TextUnformatted(file_path.c_str());
            ImGui::Separator();
            draw_comment_preview(picker, file_path, true);
            ImGui::EndTooltip();
            if (ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left)) {
                open_path_as_tab(app, file_path);
                picker.visible = false;
                ImGui::CloseCurrentPopup();
            }
        }
    }
    ImGui::EndChild();

    const std::string selected_path = trim(picker.selected_path.data());
    if (!selected_path.empty()) {
        start_comment_preview(picker, selected_path);
        ImGui::SeparatorText("Selected File Comment");
        draw_comment_preview(picker, selected_path, false);
    }

    if (ImGui::Button("Open Selected")) {
        if (!selected_path.empty()) {
            open_path_as_tab(app, selected_path);
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
    ImGui::Text("Window drawable/focused: %s / %s", app.last_window_drawable ? "yes" : "no",
                app.last_window_focused ? "yes" : "no");
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
    draw_main_menu_bar(app);
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
        draw_slice_windows(app, tab);
        draw_file_details_window(app, tab);
        draw_filter_settings_window(app, tab);
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
    static const std::array<const char *, 9> candidates = {
        "C:\\Windows\\Fonts\\segoeui.ttf",
        "C:\\Windows\\Fonts\\arial.ttf",
        "C:\\Windows\\Fonts\\verdana.ttf",
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
    return app.picker.preview_loading ||
           std::any_of(app.tabs.begin(), app.tabs.end(), [](const std::unique_ptr<FileTab> &tab) {
               return tab->loading || tab->swmr_polling || tab->filters.processing;
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

bool window_is_drawable(Uint32 flags) {
    return (flags & SDL_WINDOW_SHOWN) != 0 &&
           (flags & (SDL_WINDOW_HIDDEN | SDL_WINDOW_MINIMIZED)) == 0;
}

bool window_has_input_focus(Uint32 flags) {
    return (flags & SDL_WINDOW_INPUT_FOCUS) != 0;
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
    auto next_unfocused_frame = std::chrono::steady_clock::now();
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
        const Uint32 initial_window_flags = SDL_GetWindowFlags(window);
        const bool initially_drawable = window_is_drawable(initial_window_flags);
        const bool initially_focused = window_has_input_focus(initial_window_flags);
        const bool reduced_visibility = !initially_drawable || !initially_focused;

        if (reduced_visibility || (!force_redraw && !app_has_background_work(app))) {
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
            state_changed = poll_filter_pipeline(*tab) || state_changed;
        }

        bool background_work = app_has_background_work(app);
        const Uint32 window_flags = SDL_GetWindowFlags(window);
        const bool window_drawable = window_is_drawable(window_flags);
        const bool window_focused = window_has_input_focus(window_flags);
        app.last_window_drawable = window_drawable;
        app.last_window_focused = window_focused;

        if (!window_drawable) {
            force_redraw = true;
            app.last_event_count = event_count;
            app.last_used_idle_wait = used_idle_wait;
            app.last_fast_frame = false;
            app.last_background_work = background_work;
            app.last_ui_active = false;
            continue;
        }

        const auto before_render = std::chrono::steady_clock::now();
        if (!window_focused && before_render < next_unfocused_frame) {
            force_redraw = force_redraw || state_changed || background_work;
            app.last_event_count = event_count;
            app.last_used_idle_wait = used_idle_wait;
            app.last_fast_frame = false;
            app.last_background_work = background_work;
            app.last_ui_active = false;
            continue;
        }

        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplSDL2_NewFrame();
        ImGui::NewFrame();

        layout_windows(app);
        state_changed = handle_keyboard_shortcuts(app) || state_changed;
        background_work = app_has_background_work(app);

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

        if (window_focused) {
            next_unfocused_frame = before_render;
        } else {
            next_unfocused_frame = before_render + kIdleFrameInterval;
        }
        const bool fast_frames = window_focused && (state_changed || background_work || ui_fast_frames);
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
        if (tab->filters.processing && tab->filters.future.valid()) {
            if (tab->filters.cancel_requested) {
                tab->filters.cancel_requested->store(true, std::memory_order_relaxed);
            }
            tab->filters.future.wait();
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

#include "llama-ssd-rows.h"

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <limits>
#include <list>
#include <mutex>
#include <stdexcept>
#include <sys/stat.h>
#include <thread>
#include <unistd.h>
#include <unordered_map>
#include <unordered_set>
#include <vector>

struct llama_ssd_rows::impl {
    static constexpr size_t page_size = 16384;
    struct page { uint64_t offset; std::vector<uint8_t> data; };
    int fd = -1;
    uint64_t offset, end, file_size;
    int64_t columns, rows;
    size_t row_bytes, capacity;
    ggml_to_float_t decode;
    mutable std::mutex mutex;
    std::list<page> lru;
    std::unordered_map<uint64_t, std::list<page>::iterator> pages;
    uint64_t reads = 0;

    ~impl() { if (fd >= 0) close(fd); }

    // Reuse the evicted allocation to keep cache memory bounded even during misses.
    std::list<page>::iterator take(uint64_t aligned) {
        if (lru.size() == capacity) {
            pages.erase(lru.back().offset);
            lru.splice(lru.begin(), lru, std::prev(lru.end()));
        } else {
            lru.push_front({0, std::vector<uint8_t>(page_size)});
        }
        lru.front().offset = aligned;
        return lru.begin();
    }

    bool fill(page & p) const {
        const size_t size = std::min<uint64_t>(page_size, file_size - p.offset);
        size_t done = 0;
        while (done < size) {
            ssize_t n = pread(fd, p.data.data() + done, size - done, p.offset + done);
            if (n < 0 && errno == EINTR) continue;
            if (n <= 0) return false;
            done += size_t(n);
        }
        return true;
    }

    const page & get(uint64_t pos) {
        const uint64_t aligned = pos / page_size * page_size;
        auto found = pages.find(aligned);
        if (found != pages.end()) {
            lru.splice(lru.begin(), lru, found->second);
            return *lru.begin();
        }
        auto it = take(aligned);
        if (!fill(*it)) {
            lru.erase(it);
            throw std::runtime_error("PLE SSD read failed or file was truncated");
        }
        reads += std::min<uint64_t>(page_size, file_size - aligned);
        pages.emplace(aligned, it);
        return *it;
    }

    // Uncached random reads reach SSD throughput only with many requests in flight.
    static constexpr size_t max_readers = 16;

    // Caches every distinct page in need (at most capacity), reading the misses concurrently.
    void load(const std::vector<uint64_t> & need) {
        for (uint64_t aligned : need) {
            auto found = pages.find(aligned);
            if (found != pages.end()) lru.splice(lru.begin(), lru, found->second);
        }
        // Needed pages now lead the list, so eviction below only takes unneeded ones.
        std::vector<std::list<page>::iterator> missing;
        for (uint64_t aligned : need) {
            if (!pages.count(aligned)) missing.push_back(take(aligned));
        }
        std::atomic<size_t> next{0};
        std::atomic<bool> failed{false};
        auto work = [&]() {
            for (size_t i = next++; !failed && i < missing.size(); i = next++) {
                if (!fill(*missing[i])) failed = true;
            }
        };
        std::vector<std::thread> threads;
        try {
            for (size_t i = 1; i < std::min(missing.size(), max_readers); ++i) threads.emplace_back(work);
        } catch (...) {
            failed = true;
        }
        work();
        for (auto & t : threads) t.join();
        if (failed) {
            for (auto it : missing) lru.erase(it);
            throw std::runtime_error("PLE SSD read failed or file was truncated");
        }
        for (auto it : missing) {
            reads += std::min<uint64_t>(page_size, file_size - it->offset);
            pages.emplace(it->offset, it);
        }
    }
};

llama_ssd_rows::llama_ssd_rows(int fd, uint64_t offset, ggml_type type, int64_t columns,
                             int64_t rows, size_t cache_bytes) : pimpl(new impl) {
    auto & p = *pimpl;
    if (columns <= 0 || rows <= 0 || cache_bytes < impl::page_size ||
        type < 0 || type >= GGML_TYPE_COUNT || columns % ggml_blck_size(type)) {
        throw std::invalid_argument("invalid PLE SSD row geometry or cache budget");
    }
    p.row_bytes = ggml_row_size(type, columns);
    if (uint64_t(rows) > (std::numeric_limits<uint64_t>::max() - offset) / p.row_bytes) {
        throw std::invalid_argument("PLE SSD tensor extent overflows");
    }
    struct stat st{};
    if (fstat(fd, &st) != 0 || st.st_size < 0 || offset + uint64_t(rows)*p.row_bytes > uint64_t(st.st_size)) {
        throw std::invalid_argument("PLE SSD tensor exceeds source file");
    }
    p.decode = ggml_get_type_traits(type)->to_float;
    if (!p.decode && type != GGML_TYPE_F32) throw std::invalid_argument("unsupported PLE SSD type");
    p.offset = offset; p.end = offset + uint64_t(rows)*p.row_bytes; p.file_size = st.st_size;
    p.columns = columns; p.rows = rows; p.capacity = cache_bytes / impl::page_size;
    p.fd = dup(fd);
    if (p.fd < 0) throw std::runtime_error("cannot duplicate PLE SSD descriptor");
#ifdef __APPLE__
    if (fcntl(p.fd, F_NOCACHE, 1) != 0 || fcntl(p.fd, F_RDAHEAD, 0) != 0) {
        throw std::runtime_error("cannot disable filesystem caching for PLE SSD reads");
    }
#endif
}

llama_ssd_rows::~llama_ssd_rows() = default;

void llama_ssd_rows::read_rows(const int32_t * ids, size_t count, float * output) {
    auto & p = *pimpl;
    if (count && (!ids || !output)) throw std::invalid_argument("null PLE SSD input/output");
    for (size_t i = 0; i < count; ++i) {
        if (ids[i] < 0 || ids[i] >= p.rows) throw std::out_of_range("PLE SSD row index");
    }
    std::lock_guard<std::mutex> lock(p.mutex);
    std::vector<uint8_t> row(p.row_bytes);
    std::vector<uint64_t> need;
    std::unordered_set<uint64_t> seen;
    for (size_t first = 0; first < count;) {
        // Load as many rows' pages as the cache holds in one concurrent pass. A row larger
        // than the whole cache is left to the sequential reads below.
        need.clear();
        seen.clear();
        size_t last = first;
        for (; last < count; ++last) {
            const uint64_t pos = p.offset + uint64_t(ids[last])*p.row_bytes;
            const uint64_t lo = pos / impl::page_size, hi = (pos + p.row_bytes - 1) / impl::page_size;
            size_t added = 0;
            for (uint64_t i = lo; i <= hi; ++i) added += !seen.count(i*impl::page_size);
            if (need.size() + added > p.capacity) break;
            for (uint64_t i = lo; i <= hi; ++i) {
                if (seen.insert(i*impl::page_size).second) need.push_back(i*impl::page_size);
            }
        }
        if (last == first) {
            ++last;
        } else {
            p.load(need);
        }
        for (size_t i = first; i < last; ++i) {
            const uint64_t pos = p.offset + uint64_t(ids[i])*p.row_bytes;
            size_t done = 0;
            while (done < p.row_bytes) {
                const auto & page = p.get(pos + done);
                const size_t within = pos + done - page.offset;
                const size_t n = std::min(p.row_bytes - done, impl::page_size - within);
                memcpy(row.data() + done, page.data.data() + within, n);
                done += n;
            }
            if (p.decode) p.decode(row.data(), output + i*p.columns, p.columns);
            else memcpy(output + i*p.columns, row.data(), p.row_bytes);
        }
        first = last;
    }
}
size_t llama_ssd_rows::cached_bytes() const {
    std::lock_guard<std::mutex> lock(pimpl->mutex);
    return pimpl->lru.size()*impl::page_size;
}
uint64_t llama_ssd_rows::bytes_read() const {
    std::lock_guard<std::mutex> lock(pimpl->mutex);
    return pimpl->reads;
}

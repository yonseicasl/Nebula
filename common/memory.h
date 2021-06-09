#ifndef __NPU_MEMORY_H__
#define __NPU_MEMORY_H__

#include <cstdlib>
#include <cstdint>
#include <set>

class npu_segment {

public:
    //npu_segment(uint64_t m_ptr, uint64_t m_addr, size_t m_size);
    npu_segment(size_t m_size, uint64_t m_addr, uint64_t m_ptr = 0);
    ~npu_segment();

    bool operator < (const npu_segment &s) const;

    size_t size;
    uint64_t addr;
    uint64_t ptr;
};

class npu_mmu {
public:

    npu_mmu();
    ~npu_mmu();

    static void init(size_t m_capacity);
    static void npu_malloc(uint64_t m_ptr, size_t m_size);

    // Virtual to physical address translation.
    static uint64_t v2p(uint64_t m_ptr);            

private:

    static std::set<npu_segment> free_list;         // Free segment list
    static std::set<npu_segment> used_list;         // Used segment list

};

#endif

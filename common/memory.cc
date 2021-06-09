#include <iostream>


#include "memory.h"

#define CAPACITY 0x200000000
#define NPU_MEM_OFFSET 0x8000000


std::set<npu_segment> npu_mmu::free_list;
std::set<npu_segment> npu_mmu::used_list;

npu_segment::npu_segment(size_t m_size, uint64_t m_addr, uint64_t m_ptr) :
    size(m_size), 
    addr(m_addr),
    ptr(m_ptr) {

}

npu_segment::~npu_segment() {

}

bool npu_segment::operator < (const npu_segment &s) const {
    if(size == s.size) return addr < s.addr;
    return size < s.size;
}

npu_mmu::npu_mmu() {
    init(CAPACITY);
}

npu_mmu::~npu_mmu() {
    free_list.clear();
    used_list.clear();
}

void npu_mmu::init(size_t m_capacity) {
    free_list.emplace(m_capacity - NPU_MEM_OFFSET, NPU_MEM_OFFSET);
}


void npu_mmu::npu_malloc(uint64_t m_ptr, size_t m_size) {
    for(std::set<npu_segment>::iterator it = free_list.begin(); it != free_list.end(); ++it) {
        if(m_size <= it->size) {
            // Add the memory block to used list.
            used_list.emplace(m_size, it->addr, m_ptr);
            // Divide the memory block at free list.
            free_list.emplace(it->size - m_size, it->addr + m_size);
            // Remove the memory block from free list.
            free_list.erase(it);
        }
    }
    std::cerr << "Error: NPU malloc failed" << std::endl; exit(1);
}


void npu_mmu::npu_free(uint64_t m_ptr) {
    for(std::set<npu_segment>::iterator it = used_list.begin(); it != used_list.end(); ++it) {
        if(m_ptr == it->ptr) {
            // Add the memory block to free list.
            std::set<npu_segment>::iterator free_it = free_list.emplace(it->size, it->addr).first;
            // Remove the memory block from used list.
            used_list.erase(it);

            // Coalescing adjacent memory blocks.
            if(free_it != free_list.begin()) {
                //uint64_t addr = 
            }
        }
    }
}

uint64_t npu_mmu::v2p(uint64_t m_ptr) {
    for(std::set<npu_segment>::iterator it =used_list.begin(); it != used_list.end(); ++it) {
        if((m_ptr >= it->ptr) && (m_ptr < it->ptr+it->size)) {
            return (uint64_t)(it->addr + m_ptr - it->ptr);
        }
    }
    return 0;
}

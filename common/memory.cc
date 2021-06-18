#include <iostream>

#include "memory.h"

#define BASE_ADDR 0X80000000
#define CAPACITY 0x200000000

npu_mmu *npu_mmu::address=0;
npu_segment base_addr;

npu_segment::npu_segment() :
    addr(BASE_ADDR),
    ptr(0),
    valid(false) {

}

npu_segment::~npu_segment() {

}

npu_mmu::npu_mmu() {
    init(CAPACITY);
}

npu_mmu::~npu_mmu() {
    //free_list.clear();
    //used_list.clear();
}

void npu_mmu::init(size_t m_capacity) {
    //free_list.emplace(m_capacity - NPU_MEM_OFFSET, NPU_MEM_OFFSET);
}


void npu_mmu::npu_malloc(uint64_t m_ptr) {
    if(base_addr.valid) {
        if(m_ptr < base_addr.ptr) {
            base_addr.ptr = m_ptr;
        }
    }
    else {
        base_addr.ptr = m_ptr, base_addr.addr = BASE_ADDR, base_addr.valid = true;
    }
}


void npu_mmu::npu_free(uint64_t m_ptr) {
}

uint64_t npu_mmu::v2p(uint64_t m_ptr) {
    
    uint64_t offset;

    offset = m_ptr - base_addr.ptr;

    if(m_ptr >= base_addr.ptr && base_addr.addr + offset < CAPACITY) {
        std::cout << "Virtual address : 0x" << std::hex << m_ptr << std::endl;
        std::cout << "Physical address : 0x" << std::hex << base_addr.addr + offset << std::endl;
        return base_addr.addr + offset;
    }
    else if(m_ptr < base_addr.ptr) {
        std::cout << "Virtual address : 0x" << std::hex << m_ptr << " is bigger than the baseline virtual address : 0x" << std::hex << base_addr.ptr << std::endl;
    }
    else if(base_addr.addr + offset > CAPACITY) {
        std::cout << "Physical address : 0x" << std::hex << base_addr.addr+offset << " is bigger than the capacity 0x" << CAPACITY << std::endl;
    }
    return 0;
    

    return 0;
}

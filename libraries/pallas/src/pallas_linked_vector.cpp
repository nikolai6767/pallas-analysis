/*
 * Copyright (C) Telecom SudParis
 * See LICENSE in top-level directory.
 */

#include "pallas/pallas_linked_vector.h"
#include <cstring>
using namespace pallas;

LinkedVector::LinkedVector() {
  first = new SubVector(defaultSize);
  last = first;
}

uint64_t* LinkedVector::add(uint64_t val) {
  if (this->last->size >= this->last->allocated) {
    pallas_log(DebugLevel::Debug, "Adding a new tail to an array: %p\n", this);
    last = new SubVector(defaultSize, last);
  }
  size++;
  max = std::max(max, val);
  min = std::min(min, val);
  mean = ((size - 1) * mean + val) / size;
  return last->add(val);
}

uint64_t& LinkedVector::at(size_t pos) {
  if (pos >= size) {
    pallas_error("Getting an element whose index (%lu) is bigger than vector size (%lu)\n", pos, size);
  }
  if (first == nullptr) {
    load_timestamps();
  }
  struct SubVector* correct_sub = last;
  while (pos < correct_sub->starting_index) {
    correct_sub = correct_sub->previous;
  }
  return correct_sub->at(pos);
}

uint64_t& LinkedVector::operator[](size_t pos) {
  if (first == nullptr) {
    load_timestamps();
  }
  struct SubVector* correct_sub = last;
  while (pos < correct_sub->starting_index) {
    correct_sub = correct_sub->previous;
  }
  return (*correct_sub)[pos];
}

uint64_t& LinkedVector::front() const {
  return first->at(0);
}

uint64_t& LinkedVector::back() const {
  return last->at(size - 1);
}

void LinkedVector::print() const {
  std::cout << "[";
  if (size) {
    for (auto& i : *this) {
      std::cout << i << ((&i != &this->back()) ? ", " : "]");
    }
  } else
    std::cout << "]";
}

/* C++ Callbacks for C Usage */
LinkedVector* linked_vector_new() {
  return new LinkedVector();
}
uint64_t* linked_vector_add(LinkedVector* linkedVector, uint64_t val) {
  return linkedVector->add(val);
}
uint64_t* linked_vector_get(LinkedVector* linkedVector, size_t pos) {
  return &linkedVector->at(pos);
}
uint64_t* linked_vector_get_last(LinkedVector* linkedVector) {
  return &linkedVector->back();
}
void print(LinkedVector linkedVector) {
  return linkedVector.print();
}

/*
 * Copyright (C) Telecom SudParis
 * See LICENSE in top-level directory.
 */

#include <cinttypes>
#include <ranges>
#include <set>
#include <sstream>

#include "pallas/pallas.h"
#include "pallas/pallas_log.h"
#include "pallas/pallas_archive.h"

namespace pallas {
void Thread::loadTimestamps() {
  DOFOR(i, nb_events) {
    size_t loaded_duration = events[i].durations->front();
  }
  DOFOR(i, nb_sequences) {
    size_t loaded_duration = sequences[i]->durations->front();
  }
}

Event* Thread::getEvent(Token token) const {
  return &getEventSummary(token)->event;
}

EventSummary* Thread::getEventSummary(Token token) const {
  if (token.type != TokenType::TypeEvent) {
    pallas_error("Trying to getEventSummary of (%c%d)\n", PALLAS_TOKEN_TYPE_C(token), token.id);
  }
  pallas_assert(token.id < this->nb_events);
  return &this->events[token.id];
}

Sequence* Thread::getSequence(Token token) const {
  if (token.type != TokenType::TypeSequence) {
    pallas_error("Trying to getSequence of (%c%d)\n", PALLAS_TOKEN_TYPE_C(token), token.id);
  }
  pallas_assert(token.id < this->nb_sequences);
  return this->sequences[token.id];
}

Loop* Thread::getLoop(Token token) const {
  if (token.type != TokenType::TypeLoop) {
    pallas_error("Trying to getLoop of (%c%d)\n", PALLAS_TOKEN_TYPE_C(token), token.id);
  }
  pallas_assert(token.id < this->nb_loops);
  return &this->loops[token.id];
}

Token& Thread::getToken(Token sequenceToken, int index) const {
  if (sequenceToken.type == TypeSequence) {
    auto sequence = getSequence(sequenceToken);
    if (!sequence) {
      pallas_error("Invalid sequence ID: %d\n", sequenceToken.id);
    }
    if (index >= sequence->size()) {
      pallas_error("Invalid index (%d) in sequence %d\n", index, sequenceToken.id);
    }
    return sequence->tokens[index];
  } else if (sequenceToken.type == TypeLoop) {
    auto loop = getLoop(sequenceToken);
    if (!loop) {
      pallas_error("Invalid loop ID: %d\n", sequenceToken.id);
    }
    return loop->repeated_token;
  }
  pallas_error("Invalid parameter to getToken\n");
}

std::string Thread::getTokenString(Token token) const {
  std::ostringstream tempString;
  switch (token.type) {
  case TypeInvalid:
    tempString << "U";
    break;
  case TypeEvent:
    tempString << "E";
    break;
  case TypeSequence:
    tempString << "S";
    break;
  case TypeLoop:
    tempString << "L";
    break;
  }
  tempString << token.id;
  if (token.type == TypeEvent) {
    Event* event = getEvent(token);
    tempString << ((event->record) == PALLAS_EVENT_ENTER ? "E" : (event->record) == PALLAS_EVENT_LEAVE ? "L" : "S");
  }
  return tempString.str();
}

pallas_duration_t Thread::getDuration() const {
  return sequences[0]->durations->at(0);
}
pallas_duration_t get_duration(PALLAS(Thread) *t) { return t->getDuration(); }

pallas_timestamp_t Thread::getFirstTimestamp() const {
  return 0; 			// TODO: find the first timestamp
}
pallas_timestamp_t get_first_timestamp(PALLAS(Thread) *t) { return t->getFirstTimestamp(); }

pallas_timestamp_t Thread::getLastTimestamp() const {
  return getFirstTimestamp() + getDuration();
}
pallas_timestamp_t get_last_timestamp(PALLAS(Thread) *t) { return t->getLastTimestamp(); }

size_t Thread::getEventCount() const {
  size_t ret = 0;
  for(unsigned i=0; i<this->nb_events; i++) {
    ret += this->events[i].nb_occurences;
  }
  return ret;
}
size_t get_event_count(PALLAS(Thread) *t) { return t->getEventCount(); }

void Thread::printToken(Token token) const {
  std::cout << getTokenString(token);
}

void Thread::printTokenArray(const Token* array, size_t start_index, size_t len) const {
  printf("[");
  for (int i = 0; i < len; i++) {
    printToken(array[start_index + i]);
    printf(" ");
  }
  printf("]\n");
}

void Thread::printTokenVector(const std::vector<Token>& vector) const {
  printf("[");
  for (auto& token : vector) {
    printToken(token);
    printf(" ");
  }
  printf("]\n");
}

void Thread::printSequence(pallas::Token token) const {
  Sequence* sequence = getSequence(token);
  printf("#Sequence %d (%zu tokens)-------------\n", token.id, sequence->tokens.size());
  printTokenVector(sequence->tokens);
}

void Thread::printEvent(pallas::Event* e) const {
  char output_str[1024];
  size_t buffer_size = 1024;
  printEventToString(e, output_str, buffer_size);
  std::cout << output_str;
}

static inline void pop_data(Event* e, void* data, size_t data_size, byte*& cursor) {
  if (cursor == nullptr) {
    /* initialize the cursor to the begining of event data */
    cursor = &e->event_data[0];
  }

  uintptr_t last_event_byte = ((uintptr_t)e) + e->event_size;
  uintptr_t last_read_byte = ((uintptr_t)cursor) + data_size;
  pallas_assert(last_read_byte <= last_event_byte);

  memcpy(data, cursor, data_size);
  cursor += data_size;
}

const char* Thread::getRegionStringFromEvent(pallas::Event* e) const {
  const Region* region = NULL;
  byte* cursor = nullptr;
  switch (e->record)
    {
    case PALLAS_EVENT_ENTER:
      {
	RegionRef region_ref;
	pop_data(e, &region_ref, sizeof(region_ref), cursor);
	region = archive->global_archive->getRegion(region_ref);
	break;
      }
    case PALLAS_EVENT_LEAVE:
      {
	RegionRef region_ref;
	pop_data(e, &region_ref, sizeof(region_ref), cursor);
	region = archive->global_archive->getRegion(region_ref);
	break;
      }
    default:
      region = NULL;
    }

  return region ? archive->global_archive->getString(region->string_ref)->str : "INVALID";
}

void Thread::printEventToString(pallas::Event* e, char* output_str, size_t buffer_size) const {
  byte* cursor = nullptr;
  switch (e->record) {
  case PALLAS_EVENT_ENTER: {
    RegionRef region_ref;
    pop_data(e, &region_ref, sizeof(region_ref), cursor);
    const Region* region = archive->global_archive->getRegion(region_ref);
    const char* region_name = region ? archive->global_archive->getString(region->string_ref)->str : "INVALID";
    snprintf(output_str, buffer_size, "Enter %d (%s)", region_ref, region_name);
    break;
  }
  case PALLAS_EVENT_LEAVE: {
    RegionRef region_ref;
    pop_data(e, &region_ref, sizeof(region_ref), cursor);
    const Region* region = archive->global_archive->getRegion(region_ref);
    const char* region_name = region ? archive->global_archive->getString(region->string_ref)->str : "INVALID";
    snprintf(output_str, buffer_size, "Leave %d (%s)", region_ref, region_name);
    break;
  }

  case PALLAS_EVENT_THREAD_BEGIN:
    snprintf(output_str, buffer_size, "THREAD_BEGIN()");
    break;

  case PALLAS_EVENT_THREAD_END:
    snprintf(output_str, buffer_size, "THREAD_END()");
    break;

  case PALLAS_EVENT_THREAD_TEAM_BEGIN:
    snprintf(output_str, buffer_size, "THREAD_TEAM_BEGIN()");
    break;

  case PALLAS_EVENT_THREAD_TEAM_END:
    snprintf(output_str, buffer_size, "THREAD_TEAM_END()");
    break;

  case PALLAS_EVENT_THREAD_FORK: {
    uint32_t numberOfRequestedThreads;
    pop_data(e, &numberOfRequestedThreads, sizeof(numberOfRequestedThreads), cursor);
    snprintf(output_str, buffer_size, "THREAD_FORK(nRequThreads=%d)\n", numberOfRequestedThreads);
    break;
  }

  case PALLAS_EVENT_THREAD_JOIN:
    snprintf(output_str, buffer_size, "THREAD_JOIN\n");
    break;

  case PALLAS_EVENT_MPI_SEND: {
    uint32_t receiver;
    uint32_t communicator;
    uint32_t msgTag;
    uint64_t msgLength;

    pop_data(e, &receiver, sizeof(receiver), cursor);
    pop_data(e, &communicator, sizeof(communicator), cursor);
    pop_data(e, &msgTag, sizeof(msgTag), cursor);
    pop_data(e, &msgLength, sizeof(msgLength), cursor);
    snprintf(output_str, buffer_size, "MPI_SEND(dest=%d, comm=%x, tag=%x, len=%" PRIu64 ")", receiver, communicator,
             msgTag, msgLength);
    break;
  }
  case PALLAS_EVENT_MPI_ISEND: {
    uint32_t receiver;
    uint32_t communicator;
    uint32_t msgTag;
    uint64_t msgLength;
    uint64_t requestID;

    pop_data(e, &receiver, sizeof(receiver), cursor);
    pop_data(e, &communicator, sizeof(communicator), cursor);
    pop_data(e, &msgTag, sizeof(msgTag), cursor);
    pop_data(e, &msgLength, sizeof(msgLength), cursor);
    pop_data(e, &requestID, sizeof(requestID), cursor);
    snprintf(output_str, buffer_size, "MPI_ISEND(dest=%d, comm=%x, tag=%x, len=%" PRIu64 ", req=%" PRIx64 ")", receiver,
             communicator, msgTag, msgLength, requestID);
    break;
  }
  case PALLAS_EVENT_MPI_ISEND_COMPLETE: {
    uint64_t requestID;
    pop_data(e, &requestID, sizeof(requestID), cursor);
    snprintf(output_str, buffer_size, "MPI_ISEND_COMPLETE(req=%" PRIx64 ")", requestID);
    break;
  }
  case PALLAS_EVENT_MPI_IRECV_REQUEST: {
    uint64_t requestID;
    pop_data(e, &requestID, sizeof(requestID), cursor);
    snprintf(output_str, buffer_size, "MPI_IRECV_REQUEST(req=%" PRIx64 ")", requestID);
    break;
  }
  case PALLAS_EVENT_MPI_RECV: {
    uint32_t sender;
    uint32_t communicator;
    uint32_t msgTag;
    uint64_t msgLength;

    pop_data(e, &sender, sizeof(sender), cursor);
    pop_data(e, &communicator, sizeof(communicator), cursor);
    pop_data(e, &msgTag, sizeof(msgTag), cursor);
    pop_data(e, &msgLength, sizeof(msgLength), cursor);

    snprintf(output_str, buffer_size, "MPI_RECV(src=%d, comm=%x, tag=%x, len=%" PRIu64 ")", sender, communicator,
             msgTag, msgLength);
    break;
  }
  case PALLAS_EVENT_MPI_IRECV: {
    uint32_t sender;
    uint32_t communicator;
    uint32_t msgTag;
    uint64_t msgLength;
    uint64_t requestID;
    pop_data(e, &sender, sizeof(sender), cursor);
    pop_data(e, &communicator, sizeof(communicator), cursor);
    pop_data(e, &msgTag, sizeof(msgTag), cursor);
    pop_data(e, &msgLength, sizeof(msgLength), cursor);
    pop_data(e, &requestID, sizeof(requestID), cursor);

    snprintf(output_str, buffer_size, "MPI_IRECV(src=%d, comm=%x, tag=%x, len=%" PRIu64 ", req=%" PRIu64 ")", sender,
             communicator, msgTag, msgLength, requestID);
    break;
  }
  case PALLAS_EVENT_MPI_COLLECTIVE_BEGIN: {
    snprintf(output_str, buffer_size, "MPI_COLLECTIVE_BEGIN()");
    break;
  }
  case PALLAS_EVENT_MPI_COLLECTIVE_END: {
    uint32_t collectiveOp;
    uint32_t communicator;
    uint32_t root;
    uint64_t sizeSent;
    uint64_t sizeReceived;

    pop_data(e, &collectiveOp, sizeof(collectiveOp), cursor);
    pop_data(e, &communicator, sizeof(communicator), cursor);
    pop_data(e, &root, sizeof(root), cursor);
    pop_data(e, &sizeSent, sizeof(sizeSent), cursor);
    pop_data(e, &sizeReceived, sizeof(sizeReceived), cursor);

    snprintf(output_str, buffer_size,
             "MPI_COLLECTIVE_END(op=%x, comm=%x, root=%d, sent=%" PRIu64 ", recved=%" PRIu64 ")", collectiveOp,
             communicator, root, sizeSent, sizeReceived);
    break;
  }
  case PALLAS_EVENT_OMP_FORK: {
    uint32_t numberOfRequestedThreads;
    pop_data(e, &numberOfRequestedThreads, sizeof(numberOfRequestedThreads), cursor);
    snprintf(output_str, buffer_size, "OMP_FORK(nRequThreads=%d)", numberOfRequestedThreads);
    break;
  }
  case PALLAS_EVENT_OMP_JOIN:
    snprintf(output_str, buffer_size, "OMP_JOIN()");
    break;
  case PALLAS_EVENT_OMP_ACQUIRE_LOCK: {
    uint32_t lockID;
    uint32_t acquisitionOrder;
    pop_data(e, &lockID, sizeof(lockID), cursor);
    //pop_data(e, &acquisitionOrder, sizeof(acquisitionOrder), cursor);
    snprintf(output_str, buffer_size, "OMP_ACQUIRE_LOCK(lockID=%d)", lockID);
    break;
  }
  case PALLAS_EVENT_THREAD_ACQUIRE_LOCK: {
    uint32_t lockID;
    uint32_t acquisitionOrder;
    pop_data(e, &lockID, sizeof(lockID), cursor);
    //pop_data(e, &acquisitionOrder, sizeof(acquisitionOrder), cursor);
    snprintf(output_str, buffer_size, "THREAD_ACQUIRE_LOCK(lockID=%d)", lockID);
    break;
  }
  case PALLAS_EVENT_OMP_RELEASE_LOCK: {
    uint32_t lockID;
    uint32_t acquisitionOrder;
    pop_data(e, &lockID, sizeof(lockID), cursor);
    //pop_data(e, &acquisitionOrder, sizeof(acquisitionOrder), cursor);
    snprintf(output_str, buffer_size, "OMP_RELEASE_LOCK(lockID=%d)", lockID);
    break;
  }
  case PALLAS_EVENT_THREAD_RELEASE_LOCK: {
    uint32_t lockID;
    uint32_t acquisitionOrder;
    pop_data(e, &lockID, sizeof(lockID), cursor);
    //pop_data(e, &acquisitionOrder, sizeof(acquisitionOrder), cursor);
    snprintf(output_str, buffer_size, "THREAD_RELEASE_LOCK(lockID=%d)", lockID);
    break;
  }
  case PALLAS_EVENT_OMP_TASK_CREATE: {
    uint64_t taskID;
    pop_data(e, &taskID, sizeof(taskID), cursor);
    snprintf(output_str, buffer_size, "OMP_TASK_CREATE(taskID=%lu)", taskID);
    break;
  }
  case PALLAS_EVENT_OMP_TASK_SWITCH: {
    uint64_t taskID;
    pop_data(e, &taskID, sizeof(taskID), cursor);
    snprintf(output_str, buffer_size, "OMP_TASK_SWITCH(taskID=%lu)", taskID);
    break;
  }
  case PALLAS_EVENT_OMP_TASK_COMPLETE: {
    uint64_t taskID;
    pop_data(e, &taskID, sizeof(taskID), cursor);
    snprintf(output_str, buffer_size, "OMP_TASK_COMPLETE(taskID=%lu)", taskID);
    break;
  }
  case PALLAS_EVENT_THREAD_TASK_CREATE: {
    snprintf(output_str, buffer_size, "THREAD_TASK_CREATE()");
    break;
  }
  case PALLAS_EVENT_THREAD_TASK_SWITCH: {
    snprintf(output_str, buffer_size, "THREAD_TASK_SWITCH()");
    break;
  }
  case PALLAS_EVENT_THREAD_TASK_COMPLETE: {
    snprintf(output_str, buffer_size, "THREAD_TASK_COMPLETE()");
    break;
  }
  case PALLAS_EVENT_GENERIC: {
    StringRef eventNameRef;
    pop_data(e, &eventNameRef, sizeof(eventNameRef), cursor);
    auto eventName = archive->global_archive->getString(eventNameRef);
    snprintf(output_str, buffer_size, "%s", eventName->str);
    break;
  }
  default:
    snprintf(output_str, buffer_size, "{.record: %x, .size:%x}", e->record, e->event_size);
  }
}

Thread::Thread() {
  archive = nullptr;
  id = PALLAS_THREAD_ID_INVALID;

  events = nullptr;
  nb_allocated_events = 0;
  nb_events = 0;

  sequences = nullptr;
  nb_allocated_sequences = 0;
  nb_sequences = 0;

  loops = nullptr;
  nb_allocated_loops = 0;
  nb_loops = 0;
}

void Thread::initThread(Archive* a, ThreadId thread_id) {
  archive = a;
  id = thread_id;

  nb_allocated_events = NB_EVENT_DEFAULT;
  events = new EventSummary[nb_allocated_events]();
  nb_events = 0;

  nb_allocated_sequences = NB_SEQUENCE_DEFAULT;
  sequences = new Sequence*[nb_allocated_sequences]();
  nb_sequences = 0;
  hashToSequence = std::unordered_map<uint32_t, std::vector<TokenId>>();
  hashToEvent = std::unordered_map<uint32_t, std::vector<TokenId>>();

  nb_allocated_loops = NB_LOOP_DEFAULT;
  loops = new Loop[nb_allocated_loops]();
  nb_loops = 0;

  pthread_mutex_lock(&archive->lock);
  while (archive->nb_threads >= archive->nb_allocated_threads) {
    DOUBLE_MEMORY_SPACE(archive->threads, archive->nb_allocated_threads, Thread*);
  }
  for (int i = 0; i < nb_allocated_sequences; i++) {
    sequences[i] = new Sequence();
  }
  archive->threads[archive->nb_threads++] = this;
  pthread_mutex_unlock(&archive->lock);
}

Thread::~Thread() {
  DOFOR(i, nb_events) {
    delete events[i].durations;
    delete[] events[i].attribute_buffer;
  }
  delete[] events;
  DOFOR(i, nb_sequences) {
    delete sequences[i];
  }
  delete[] sequences;
  delete[] loops;
}

const char* Thread::getName() const {
  return archive->global_archive->getString(archive->global_archive->getLocation(id)->name)->str;
}

bool Sequence::isFunctionSequence(const struct Thread* thread) const {
  if (tokens.front().type == TypeEvent && tokens.back().type == TypeEvent) {
    auto frontToken = thread->getEvent(tokens.front());
    auto backToken = thread->getEvent(tokens.back());
    return frontToken->record == PALLAS_EVENT_ENTER && backToken->record == PALLAS_EVENT_LEAVE;
  }
  return false;
};

size_t Sequence::getEventCount(const struct Thread* thread) {
  // TODO This function doesn't really makes sense, since the number of event is dependant on iteration of the loops inside of it.
  return 0;
  // TokenCountMap tokenCount = getTokenCount(thread);
  return tokenCount.getEventCount();
}


void _sequenceGetTokenCountReading(Sequence* seq,
                                   const Thread* thread,
                                   TokenCountMap& readerTokenCountMap,
                                   TokenCountMap& sequenceTokenCountMap,
                                   bool isReversedOrder);


TokenCountMap tempSeen;
void _loopGetTokenCountReading(const Loop* loop,
                               const Thread* thread,
                               TokenCountMap& readerTokenCountMap,
                               TokenCountMap& sequenceTokenCountMap,
                               bool isReversedOrder) {
  size_t cur_index = readerTokenCountMap.get_value(loop->self_id);
  size_t loop_nb_iterations = loop->nb_iterations[cur_index];
  auto* loop_sequence = thread->getSequence(loop->repeated_token);
  if (loop_sequence->contains_loops) {
    for (size_t temp_loop_index = 0; temp_loop_index < loop_nb_iterations; temp_loop_index++) {
      _sequenceGetTokenCountReading(loop_sequence, thread, readerTokenCountMap, sequenceTokenCountMap, isReversedOrder);
      readerTokenCountMap[loop->repeated_token]++;
      sequenceTokenCountMap[loop->repeated_token]++;
    }
  } else {
    // This creates bug idk why ?????
    TokenCountMap temp = loop_sequence->getTokenCountReading(thread, readerTokenCountMap, isReversedOrder);
    temp *= loop_nb_iterations;
    readerTokenCountMap += temp;
    sequenceTokenCountMap += temp;
    readerTokenCountMap[loop->repeated_token]+= loop_nb_iterations;
    sequenceTokenCountMap[loop->repeated_token]+= loop_nb_iterations;
  }
}

void _sequenceGetTokenCountReading(Sequence* seq,
                                            const Thread* thread,
                                            TokenCountMap& readerTokenCountMap,
                                            TokenCountMap& sequenceTokenCountMap,
                                            bool isReversedOrder) {
  for (auto& token : seq->tokens) {
    if (token.type == TypeSequence) {
      auto* s = thread->getSequence(token);
      _sequenceGetTokenCountReading(s, thread, readerTokenCountMap, sequenceTokenCountMap, isReversedOrder);
      seq->contains_loops = seq->contains_loops || s->contains_loops;
    }
    if (token.type == TypeLoop) {
      seq->contains_loops = true;
      auto* loop = thread->getLoop(token);
      _loopGetTokenCountReading(loop, thread, readerTokenCountMap, sequenceTokenCountMap, isReversedOrder);
    }
    readerTokenCountMap[token]++;
    sequenceTokenCountMap[token]++;
  }
}

TokenCountMap Sequence::getTokenCountReading(const Thread* thread,
                                      const TokenCountMap& threadReaderTokenCountMap,
                                      bool isReversedOrder) {
  if (tokenCount.empty()) {
    auto tokenCountMapCopy = TokenCountMap(threadReaderTokenCountMap);
    auto tempTokenCount = TokenCountMap();
    _sequenceGetTokenCountReading(this, thread, tokenCountMapCopy, tempTokenCount, isReversedOrder);
    if (contains_loops) {
      return tempTokenCount;
    } else {
      tokenCount = tempTokenCount;
    }
  }
  return tokenCount;
}

void _sequenceGetTokenCountWriting(Sequence* seq, const Thread* thread, TokenCountMap& reverseTokenCount);

inline static void _loopGetTokenCountWriting(const Loop* loop, const Thread* thread, TokenCountMap& reverseTokenCount) {
  size_t cur_index = loop->nb_iterations.size() - reverseTokenCount[loop->self_id] - 1;
  size_t loop_nb_iterations = loop->nb_iterations[cur_index];
  auto* loop_sequence = thread->getSequence(loop->repeated_token);
  if (loop_sequence->contains_loops) {
    for (size_t temp_loop_index = 0; temp_loop_index < loop_nb_iterations; temp_loop_index++) {
      _sequenceGetTokenCountWriting(loop_sequence, thread, reverseTokenCount);
      reverseTokenCount[loop->repeated_token]++;
    }
  } else {
    auto temp = loop_sequence->getTokenCountWriting(thread);
    reverseTokenCount += temp * loop_nb_iterations;
    reverseTokenCount[loop->repeated_token] += loop_nb_iterations;
  }
}

void _sequenceGetTokenCountWriting(Sequence* seq, const Thread* thread, TokenCountMap& reverseTokenCount) {
  for (auto& token : seq->tokens) {
    if (token.type == TypeSequence) {
      auto* s = thread->getSequence(token);
      _sequenceGetTokenCountWriting(s, thread, reverseTokenCount);
      seq->contains_loops = seq->contains_loops || s->contains_loops;
    }
    if (token.type == TypeLoop) {
      seq->contains_loops = true;
      auto* loop = thread->getLoop(token);
      _loopGetTokenCountWriting(loop, thread, reverseTokenCount);
    }
    reverseTokenCount[token]++;
  }
}

TokenCountMap Sequence::getTokenCountWriting(const Thread* thread, const TokenCountMap* offset) {
  bool canStoreTokenCount = true;
  if (tokenCount.empty()) {
    TokenCountMap updatingOffset;
    if (offset)
      updatingOffset = TokenCountMap(*offset);
    else
      updatingOffset = TokenCountMap();
    for (auto& token : std::ranges::reverse_view(tokens)) {
      updatingOffset[token]++;
      if (token.type == TypeSequence) {
        auto* s = thread->getSequence(token);
        _sequenceGetTokenCountWriting(s, thread, updatingOffset);
        contains_loops = contains_loops || s->contains_loops;
      }
      if (token.type == TypeLoop) {
        canStoreTokenCount = false;
        auto* loop = thread->getLoop(token);
        _loopGetTokenCountWriting(loop, thread, updatingOffset);
      }
    }

    if (offset)
      updatingOffset -= *offset;
    if (!canStoreTokenCount) {
      return updatingOffset;
    }
    tokenCount = updatingOffset;
  }
  return tokenCount;
}
}  // namespace pallas

void* pallas_realloc(void* buffer, int cur_size, int new_size, size_t datatype_size) {
  void* new_buffer = (void*)realloc(buffer, new_size * datatype_size);
  if (new_buffer == NULL) {
    new_buffer = (void*)calloc(new_size, datatype_size);
    if (new_buffer == NULL) {
      pallas_error("Failed to allocate memory using realloc AND malloc\n");
    }
    memmove(new_buffer, buffer, cur_size * datatype_size);
    free(buffer);
  } else {
    /* realloc changed the size of the buffer, leaving some bytes */
    /* uninitialized. Let's fill the rest of the buffer with zeros to*/
    /* prevent problems. */

    if (new_size > cur_size) {
      uintptr_t old_end_addr = (uintptr_t)(new_buffer) + (cur_size * datatype_size);
      uintptr_t rest_size = (new_size - cur_size) * datatype_size;
      memset((void*)old_end_addr, 0, rest_size);
    }
  }
  return new_buffer;
}

/* C bindings now */

pallas::Thread* pallas_thread_new() {
  return new pallas::Thread();
};

const char* pallas_thread_get_name(pallas::Thread* thread) {
  return thread->getName();
}

void pallas_print_sequence(pallas::Thread* thread, pallas::Token seq_id) {
  thread->printSequence(seq_id);
}

void pallas_print_token_array(pallas::Thread* thread, pallas::Token* token_array, int index_start, int index_stop) {
  thread->printTokenArray(token_array, index_start, index_stop);
}

void pallas_print_token(pallas::Thread* thread, pallas::Token token) {
  thread->printToken(token);
}

void pallas_print_event(pallas::Thread* thread, pallas::Event* e) {
  thread->printEvent(e);
}
pallas::Loop* pallas_get_loop(pallas::Thread* thread, pallas::Token id) {
  return thread->getLoop(id);
}
pallas::Sequence* pallas_get_sequence(pallas::Thread* thread, pallas::Token id) {
  return thread->getSequence(id);
}
pallas::Event* pallas_get_event(pallas::Thread* thread, pallas::Token id) {
  return thread->getEvent(id);
}
pallas::Token pallas_get_token(pallas::Thread* thread, pallas::Token sequence, int index) {
  return thread->getToken(sequence, index);
}

size_t pallas_sequence_get_size(pallas::Sequence* sequence) {
  return sequence->size();
}
pallas::Token pallas_sequence_get_token(pallas::Sequence* sequence, int index) {
  return sequence->tokens[index];
}

size_t pallas_loop_count(pallas::Loop* loop) {
  return loop->nb_iterations.size();
};
size_t pallas_loop_get_count(PALLAS(Loop) * loop, size_t index) {
  return loop->nb_iterations[index];
};

/* -*-
  mode: cpp;
  c-file-style: "k&r";
  c-basic-offset 2;
  tab-width 2 ;
  indent-tabs-mode nil
  -*- */

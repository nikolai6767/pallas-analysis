/*
 * Copyright (C) Telecom SudParis
 * See LICENSE in top-level directory.
 */

#include "pallas/pallas_read.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include "pallas/pallas.h"
#include "pallas/pallas_archive.h"
#include "pallas/pallas_log.h"

namespace pallas {

CallstackFrame::CallstackFrame() {
  this->referential_timestamp = 0;
  this->frame_index = 0;
}

CallstackFrame::~CallstackFrame() = default;

ThreadReader::ThreadReader(Archive* archive, ThreadId threadId, int read_flags) {
  // Setup the basic
  this->archive = archive;
  this->pallas_read_flag = read_flags;

  pallas_assert(threadId != PALLAS_THREAD_ID_INVALID);
  this->thread_trace = archive->getThread(threadId);
  pallas_assert(this->thread_trace != nullptr);

  if (debugLevel >= DebugLevel::Verbose) {
    pallas_log(DebugLevel::Verbose, "init callstack for thread %d\n", threadId);
    pallas_log(DebugLevel::Verbose, "The trace contains:\n");
    this->thread_trace->printSequence(Token(TypeSequence, 0));
  }

  // And initialize the callstack
  // ie set the cursor on the first event
  this->current_frame_index = 0;
  this->currentState = &callstack[0];
  currentState->callstack_iterable = Token(TypeSequence, 0);

  // Enter main sequence
  enterBlock();
}

const Token& ThreadReader::getFrameInCallstack(int frame_number) const {
  if (frame_number < 0 || frame_number >= MAX_CALLSTACK_DEPTH) {
    pallas_error("Frame number is too high or negative: %d\n", frame_number);
  }
  return callstack[frame_number].callstack_iterable;
}

const Token& ThreadReader::getTokenInCallstack(int frame_number) const {
  if (frame_number < 0 || frame_number >= MAX_CALLSTACK_DEPTH) {
    pallas_error("Frame number is too high or negative: %d\n", frame_number);
  }
  auto sequence = getFrameInCallstack(frame_number);
  pallas_assert(sequence.isIterable());
  return thread_trace->getToken(sequence, callstack[frame_number].frame_index);
}
void ThreadReader::printCurToken() const {
  thread_trace->printToken(pollCurToken());
}
const Token& ThreadReader::getCurIterable() const {
  return currentState->callstack_iterable;
}
void ThreadReader::printCurSequence() const {
  thread_trace->printSequence(getCurIterable());
}

void ThreadReader::printCallstack() const {
  printf("# Callstack (depth: %d) ------------\n", current_frame_index + 1);
  for (int i = 0; i < current_frame_index + 1; i++) {
    auto current_sequence_id = getFrameInCallstack(i);
    auto current_token = getTokenInCallstack(i);

    printf("%.*s[%d] ", i * 2, "                       ", i);
    thread_trace->printToken(current_sequence_id);

    if (current_sequence_id.type == TypeLoop) {
      auto* loop = thread_trace->getLoop(current_sequence_id);
      printf(" iter %d/%d", callstack[i].frame_index,
             loop->nb_iterations[currentState->tokenCount.get_value(current_sequence_id)]);
      pallas_assert(callstack[i].frame_index < MAX_CALLSTACK_DEPTH);
    } else if (current_sequence_id.type == TypeSequence) {
      auto* sequence = thread_trace->getSequence(current_sequence_id);
      printf(" pos %d/%lu", callstack[i].frame_index, sequence->size());
      pallas_assert(callstack[i].frame_index < MAX_CALLSTACK_DEPTH);
    }

    printf("\t-> ");
    thread_trace->printToken(current_token);
    printf("\n");
  }
}
EventSummary* ThreadReader::getEventSummary(Token event) const {
  pallas_assert(event.type == TypeEvent);
  if (event.id < thread_trace->nb_events) {
    return &thread_trace->events[event.id];
  }
  pallas_error("Given event (%d) was invalid\n", event.id);
}
pallas_timestamp_t ThreadReader::getEventTimestamp(Token event, int occurence_id) const {
  pallas_assert(event.type == TypeEvent);
  auto summary = getEventSummary(event);
  if (0 <= occurence_id && occurence_id < summary->nb_occurences) {
    return summary->durations->at(occurence_id);
  }
  pallas_error("Given occurence_id (%d) was invalid for event %d\n", occurence_id, event.id);
}
bool ThreadReader::isEndOfSequence(int current_index, Token sequence_id) const {
  if (sequence_id.type == TypeSequence) {
    auto* sequence = thread_trace->getSequence(sequence_id);
    return current_index + 1 >= sequence->size();
    // We are in a sequence and index is beyond the end of the sequence
  }
  pallas_error("The given sequence_id was the wrong type: %d\n", sequence_id.type);
}
bool ThreadReader::isEndOfLoop(int current_index, Token loop_id) const {
  if (loop_id.type == TypeLoop) {
    auto* loop = thread_trace->getLoop(loop_id);
    pallas_assert(current_index < loop->nb_iterations[currentState->tokenCount.get_value(loop_id)]);
    return current_index + 1 >= loop->nb_iterations[currentState->tokenCount.get_value(loop_id)];
    // We are in a loop and index is beyond the number of iterations
  }
  pallas_error("The given loop_id was the wrong type: %d\n", loop_id.type);
}
bool ThreadReader::isEndOfBlock(int index, Token iterable_token) const {
  switch (iterable_token.type) {
  case TypeSequence:
    return isEndOfSequence(index, iterable_token);
  case TypeLoop:
    return isEndOfLoop(index, iterable_token);
  case TypeEvent:
    return false;
  case TypeInvalid:
    pallas_error("Current frame is invalid");
  }
  return false;
}
bool ThreadReader::isEndOfCurrentBlock() const {
  pallas_assert(current_frame_index >= 0);

  int current_index = currentState->frame_index;
  auto current_iterable_token = currentState->callstack_iterable;

  return isEndOfBlock(current_index, current_iterable_token);
}
bool ThreadReader::isEndOfTrace() const {
  return current_frame_index == 0;
}

pallas_duration_t ThreadReader::getLoopDuration(Token loop_id) const {
  pallas_assert(loop_id.type == TypeLoop);
  pallas_duration_t sum = 0;
  const auto* loop = thread_trace->getLoop(loop_id);
  const auto* sequence = thread_trace->getSequence(loop->repeated_token);

  const Token sequence_id = loop->repeated_token;

  const size_t loopIndex = currentState->tokenCount.get_value(loop_id);
  const size_t offset = currentState->tokenCount.get_value(sequence_id);
  const size_t nIterations = loop->nb_iterations.at(loopIndex);
  DOFOR(i, nIterations) {
    sum += sequence->durations->at(offset + i);
  }
  return sum;
}

EventOccurence ThreadReader::getEventOccurence(Token event_id, size_t occurence_id) const {
  auto eventOccurence = EventOccurence();
  auto* es = getEventSummary(event_id);
  eventOccurence.event = thread_trace->getEvent(event_id);

  eventOccurence.timestamp = currentState->referential_timestamp;
  eventOccurence.duration = es->durations->at(occurence_id);
  eventOccurence.attributes = getEventAttributeList(event_id, occurence_id);
  return eventOccurence;
}

SequenceOccurence ThreadReader::getSequenceOccurence(Token sequence_id,
                                                     size_t occurence_id) const {
  auto sequenceOccurence = SequenceOccurence();
  sequenceOccurence.sequence = thread_trace->getSequence(sequence_id);

  sequenceOccurence.timestamp = currentState->referential_timestamp;
  sequenceOccurence.duration = sequenceOccurence.sequence->durations->at(occurence_id);
  sequenceOccurence.full_sequence = nullptr;

  return sequenceOccurence;
};

LoopOccurence ThreadReader::getLoopOccurence(Token loop_id, size_t occurence_id) const {
  auto loopOccurence = LoopOccurence();
  loopOccurence.loop = thread_trace->getLoop(loop_id);
  loopOccurence.nb_iterations = loopOccurence.loop->nb_iterations[occurence_id];
  loopOccurence.full_loop = nullptr;
  loopOccurence.timestamp = currentState->referential_timestamp;
  loopOccurence.duration = getLoopDuration(loop_id);
  return loopOccurence;
}

AttributeList* ThreadReader::getEventAttributeList(Token event_id, size_t occurence_id) const {
  auto* summary = getEventSummary(event_id);
  if (summary->attribute_buffer == nullptr)
    return nullptr;

  if (summary->attribute_pos < summary->attribute_buffer_size) {
    auto* l = (AttributeList*)&summary->attribute_buffer[summary->attribute_pos];

    while (l->index < occurence_id) { /* move to the next attribute until we reach the needed index */
      summary->attribute_pos += l->struct_size;
      l = (AttributeList*)&summary->attribute_buffer[summary->attribute_pos];
    }
    if (l->index == occurence_id) {
      return l;
    }
    if (l->index > occurence_id) {
      pallas_error("Error fetching attribute %zu. We went too far (cur position: %d) !\n", occurence_id, l->index);
    }
  }
  return nullptr;
}

//******************* EXPLORATION FUNCTIONS ********************

const Token& ThreadReader::pollCurToken() const {
  return getTokenInCallstack(current_frame_index);
}

Token ThreadReader::pollNextToken(int flags) const {
  if (current_frame_index < 0)
    // return an invalid token
    return Token();

  if (flags == PALLAS_READ_FLAG_NONE)
    flags = pallas_read_flag;

  int current_frame = current_frame_index;
  int current_index = currentState->frame_index;
  auto current_iterable_token = currentState->callstack_iterable;
  pallas_assert(current_iterable_token.isIterable());

  if (const Token current_token = pollCurToken(); current_token.isIterable()) {
    if (current_token.type == TypeSequence && flags & PALLAS_READ_FLAG_UNROLL_SEQUENCE) {
      return thread_trace->getSequence(current_token)->tokens.at(0);
    }
    if (current_token.type == TypeLoop && flags & PALLAS_READ_FLAG_UNROLL_LOOP) {
      return thread_trace->getLoop(current_token)->repeated_token;
    }
  }
  while (isEndOfBlock(current_index, current_iterable_token)) {
    if (current_frame == 0)
      return Token();
    if (current_iterable_token.type == TypeSequence && flags & PALLAS_READ_FLAG_UNROLL_SEQUENCE) {
      current_frame--;
      current_index = currentState->frame_index;
      current_iterable_token = currentState->callstack_iterable;
    } else if (current_iterable_token.type == TypeLoop && flags & PALLAS_READ_FLAG_UNROLL_LOOP) {
      current_frame--;
      current_index = currentState->frame_index;
      current_iterable_token = currentState->callstack_iterable;
    } else {
      return Token();
    }
  }
  return thread_trace->getToken(current_iterable_token, current_index + 1);

}

Token ThreadReader::pollPrevToken(int flags) const {
  if (current_frame_index < 0)
    return Token();

  if (flags == PALLAS_READ_FLAG_NONE)
    flags = pallas_read_flag;

  int current_frame = current_frame_index;
  int current_index = currentState->frame_index;
  auto current_iterable_token = currentState->callstack_iterable;
  pallas_assert(current_iterable_token.isIterable());

  while (current_index == 0) {
    if (current_frame == 0)
      return Token();
    if (current_iterable_token.type == TypeSequence && flags & PALLAS_READ_FLAG_UNROLL_SEQUENCE) {
      current_frame--;
      current_index = currentState->frame_index;
      current_iterable_token = currentState->callstack_iterable;
    } else if (current_iterable_token.type == TypeLoop && flags & PALLAS_READ_FLAG_UNROLL_LOOP) {
      current_frame--;
      current_index = currentState->frame_index;
      current_iterable_token = currentState->callstack_iterable;
    } else {
      return Token();
    }
  }
  Token result = thread_trace->getToken(current_iterable_token, current_index - 1);
  while (result.isIterable()) {
    if (result.type == TypeSequence && flags & PALLAS_READ_FLAG_UNROLL_SEQUENCE) {
      result = thread_trace->getSequence(result)->tokens.at(thread_trace->getSequence(result)->tokens.size() - 1);
    } else if (result.type == TypeLoop && flags & PALLAS_READ_FLAG_UNROLL_LOOP) {
      result = thread_trace->getLoop(result)->repeated_token;
    } else if (result.type == TypeSequence || result.type == TypeLoop) {
      break;
    }
  }
  return result;
}

bool ThreadReader::moveToNextToken(int flags) {
  // Check if we've reached the end of the trace
  if (current_frame_index < 0) {
    pallas_log(DebugLevel::Debug, "End of trace %d!\n", __LINE__);
    return false;
  }

  if (flags == PALLAS_READ_FLAG_NONE)
    flags = pallas_read_flag;

  auto current_iterable_token = getCurIterable();
  pallas_assert(current_iterable_token.isIterable());

  auto current_token = this->pollCurToken();

  // If we can enter a block, then we enter
  if (enterIfStartOfBlock(flags))
    return true;

  // Exit every block we can
  bool exited = exitIfEndOfBlock(flags);
  while (exitIfEndOfBlock(flags)) {
  }

  if (isEndOfCurrentBlock())
    return false;

  current_token = this->pollCurToken();

  // Update referential timestamp and token count according to current token
  pallas_duration_t token_duration = 0;
  switch (current_token.type) {
  case TypeEvent:
    token_duration = getEventSummary(current_token)->durations->at(currentState->tokenCount[current_token]);
    break;

    case TypeLoop: {
      token_duration = getLoopDuration(current_token);
      auto loop = thread_trace->getLoop(current_token);
      auto loopCount = loop->nb_iterations[currentState->tokenCount[current_token]];
      for (size_t i = 0; i < loopCount; i ++) {
        currentState->tokenCount += thread_trace->getSequence(loop->repeated_token)->getTokenCountReading(thread_trace, currentState->tokenCount);
        currentState->tokenCount[loop->repeated_token] += 1;
      }
      break;
    }

    case TypeSequence: {
      auto seq = thread_trace->getSequence(current_token);
      token_duration = seq->durations->at(currentState->tokenCount[current_token]);
      currentState->tokenCount += seq->getTokenCountReading(thread_trace, currentState->tokenCount);
      break;
    }

    case TypeInvalid:
      pallas_error("Token is Invalid");
    }
#ifdef DEBUG
  // Some error checking, helps debugging writing errors
  if (currentState->referential_timestamp + token_duration < currentState->referential_timestamp) {
      pallas_error("Token duration negative for (%c.%d): %lu\n", PALLAS_TOKEN_TYPE_C(current_token), current_token.id,
                   token_duration);
    }
#endif

  currentState->referential_timestamp += token_duration;
  currentState->tokenCount[current_token]++;
  currentState->frame_index++;

  return true;
}

bool ThreadReader::moveToNextTokenInBlock() {
  return moveToNextToken(PALLAS_READ_FLAG_NO_UNROLL);
}

bool ThreadReader::moveToPrevToken(int flags) {
  // Check if we've reached the end of the trace
  if (current_frame_index < 0) {
    pallas_log(DebugLevel::Debug, "End of trace %d!\n", __LINE__);
    return false;
  }

  if (flags == PALLAS_READ_FLAG_NONE)
    flags = pallas_read_flag;

  auto current_iterable_token = currentState->callstack_iterable;
  pallas_assert(current_iterable_token.isIterable());

  if (currentState->frame_index == 1) {
    if (current_frame_index <= 1)
      return false;
    Token current_iterable_token = currentState->callstack_iterable;
    if (current_iterable_token.type == TypeSequence && !(flags & PALLAS_READ_FLAG_UNROLL_SEQUENCE)) {
      return false;
    }
    if (current_iterable_token.type == TypeLoop && !(flags & PALLAS_READ_FLAG_UNROLL_LOOP)) {
      return false;
    }
    leaveBlock();
    return true;
  }

  /* Move to the previous event in the Sequence */
  auto previous_token = pollPrevToken(PALLAS_READ_FLAG_NO_UNROLL);

  while (previous_token.isIterable()) {
    size_t ntokens;
    TokenCountMap previous_token_count;
    if (previous_token.type == TypeSequence && flags & PALLAS_READ_FLAG_UNROLL_SEQUENCE) {
      ntokens = thread_trace->getSequence(previous_token)->size();
    } else if (previous_token.type == TypeLoop && flags & PALLAS_READ_FLAG_UNROLL_LOOP) {
      ntokens = thread_trace->getLoop(previous_token)->nb_iterations.at(currentState->tokenCount[previous_token]-1);
    } else {
      break;
    }
    // Save cursor before entering
    // TODO : Optimiser ça
    moveToPrevToken(PALLAS_READ_FLAG_NO_UNROLL);
    if (!moveToNextToken(PALLAS_READ_FLAG_NO_UNROLL)) { // If we didn't come back, we come back manually
      Token current_token = pollCurToken();
      pallas_duration_t token_duration = 0;
      switch (current_token.type) {
      case TypeEvent:
        token_duration = getEventSummary(current_token)->durations->at(currentState->tokenCount[current_token]);
        break;

      case TypeLoop:
        token_duration = getLoopDuration(current_token);
        for (int i = 0; i < thread_trace->getLoop(current_token)->nb_iterations[currentState->tokenCount[current_token]];
             i++) {
          currentState->tokenCount +=
            thread_trace->getSequence(thread_trace->getLoop(current_token)->repeated_token)->getTokenCountReading(thread_trace, currentState->tokenCount, true);
          currentState->tokenCount[thread_trace->getLoop(current_token)->repeated_token]++;
             }
        break;

      case TypeSequence:
        token_duration = thread_trace->getSequence(current_token)->durations->at(currentState->tokenCount[current_token]);
        currentState->tokenCount += thread_trace->getSequence(current_token)->getTokenCountReading(thread_trace, currentState->tokenCount, true);
        break;

      case TypeInvalid:
        pallas_error("Token is Invalid");
      }

      currentState->referential_timestamp += token_duration;
      currentState->tokenCount[current_token]++;
      currentState->frame_index++;
    }
    currentState->tokenCount[previous_token]--;
    // Enter block
    current_frame_index++;
    currentState++;
    currentState->callstack_iterable = previous_token;
    currentState->frame_index = ntokens;
    previous_token = pollPrevToken(PALLAS_READ_FLAG_NO_UNROLL);
  }

  currentState->tokenCount[previous_token]--;
  currentState->frame_index--;

  switch (previous_token.type) {
  case TypeEvent:
    currentState->referential_timestamp -=
      getEventSummary(previous_token)->durations->at(currentState->tokenCount[previous_token]);
    break;

  case TypeLoop: {
    auto prev_loop = thread_trace->getLoop(previous_token);
    for (int i = 0; i < prev_loop->nb_iterations[currentState->tokenCount[previous_token]]; i++) {
      currentState->tokenCount -= thread_trace->getSequence(prev_loop->repeated_token)->getTokenCountReading(thread_trace, currentState->tokenCount, true);
      currentState->tokenCount[prev_loop->repeated_token]--;
    }
    currentState->referential_timestamp -= getLoopDuration(previous_token);
    break;
  }
  case TypeSequence: {
    auto prev_sequence = thread_trace->getSequence(previous_token);
    currentState->tokenCount -= prev_sequence->getTokenCountReading(thread_trace, currentState->tokenCount, true);
    currentState->referential_timestamp -= prev_sequence->durations->at(currentState->tokenCount[previous_token]);
    break;
  }
  case TypeInvalid:
    pallas_error("Token is Invalid");
  }
  return true;
}
bool ThreadReader::moveToPrevTokenInBlock() {
  return moveToPrevToken(PALLAS_READ_FLAG_NO_UNROLL);
}

Token ThreadReader::getNextToken(int flags) {
  if (flags == PALLAS_READ_FLAG_NONE)
    flags = pallas_read_flag;
  if (!moveToNextToken(flags))
    return Token();
  return pollCurToken();
}
Token ThreadReader::getPrevToken(int flags) {
  if (flags == PALLAS_READ_FLAG_NONE)
    flags = pallas_read_flag;
  if (!moveToPrevToken(flags))
    return Token();
  return pollCurToken();
}

void ThreadReader::enterBlock() {
  auto new_block = pollCurToken();
  pallas_assert(new_block.isIterable());
  if (debugLevel >= DebugLevel::Debug) {
    pallas_log(DebugLevel::Debug, "[%d] Enter Block ", current_frame_index);
    printCurToken();
    printf("\n");
  }

  current_frame_index++;
  currentState++;
  currentState->frame_index = 0;
  currentState->referential_timestamp = callstack[current_frame_index-1].referential_timestamp;
  currentState->callstack_iterable = new_block;
  currentState->tokenCount = (currentState-1)->tokenCount;
}

void ThreadReader::leaveBlock() {
  if (debugLevel >= DebugLevel::Debug) {
    pallas_log(DebugLevel::Debug, "[%d] Leave ", current_frame_index);
    printCurSequence();
    printf("\n");
  }

  pallas_assert(current_frame_index > 0);

  current_frame_index--;
  currentState--;

  if (debugLevel >= DebugLevel::Debug && current_frame_index >= 0) {
    auto current_sequence = getCurIterable();
    pallas_assert(current_sequence.isIterable());
  }
}

bool ThreadReader::exitIfEndOfBlock(int flags) {
  if (flags == PALLAS_READ_FLAG_NONE)
    flags = pallas_read_flag;

  if (current_frame_index == 0)
    return false;
  int current_index = currentState->frame_index;
  auto current_iterable_token = currentState->callstack_iterable;
  if (current_iterable_token.type == TypeSequence) {
    if (isEndOfSequence(current_index, current_iterable_token) && flags & PALLAS_READ_FLAG_UNROLL_SEQUENCE) {
      /* We've reached the end of a sequence. Leave the block. */
      leaveBlock();
      return true;
    }
  } else {
    if (isEndOfLoop(current_index, current_iterable_token) && flags & PALLAS_READ_FLAG_UNROLL_LOOP) {
      /* We've reached the end of the loop. Leave the block. */
      leaveBlock();
      return true;
    }
  }
  return false;
}
bool ThreadReader::enterIfStartOfBlock(int flags) {
  if (flags == PALLAS_READ_FLAG_NONE)
    flags = pallas_read_flag;

  auto current_token = pollCurToken();
  if (!current_token.isIterable())
    return false;
  if (current_token.type == TypeSequence && flags & PALLAS_READ_FLAG_UNROLL_SEQUENCE) {
    /* We've reached the end of a sequence. Leave the block. */
    enterBlock();
    return true;
  }
  if (current_token.type == TypeLoop && flags & PALLAS_READ_FLAG_UNROLL_LOOP) {
    /* We've reached the end of the loop. Leave the block. */
    enterBlock();
    return true;
  }
  return false;
}

ThreadReader::~ThreadReader() {
  if (archive)
    archive->freeThread(thread_trace->id);
}

ThreadReader::ThreadReader(const ThreadReader& other) {
  archive = other.archive;
  thread_trace = other.thread_trace;
  current_frame_index = other.current_frame_index;
  DOFOR(i, MAX_CALLSTACK_DEPTH) {
    callstack[i].tokenCount = other.callstack[i].tokenCount;
    callstack[i].frame_index = other.callstack[i].frame_index;
    callstack[i].callstack_iterable = other.callstack[i].callstack_iterable;
    callstack[i].referential_timestamp = other.callstack[i].referential_timestamp;
  }
  currentState = &callstack[current_frame_index];
  pallas_read_flag = other.pallas_read_flag;
}

ThreadReader::ThreadReader(ThreadReader&& other) noexcept {
  archive = other.archive;
  thread_trace = other.thread_trace;
  current_frame_index = other.current_frame_index;
  DOFOR(i, MAX_CALLSTACK_DEPTH) {
    callstack[i].tokenCount = other.callstack[i].tokenCount;
    callstack[i].frame_index = other.callstack[i].frame_index;
    callstack[i].callstack_iterable = other.callstack[i].callstack_iterable;
    callstack[i].referential_timestamp = other.callstack[i].referential_timestamp;
  }
  currentState = &callstack[current_frame_index];
  pallas_read_flag = other.pallas_read_flag;
  // Set other to 0 for everything
  other.archive = nullptr;
  other.thread_trace = nullptr;
  other.currentState = nullptr;
  other.pallas_read_flag = 0;
}

ThreadReader& ThreadReader::operator=(const ThreadReader& other) {
  archive = other.archive;
  thread_trace = other.thread_trace;
  current_frame_index = other.current_frame_index;
  DOFOR(i, MAX_CALLSTACK_DEPTH) {
    callstack[i].tokenCount = other.callstack[i].tokenCount;
    callstack[i].frame_index = other.callstack[i].frame_index;
    callstack[i].callstack_iterable = other.callstack[i].callstack_iterable;
    callstack[i].referential_timestamp = other.callstack[i].referential_timestamp;
  }
  currentState = &callstack[current_frame_index];
  pallas_read_flag = other.pallas_read_flag;
  return *this;
}

ThreadReader& ThreadReader::operator=(ThreadReader&& other) noexcept {
  archive = other.archive;
  thread_trace = other.thread_trace;
  current_frame_index = other.current_frame_index;
  DOFOR(i, MAX_CALLSTACK_DEPTH) {
    callstack[i].tokenCount = other.callstack[i].tokenCount;
    callstack[i].frame_index = other.callstack[i].frame_index;
    callstack[i].callstack_iterable = other.callstack[i].callstack_iterable;
    callstack[i].referential_timestamp = other.callstack[i].referential_timestamp;
  }
  currentState = &callstack[current_frame_index];
  pallas_read_flag = other.pallas_read_flag;
  // Set other to 0 for everything
  other.archive = nullptr;
  other.thread_trace = nullptr;
  other.currentState = nullptr;
  other.pallas_read_flag = 0;
  return *this;
}

ThreadReader ThreadReader::copy() const {
  return ThreadReader(*this);
}

ThreadReader pallasCreateThreadReader(Archive* archive, ThreadId threadId, int options) {
  return {archive, threadId, options};
}
void pallasPrintCurToken(ThreadReader* thread_reader) {
  thread_reader->printCurToken();
}
Token pallasGetCurIterable(ThreadReader* thread_reader) {
  return thread_reader->getCurIterable();
}
void pallasPrintCurSequence(ThreadReader* thread_reader) {
  thread_reader->printCurSequence();
}
void pallasPrintCallstack(ThreadReader* thread_reader) {
  thread_reader->printCallstack();
}
EventSummary* pallasGetEventSummary(ThreadReader* thread_reader, Token event) {
  return thread_reader->getEventSummary(event);
}
pallas_timestamp_t pallasGetEventTimestamp(ThreadReader* thread_reader, Token event, int occurence_id) {
  return thread_reader->getEventTimestamp(event, occurence_id);
}
bool pallasIsEndOfSequence(ThreadReader* thread_reader, int current_index, Token sequence_id) {
  return thread_reader->isEndOfSequence(current_index, sequence_id);
}
bool pallasIsEndOfLoop(ThreadReader* thread_reader, int current_index, Token loop_id) {
  return thread_reader->isEndOfLoop(current_index, loop_id);
}
bool pallasIsEndOfCurrentBlock(ThreadReader* thread_reader) {
  return thread_reader->isEndOfCurrentBlock();
}
bool pallasIsEndOfTrace(ThreadReader* thread_reader) {
  return thread_reader->isEndOfTrace();
}
EventOccurence pallasGetEventOccurence(ThreadReader* thread_reader, Token event_id, size_t occurence_id) {
  return thread_reader->getEventOccurence(event_id, occurence_id);
}
SequenceOccurence pallasGetSequenceOccurence(ThreadReader* thread_reader,
                                             Token sequence_id,
                                             size_t occurence_id) {
  return thread_reader->getSequenceOccurence(sequence_id, occurence_id);
}
LoopOccurence pallasGetLoopOccurence(ThreadReader* thread_reader, Token loop_id, size_t occurence_id) {
  return thread_reader->getLoopOccurence(loop_id, occurence_id);
}
AttributeList* pallasGetEventAttributeList(ThreadReader* thread_reader, Token event_id, size_t occurence_id) {
  return thread_reader->getEventAttributeList(event_id, occurence_id);
}
Token pallasPollCurToken(ThreadReader* thread_reader) {
  return thread_reader->pollCurToken();
}
Token pallasPollNextToken(ThreadReader* thread_reader, int flags) {
  return thread_reader->pollNextToken(flags);
}
Token pallasPollPrevToken(ThreadReader* thread_reader, int flags) {
  return thread_reader->pollPrevToken(flags);
}
bool pallasMoveToNextToken(ThreadReader* thread_reader, int flags) {
  return thread_reader->moveToNextToken(flags);
}
bool pallasMoveToNextTokenInBlock(ThreadReader* thread_reader) {
  return pallasMoveToNextToken(thread_reader, PALLAS_READ_FLAG_NO_UNROLL);
}
bool pallasMoveToPrevToken(ThreadReader* thread_reader, int flags) {
  return thread_reader->moveToPrevToken(flags);
}
bool pallasMoveToPrevTokenInBlock(ThreadReader* thread_reader) {
  return pallasMoveToPrevToken(thread_reader, PALLAS_READ_FLAG_NO_UNROLL);
}
Token pallasGetNextToken(ThreadReader* thread_reader, int flags) {
  return thread_reader->getNextToken(flags);
}
Token pallasGetPrevToken(ThreadReader* thread_reader, int flags) {
  return thread_reader->getPrevToken(flags);
}
void pallasEnterBlock(ThreadReader* thread_reader) {
  thread_reader->enterBlock();
}
void pallasLeaveBlock(ThreadReader* thread_reader) {
  thread_reader->leaveBlock();
}
bool pallasExitIfEndOfBlock(ThreadReader* thread_reader, int flags) {
  return thread_reader->exitIfEndOfBlock(flags);
}
bool pallasEnterIfStartOfBlock(ThreadReader* thread_reader, int flags) {
  return thread_reader->enterIfStartOfBlock(flags);
}
ThreadReader pallasCreateCheckpoint(ThreadReader *thread_reader) {
  return thread_reader->copy();
}
void pallasLoadCheckpoint(ThreadReader *dest, ThreadReader *src) {
  *dest = src->copy();
}

TokenOccurence::~TokenOccurence() {
  if (token == nullptr || occurence == nullptr) {
    return;
  }
  if (token->type == TypeLoop) {
    auto& loopOccurence = occurence->loop_occurence;
    if (loopOccurence.full_loop) {
      delete[] loopOccurence.full_loop;
    }
  }
  delete occurence;
}

} /* namespace pallas */

/* -*-
   mode: c;
   c-file-style: "k&r";
   c-basic-offset 2;
   tab-width 2 ;
   indent-tabs-mode nil
   -*- */

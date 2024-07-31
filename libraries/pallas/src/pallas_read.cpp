/*
 * Copyright (C) Telecom SudParis
 * See LICENSE in top-level directory.
 */

#include "pallas/pallas_read.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include "pallas/pallas_archive.h"
#include "pallas/pallas_log.h"

namespace pallas {

Checkpoint::Checkpoint(const ThreadReader* reader) {
  if ((reader->options & ThreadReaderOptions::NoTimestamps) == 0) {
    referential_timestamp = reader->referential_timestamp;
  }

  memcpy(callstack_iterable, reader->callstack_iterable, sizeof(Token) * MAX_CALLSTACK_DEPTH);

  memcpy(callstack_index, reader->callstack_index, sizeof(int) * MAX_CALLSTACK_DEPTH);

  current_frame = reader->current_frame;

  tokenCount = reader->tokenCount;
}
Checkpoint::~Checkpoint() {
}

ThreadReader::ThreadReader(Archive* archive, ThreadId threadId, int options) {
  // Setup the basic
  this->archive = archive;
  this->options = options;
  pallas_assert(threadId != PALLAS_THREAD_ID_INVALID);
  thread_trace = archive->getThread(threadId);
  pallas_assert(thread_trace != nullptr);

  if (debugLevel >= DebugLevel::Verbose) {
    pallas_log(DebugLevel::Verbose, "init callstack for thread %d\n", threadId);
    pallas_log(DebugLevel::Verbose, "The trace contains:\n");
    thread_trace->printSequence(Token(TypeSequence, 0));
  }

  // And initialize the callstack
  // ie set the cursor on the first event
  referential_timestamp = 0;
  current_frame = 0;
  std::memset(callstack_index, 0, MAX_CALLSTACK_DEPTH * sizeof(int));
  std::memset((void*)callstack_iterable, 0, MAX_CALLSTACK_DEPTH * sizeof(Token));
  callstack_iterable[0].type = TypeSequence;
  callstack_iterable[0].id = 0;

  // Enter sequence 0
  enterBlock(pollCurToken());
}

const Token& ThreadReader::getFrameInCallstack(int frame_number) const {
  if (frame_number < 0 || frame_number >= MAX_CALLSTACK_DEPTH) {
    pallas_error("Frame number is too high or negative: %d\n", frame_number);
  }
  return callstack_iterable[frame_number];
}

const Token& ThreadReader::getTokenInCallstack(int frame_number) const {
  if (frame_number < 0 || frame_number >= MAX_CALLSTACK_DEPTH) {
    pallas_error("Frame number is too high or negative: %d\n", frame_number);
  }
  auto sequence = getFrameInCallstack(frame_number);
  pallas_assert(sequence.isIterable());
  return thread_trace->getToken(sequence, callstack_index[frame_number]);
}
void ThreadReader::printCurToken() const {
  thread_trace->printToken(pollCurToken());
}
const Token& ThreadReader::getCurIterable() const {
  return getFrameInCallstack(current_frame);
}
void ThreadReader::printCurSequence() const {
  thread_trace->printSequence(getCurIterable());
}

void ThreadReader::printCallstack() const {
  printf("# Callstack (depth: %d) ------------\n", current_frame + 1);
  for (int i = 0; i < current_frame + 1; i++) {
    auto current_sequence_id = getFrameInCallstack(i);
    auto current_token = getTokenInCallstack(i);

    printf("%.*s[%d] ", i * 2, "                       ", i);
    thread_trace->printToken(current_sequence_id);

    if (current_sequence_id.type == TypeLoop) {
      auto* loop = thread_trace->getLoop(current_sequence_id);
      printf(" iter %d/%d", callstack_index[i],
             loop->nb_iterations[tokenCount.get_value(current_sequence_id)]);
      pallas_assert(callstack_index[i] < MAX_CALLSTACK_DEPTH);
    } else if (current_sequence_id.type == TypeSequence) {
      auto* sequence = thread_trace->getSequence(current_sequence_id);
      printf(" pos %d/%lu", callstack_index[i], sequence->size());
      pallas_assert(callstack_index[i] < MAX_CALLSTACK_DEPTH);
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
    return current_index + 1 >= loop->nb_iterations[tokenCount.get_value(loop_id)];
    // We are in a loop and index is beyond the number of iterations
  }
  pallas_error("The given loop_id was the wrong type: %d\n", loop_id.type);
}
bool ThreadReader::isEndOfCurrentBlock() const {
  pallas_assert(current_frame >= 0);

  int current_index = callstack_index[current_frame];
  auto current_iterable_token = callstack_iterable[current_frame];

  switch (current_iterable_token.type) {
  case TypeSequence:
    return isEndOfSequence(current_index, current_iterable_token);
  case TypeLoop:
    return isEndOfLoop(current_index, current_iterable_token);
  case TypeEvent:
    return false;
  case TypeInvalid:
    pallas_error("Current frame is invalid");
  }
  return false;
}
bool ThreadReader::isEndOfTrace() const {
  pallas_assert(current_frame >= 0);

  int current_index = callstack_index[current_frame];
  auto current_iterable_token = callstack_iterable[current_frame];
  pallas_assert(current_iterable_token.isIterable());

  if (current_frame > 0 || current_iterable_token.type != TypeSequence) {
    return false;
  }
  return isEndOfSequence(current_index, current_iterable_token);
}

pallas_duration_t ThreadReader::getLoopDuration(Token loop_id) const {
  pallas_assert(loop_id.type == TypeLoop);
  pallas_duration_t sum = 0;
  const auto* loop = thread_trace->getLoop(loop_id);
  const auto* sequence = thread_trace->getSequence(loop->repeated_token);

  const Token sequence_id = loop->repeated_token;

  const size_t loopIndex = tokenCount.get_value(loop_id);
  const size_t offset = tokenCount.get_value(sequence_id);
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

  if ((options & ThreadReaderOptions::NoTimestamps) == 0) {
    eventOccurence.timestamp = referential_timestamp;
    eventOccurence.duration = es->durations->at(occurence_id);
  }
  eventOccurence.attributes = getEventAttributeList(event_id, occurence_id);
  return eventOccurence;
}

SequenceOccurence ThreadReader::getSequenceOccurence(Token sequence_id,
                                                     size_t occurence_id,
                                                     bool save_checkpoint) const {
  auto sequenceOccurence = SequenceOccurence();
  sequenceOccurence.sequence = thread_trace->getSequence(sequence_id);

  if ((options & ThreadReaderOptions::NoTimestamps) == 0) {
    sequenceOccurence.timestamp = referential_timestamp;
    sequenceOccurence.duration = sequenceOccurence.sequence->durations->at(occurence_id);
  }
  sequenceOccurence.full_sequence = nullptr;

  if (save_checkpoint)
    sequenceOccurence.checkpoint = new Checkpoint(this);

  //  auto localTokenCount = sequenceOccurence.sequence->getTokenCount(thread_trace, &this->tokenCount);
  return sequenceOccurence;
};

LoopOccurence ThreadReader::getLoopOccurence(Token loop_id, size_t occurence_id) const {
  auto loopOccurence = LoopOccurence();
  loopOccurence.loop = thread_trace->getLoop(loop_id);
  loopOccurence.nb_iterations = loopOccurence.loop->nb_iterations[occurence_id];
  loopOccurence.full_loop = nullptr;
  if ((options & ThreadReaderOptions::NoTimestamps) == 0) {
    loopOccurence.timestamp = referential_timestamp;
    loopOccurence.duration = getLoopDuration(loop_id);
  }
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
void ThreadReader::loadCheckpoint(Checkpoint* checkpoint) {
  if ((options & ThreadReaderOptions::NoTimestamps) == 0)
    referential_timestamp = checkpoint->referential_timestamp;
  memcpy(callstack_iterable, checkpoint->callstack_iterable, sizeof(int) * MAX_CALLSTACK_DEPTH);
  memcpy(callstack_index, checkpoint->callstack_index, sizeof(int) * MAX_CALLSTACK_DEPTH);
  current_frame = checkpoint->current_frame;
  tokenCount = checkpoint->tokenCount;
};

//******************* EXPLORATION FUNCTIONS ********************

const Token& ThreadReader::pollCurToken() const {
  return getTokenInCallstack(current_frame);
}

std::optional<Token> ThreadReader::pollNextToken() const {
  if (current_frame < 0)
    return std::nullopt;

  int current_index = callstack_index[current_frame];
  auto current_iterable_token = callstack_iterable[current_frame];
  pallas_assert(current_iterable_token.isIterable());

  /* First update the current loop / sequence. */
  if (current_iterable_token.type == TypeSequence) {
    if (isEndOfSequence(current_index, current_iterable_token)) {
      /* We've reached the end of a sequence. */
      return std::nullopt;
    } else {
      return thread_trace->getToken(current_iterable_token, current_index + 1);
    }
  } else {
    if (isEndOfLoop(current_index, current_iterable_token)) {
      /* We've reached the end of the loop. */
      return std::nullopt;
    } else {
      return thread_trace->getToken(current_iterable_token, current_index + 1);
    }
  }
}

std::optional<Token> ThreadReader::pollPrevToken() const {
  if (current_frame < 0)
    return std::nullopt;

  int current_index = callstack_index[current_frame];
  auto current_iterable_token = callstack_iterable[current_frame];
  pallas_assert(current_iterable_token.isIterable());

  if (current_index == 0) {
    /* We've reached the end of a sequence. */
    return std::nullopt;
  } else {
    return thread_trace->getToken(current_iterable_token, current_index - 1);
  }
}

void ThreadReader::moveToNextToken() {
  // Check if we've reached the end of the trace
  if (current_frame < 0) {
    pallas_log(DebugLevel::Debug, "End of trace %d!\n", __LINE__);
    return;
  }

  int current_index = callstack_index[current_frame];
  auto current_iterable_token = callstack_iterable[current_frame];
  pallas_assert(current_iterable_token.isIterable());

  /* First update the current loop / sequence. */
  if (current_iterable_token.type == TypeSequence) {
    if (isEndOfSequence(current_index, current_iterable_token)) {
      pallas_error("End of sequence");
    } else {
      /* Move to the next event in the Sequence */
      auto current_token = this->pollCurToken();
      pallas_duration_t token_duration = 0;
      switch (current_token.type) {
      case TypeEvent:
        token_duration = getEventSummary(current_token)->durations->at(tokenCount[current_token]);
        break;

      case TypeLoop:
        token_duration = getLoopDuration(current_token);
        for (int i = 0; i < thread_trace->getLoop(current_token)->nb_iterations[tokenCount[current_token]]; i++) {
          tokenCount +=  thread_trace->getSequence(thread_trace->getLoop(current_token)->repeated_token)->getTokenCount(thread_trace);
          tokenCount[thread_trace->getLoop(current_token)->repeated_token]++;
        }
        break;

      case TypeSequence:
        token_duration = thread_trace->getSequence(current_token)->durations->at(tokenCount[current_token]);
        tokenCount += thread_trace->getSequence(current_token)->getTokenCount(thread_trace);
        break;

      case TypeInvalid:
        pallas_error("Token is Invalid");
      }
#ifdef DEBUG
      if (referential_timestamp + token_duration < referential_timestamp) {
        pallas_error("Token duration negative for (%c.%d): %lu\n", PALLAS_TOKEN_TYPE_C(current_token), current_token.id, token_duration);
      }
#endif
      referential_timestamp+=token_duration;
      tokenCount[current_token]++;
      callstack_index[current_frame]++;
    }
  } else {
    if (isEndOfLoop(current_index, current_iterable_token)) {
      pallas_error("End of loop");
    } else {
      /* just move to the next iteration in the loop */
      auto current_token = this->pollCurToken();
      pallas_duration_t token_duration = 0;
      switch (current_token.type) {
      case TypeEvent:
        token_duration = getEventSummary(current_token)->durations->at(tokenCount[current_token]);
        break;

      case TypeLoop:
        for (int i = 0; i < thread_trace->getLoop(current_token)->nb_iterations[tokenCount[current_token]]; i++)
          tokenCount += thread_trace->getSequence(
            thread_trace->getLoop(current_token)->repeated_token
            )->getTokenCount(thread_trace);
        token_duration = getLoopDuration(current_token);
        break;

      case TypeSequence:
        tokenCount += thread_trace->getSequence(current_token)->getTokenCount(thread_trace);
        token_duration = thread_trace->getSequence(current_token)->durations->at(tokenCount[current_token]);
        break;

      case TypeInvalid:
        pallas_error("Token is Invalid");
      }
      pallas_assert(referential_timestamp + token_duration > referential_timestamp);
      referential_timestamp+=token_duration;
      tokenCount[current_token]++;
      callstack_index[current_frame]++;
    }
  }
}

void ThreadReader::moveToPrevToken() {
  // Check if we've reached the end of the trace
  if (current_frame < 0) {
    pallas_log(DebugLevel::Debug, "End of trace %d!\n", __LINE__);
    return;
  }

  int current_index = callstack_index[current_frame];
  auto current_iterable_token = callstack_iterable[current_frame];
  pallas_assert(current_iterable_token.isIterable());

  if (current_index <= 0) {
    pallas_error("Beginning of block");
  } else {
    /* Move to the previous event in the Sequence */
    auto previous_token = pollPrevToken().value();
    auto current_token = pollCurToken();
    tokenCount[previous_token]--;
    callstack_index[current_frame]--;
    switch (previous_token.type) {
    case TypeEvent:
      referential_timestamp-=getEventSummary(previous_token)->durations->at(tokenCount[previous_token]);
      break;

    case TypeLoop:
      for (int i = 0; i < thread_trace->getLoop(previous_token)->nb_iterations[tokenCount[previous_token]]; i++) {
        tokenCount -=  thread_trace->getSequence(thread_trace->getLoop(previous_token)->repeated_token)->getTokenCount(thread_trace);
        tokenCount[thread_trace->getLoop(previous_token)->repeated_token]--;
      }
      referential_timestamp-=getLoopDuration(previous_token);
      break;

    case TypeSequence:
      tokenCount -= thread_trace->getSequence(previous_token)->getTokenCount(thread_trace);
      referential_timestamp-=thread_trace->getSequence(previous_token)->durations->at(tokenCount[previous_token]);
      break;

    case TypeInvalid:
      pallas_error("Token is Invalid");
    }
  }
}

std::optional<Token> ThreadReader::getNextToken(const int flags) {
  if (current_frame < 0)
    return std::nullopt;

  pallas_timestamp_t current_timestamp = referential_timestamp;

  auto current_token = pollCurToken();

  /*pallas_timestamp_t expected_timestamp = current_timestamp;
  switch (current_token.type) {
  case TypeEvent:
    expected_timestamp += this->getEventSummary(current_token)->durations->at(tokenCount[current_token]);
    break;
  case TypeSequence:
    expected_timestamp += this->thread_trace->getSequence(current_token)->durations->at(tokenCount[current_token]);
    break;
  case TypeLoop:
    expected_timestamp += this->getLoopDuration(current_token);
    break;
  case TypeInvalid:
    pallas_error("Invalid Token");
    break;
  }*/
  /* Perform callstack actions based on flags and current state*/
  if (current_token.type == TypeSequence && flags & PALLAS_READ_UNROLL_SEQUENCE) {
    enterBlock(current_token);
    return pollCurToken();
  } else if (current_token.type == TypeLoop && flags & PALLAS_READ_UNROLL_LOOP) {
    enterBlock(current_token);
    return pollCurToken();
  } else if (current_frame > 0) {
    bool exited_block;
    do {
      exited_block = exitIfEndOfBlock(flags);
    } while (exited_block);
  }

  const auto next_token = pollNextToken();
  if (next_token.has_value()) {
    moveToNextToken();
  }
  /*if (referential_timestamp != expected_timestamp) {
    std::cerr << "Expected " << expected_timestamp / 1e9 << ", got " << referential_timestamp / 1e9 << " in " << std::endl;
    pallas_error("");
  }*/
  return next_token;
}

void ThreadReader::enterBlock(const Token new_block) {
  pallas_assert(new_block.isIterable());
  if (debugLevel >= DebugLevel::Debug) {
    pallas_log(DebugLevel::Debug, "[%d] Enter Block ", current_frame);
    printCurToken();
    printf("\n");
  }

  callstack_checkpoints[current_frame] = Checkpoint(this);
  current_frame++;
  callstack_index[current_frame] = 0;
  callstack_iterable[current_frame] = new_block;
}

void ThreadReader::leaveBlock() {
  if (debugLevel >= DebugLevel::Debug) {
    pallas_log(DebugLevel::Debug, "[%d] Leave ", current_frame);
    printCurSequence();
    printf("\n");
  }

  pallas_assert(current_frame > 0);
  loadCheckpoint(&callstack_checkpoints[current_frame-1]);

  if (debugLevel >= DebugLevel::Debug && current_frame >= 0) {
    auto current_sequence = getCurIterable();
    pallas_assert(current_sequence.type == TypeLoop || current_sequence.type == TypeSequence);
  }
}

bool ThreadReader::exitIfEndOfBlock(int flags) {
  if (current_frame <= 0)
    return false;
  int current_index = callstack_index[current_frame];
  auto current_iterable_token = callstack_iterable[current_frame];
  if (current_iterable_token.type == TypeSequence) {
    if (isEndOfSequence(current_index, current_iterable_token) && flags & PALLAS_READ_UNROLL_SEQUENCE) {
      /* We've reached the end of a sequence. Leave the block. */
      leaveBlock();
      return true;
    }
  } else {
    if (isEndOfLoop(current_index, current_iterable_token) && flags & PALLAS_READ_UNROLL_LOOP) {
      /* We've reached the end of the loop. Leave the block. */
      leaveBlock();
      return true;
    }
  }
  return false;
}

ThreadReader::~ThreadReader() {
  if (archive)
    archive->freeThread(thread_trace->id);
}

ThreadReader::ThreadReader(ThreadReader&& other) noexcept {
  archive = other.archive;
  thread_trace = other.thread_trace;
  referential_timestamp = other.referential_timestamp;
  std::memcpy(callstack_iterable, other.callstack_iterable, sizeof(Token) *MAX_CALLSTACK_DEPTH);
  std::memcpy(callstack_index, other.callstack_index, sizeof(int) *MAX_CALLSTACK_DEPTH);
  std::memcpy((void*)callstack_checkpoints, other.callstack_checkpoints, sizeof(Checkpoint) *MAX_CALLSTACK_DEPTH);
  current_frame = other.current_frame;
  tokenCount = TokenCountMap(other.tokenCount);
  options = other.options;
  // Set other to 0 for everything
  other.archive = nullptr;
  other.thread_trace = nullptr;
  other.referential_timestamp = 0;
  std::memset(other.callstack_index, 0, sizeof(Token) * MAX_CALLSTACK_DEPTH);
  std::memset((void*)other.callstack_iterable, 0, sizeof(int) * MAX_CALLSTACK_DEPTH);
  other.current_frame = 0;
  other.tokenCount.clear();
  other.options = 0;
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

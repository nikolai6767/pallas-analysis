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
#include <time.h>
#include "pallas/pallas_dbg.h"
#include <cfloat>
#include <fstream>


void duration_init(Duration* d) {
	d->total_d = 0.0;
	d->min_d = DBL_MAX;
	d->max_d = 0.0;
	d-> count = 0;
}

void update_duration(Duration* d, struct timespec t1, struct timespec t2){
	long time = (t2.tv_sec - t1.tv_sec) * 1e9 + (t2.tv_nsec - t1.tv_nsec);
	d->total_d += time;
	if (time < d ->min_d) d->min_d = time;
	if (time > d -> max_d) d-> max_d = time;
	d->count++;
}

void duration_write_csv(const char* filename, const Duration* d) {
    std::ofstream file(std::string(filename) + ".csv");
    file << "name,calls,total_us,min_us,max_us,average_us\n";
    file << std::string(filename) << "," << d->count << "," << d->total_d << "," << d->min_d << "," << d->max_d << ",";
    file << (d->count ? d->total_d / d->count : 0) << "\n";
}

void write_csv_details(const char* filename, const char* output, const char* info, struct timespec t1, struct timespec t2) {
    std::ofstream file(std::string(output) + ".csv", std::ios::app);
    long time = (t2.tv_sec - t1.tv_sec) * 1e9 + (t2.tv_nsec - t1.tv_nsec);
    file << std::string(filename) << "," << time << "," << std::string(info) << "\n";
}

Duration durations[NB_FUNCTIONS] = {};



void duration_write_all_csv(const char* filename) {
  std::ofstream file(std::string(filename) + ".csv");
  file << "function,calls,total,min,max,average\n";

  const char* function_names[NB_FUNCTIONS] = {
//   "PRINT_TIMESTAMP",
//   "PRINT_TIMESTAMP_PRECISION",
//   "PRINT_TIMESTAMP_ELSE",
//   "PRINT_TIMESTAMP_HEADER",
//   "PRINT_DURATION",
//   "PRINT_DURATION_HEADER",
//   "PRINT_EVENT",
//   "PRINT_EVENT_PRINT_TIMESTAMP",
//   "PRINT_EVENT_GET_NAME",
//   "PRINT_EVENT_GET_TOKEN_STRING",
//   "PRINT_EVENT_GET_EVENT_STRING",
//   "PRINT_EVENT_GET_PRINT_EV_ATT",
//   "PRINT_EV_ATT",
//   "PRINT_EVENT_ENDL",
//   "PRINT_FLAME",
//   "PRINT_CSV",
//   "PRINT_CSV_BULK",
//   "PRINT_TRACE",
//   "PRINT_TRACE_GET_THREAD_LIST",
//   "GET_THREAD_LIST_GET_ARCHIVE",
//   "GET_THREAD_LIST_GET_THREAD",
//   "PRINT_TRACE_EMPLACE_BACK",
//   "PRINT_TRACE_POLLCURTOKEN",
//   "PRINT_TRACE_PRINT_EVENT",
//   "PRINT_TRACE_GET_EV_OCC",
//   "PRINT_TRACE_GET_NEXT_TOKEN",
//   "STRUCTURE_INDENT", 
//   "STRUCTURE_INDENT_POLLCURTOKEN",
//   "PRINT_THREAD_STRUCTURE",
//   "PRINT_STRUCTURE", 
//   "GET_EVENT_OCC",
//   "GET_TOKEN_IN_CALL_STACK",
//   "ENTER_BLOCK",
//   "PALLAS_PRINT",
//   "OPEN_TRACE",
  "ZSTD",
  };

  for (int i = 0; i < NB_FUNCTIONS; ++i) {
    const Duration& d = durations[i];
    if (d.count > 0){
      double avg = d.count ? d.total_d / d.count : 0.0;
      file << function_names[i] << "," << d.count << "," << d.total_d << "," << d.min_d << "," << d.max_d << "," << avg << "\n";
    }
  }

}

namespace pallas {

CallstackFrame::CallstackFrame() {
    this->referential_timestamp = 0;
    this->frame_index = 0;
}

CallstackFrame::~CallstackFrame() = default;

Cursor::Cursor(const Cursor& other) {
    current_frame_index = other.current_frame_index;
    DOFOR(i, MAX_CALLSTACK_DEPTH) {
        callstack[i].tokenCount = other.callstack[i].tokenCount;
        callstack[i].frame_index = other.callstack[i].frame_index;
        callstack[i].callstack_iterable = other.callstack[i].callstack_iterable;
        callstack[i].referential_timestamp = other.callstack[i].referential_timestamp;
    }
    currentFrame = &callstack[current_frame_index];
}
Cursor& Cursor::operator=(const Cursor& other) {
    current_frame_index = other.current_frame_index;
    DOFOR(i, MAX_CALLSTACK_DEPTH) {
        callstack[i].tokenCount = other.callstack[i].tokenCount;
        callstack[i].frame_index = other.callstack[i].frame_index;
        callstack[i].callstack_iterable = other.callstack[i].callstack_iterable;
        callstack[i].referential_timestamp = other.callstack[i].referential_timestamp;
    }
    currentFrame = &callstack[current_frame_index];
    return *this;
}

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
    this->currentState.current_frame_index = 0;
    this->currentState.currentFrame = &currentState.callstack[0];
    this->currentState.currentFrame->callstack_iterable = Token(TypeSequence, 0);
    this->currentState.currentFrame->referential_timestamp = this->thread_trace->first_timestamp;
    // Enter main sequence
    enterBlock();
}

const Token& ThreadReader::getFrameInCallstack(int frame_number) const {
    if (frame_number < 0 || frame_number >= MAX_CALLSTACK_DEPTH) {
        pallas_error("Frame number is too high or negative: %d\n", frame_number);
    }
    return currentState.callstack[frame_number].callstack_iterable;
}

const Token& ThreadReader::getTokenInCallstack(int frame_number) const {
    struct timespec t1, t2;
    clock_gettime(CLOCK_MONOTONIC, &t1);
    if (frame_number < 0 || frame_number >= MAX_CALLSTACK_DEPTH) {
        pallas_error("Frame number is too high or negative: %d\n", frame_number);
    }
    auto sequence = getFrameInCallstack(frame_number);
    pallas_assert(sequence.isIterable());
    const Token&  retval =  thread_trace->getToken(sequence, currentState.callstack[frame_number].frame_index);
    clock_gettime(CLOCK_MONOTONIC, &t2);
    update_duration(&durations[GET_TOKEN_IN_CALL_STACK], t1, t2);
    return retval; 
}
void ThreadReader::printCurToken() const {
    std::cout << thread_trace->getTokenString(pollCurToken()) << std::endl;
}
const Token& ThreadReader::getCurIterable() const {
    return currentState.currentFrame->callstack_iterable;
}
void ThreadReader::printCurSequence() const {
    thread_trace->printSequence(getCurIterable());
}

void ThreadReader::printCallstack() const {
    printf("# Callstack (depth: %d) ------------\n", currentState.current_frame_index + 1);
    for (int i = 0; i < currentState.current_frame_index + 1; i++) {
        auto current_sequence_id = getFrameInCallstack(i);
        auto current_token = getTokenInCallstack(i);

        printf("%.*s[%d] ", i * 2, "                       ", i);
        std::cout << thread_trace->getTokenString(current_sequence_id) << std::endl;

        if (current_sequence_id.type == TypeLoop) {
            auto* loop = thread_trace->getLoop(current_sequence_id);
            printf(" iter %d/%d", currentState.callstack[i].frame_index, loop->nb_iterations);
            pallas_assert(currentState.callstack[i].frame_index < MAX_CALLSTACK_DEPTH);
        } else if (current_sequence_id.type == TypeSequence) {
            auto* sequence = thread_trace->getSequence(current_sequence_id);
            printf(" pos %d/%lu", currentState.callstack[i].frame_index, sequence->size());
            pallas_assert(currentState.callstack[i].frame_index < MAX_CALLSTACK_DEPTH);
        }

        std::cout << "\t-> " << thread_trace->getTokenString(current_token) << std::endl;
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
        return summary->timestamps->at(occurence_id);
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
        pallas_assert(current_index < loop->nb_iterations);
        return current_index + 1 >= loop->nb_iterations;
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
    pallas_assert(currentState.current_frame_index >= 0);

    int current_index = currentState.currentFrame->frame_index;
    auto current_iterable_token = currentState.currentFrame->callstack_iterable;

    return isEndOfBlock(current_index, current_iterable_token);
}
bool ThreadReader::isEndOfTrace() const {
    return currentState.current_frame_index == 0;
}

pallas_duration_t ThreadReader::getLoopDuration(Token loop_id) const {
    pallas_assert(loop_id.type == TypeLoop);
    const auto* loop = thread_trace->getLoop(loop_id);
    const auto* sequence = thread_trace->getSequence(loop->repeated_token);

    const Token sequence_id = loop->repeated_token;

    size_t offset;
    if (getCurIterable() != loop_id)
        offset = currentState.currentFrame->tokenCount.get_value(sequence_id);
    else
        offset = currentState.callstack[currentState.current_frame_index - 1].tokenCount.get_value(sequence_id);
    const size_t nIterations = loop->nb_iterations;
    return sequence->timestamps->at(offset + nIterations - 1) - sequence->timestamps->at(offset) + sequence->durations->at(offset + nIterations - 1);
}

EventOccurence ThreadReader::getEventOccurence(Token event_id, size_t occurence_id) const {
    auto eventOccurence = EventOccurence();
    auto* es = getEventSummary(event_id);
    eventOccurence.event = thread_trace->getEvent(event_id);

    eventOccurence.timestamp = es->timestamps->at(occurence_id);
    eventOccurence.attributes = getEventAttributeList(event_id, occurence_id);
    return eventOccurence;
}

SequenceOccurence ThreadReader::getSequenceOccurence(Token sequence_id, size_t occurence_id) const {
    auto sequenceOccurence = SequenceOccurence();
    sequenceOccurence.sequence = thread_trace->getSequence(sequence_id);

    sequenceOccurence.timestamp = sequenceOccurence.sequence->timestamps->at(occurence_id);
    sequenceOccurence.duration = sequenceOccurence.sequence->durations->at(occurence_id);
    sequenceOccurence.full_sequence = nullptr;

    return sequenceOccurence;
};

LoopOccurence ThreadReader::getLoopOccurence(Token loop_id, size_t occurence_id) const {
    auto loopOccurence = LoopOccurence();
    loopOccurence.loop = thread_trace->getLoop(loop_id);
    loopOccurence.nb_iterations = loopOccurence.loop->nb_iterations;
    loopOccurence.full_loop = nullptr;
    loopOccurence.timestamp = currentState.currentFrame->referential_timestamp;
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

void ThreadReader::guessSequencesNames(std::map<pallas::Sequence*, std::string>& names) const {
    // Let's call the main sequence "main"
    names[thread_trace->sequences[0]] = "main";

    for (int i = 1; i < thread_trace->nb_sequences; i++) {
        pallas::Sequence* s = thread_trace->sequences[i];

        if (names.count(s) == 0) {
            // The sequence is not named yet

            bool name_found = false;
            if (s->size() <= 4) {
                // for small (enter/leave function) sequence, use the name of the function
                pallas::Token t_start = s->tokens[0];
                if (t_start.type == pallas::TypeEvent) {
                    pallas::Event* event = thread_trace->getEvent(t_start);
                    if (event->record == pallas::PALLAS_EVENT_ENTER) {
                        const char* event_name = thread_trace->getRegionStringFromEvent(event);
                        // TODO if that's an MPI call (eg MPI_Send, MPI_Allreduce, ...)
                        //      we may want to get the function parameters (eg. dest, tag, ...)
                        names[s] = std::string(event_name);
                        name_found = true;
                    } else if (event->record == pallas::PALLAS_EVENT_THREAD_BEGIN) {
                        names[s] = "main";
                        name_found = true;
                    }
                }
            }

            if (!name_found) {
                // is it a loop ?
                for (int j = 0; j < thread_trace->nb_loops; j++) {
                    pallas::Loop& l = thread_trace->loops[j];
                    if (thread_trace->getSequence(l.repeated_token) == s) {
                        char buff[128];
                        snprintf(buff, sizeof(buff), "Loop_%d", l.self_id.id);
                        names[s] = std::string(buff);
                        name_found = true;
                        break;
                    }
                }
            }

            if (!name_found) {
                // probably a complex/long sequence. Just name it randomly
                char buff[128];
                snprintf(buff, sizeof(buff), "Sequence_%d", s->id);
                names[s] = std::string(buff);
            }
        }
    }
}

//******************* EXPLORATION FUNCTIONS ********************

const Token& ThreadReader::pollCurToken() const {
    return getTokenInCallstack(currentState.current_frame_index);
}

Token ThreadReader::pollNextToken(int flags) const {
    if (currentState.current_frame_index < 0)
        // return an invalid token
        return Token();

    if (flags == PALLAS_READ_FLAG_NONE)
        flags = pallas_read_flag;

    int current_frame = currentState.current_frame_index;
    int current_index = currentState.currentFrame->frame_index;
    auto current_iterable_token = currentState.currentFrame->callstack_iterable;
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
            current_index = currentState.currentFrame->frame_index;
            current_iterable_token = currentState.currentFrame->callstack_iterable;
        } else if (current_iterable_token.type == TypeLoop && flags & PALLAS_READ_FLAG_UNROLL_LOOP) {
            current_frame--;
            current_index = currentState.currentFrame->frame_index;
            current_iterable_token = currentState.currentFrame->callstack_iterable;
        } else {
            return Token();
        }
    }
    return thread_trace->getToken(current_iterable_token, current_index + 1);
}

Token ThreadReader::pollPrevToken(int flags) const {
    if (currentState.current_frame_index < 0)
        return Token();

    if (flags == PALLAS_READ_FLAG_NONE)
        flags = pallas_read_flag;

    int current_frame = currentState.current_frame_index;
    int current_index = currentState.currentFrame->frame_index;
    auto current_iterable_token = currentState.currentFrame->callstack_iterable;
    pallas_assert(current_iterable_token.isIterable());

    while (current_index == 0) {
        if (current_frame == 0)
            return Token();
        if (current_iterable_token.type == TypeSequence && flags & PALLAS_READ_FLAG_UNROLL_SEQUENCE) {
            current_frame--;
            current_index = currentState.currentFrame->frame_index;
            current_iterable_token = currentState.currentFrame->callstack_iterable;
        } else if (current_iterable_token.type == TypeLoop && flags & PALLAS_READ_FLAG_UNROLL_LOOP) {
            current_frame--;
            current_index = currentState.currentFrame->frame_index;
            current_iterable_token = currentState.currentFrame->callstack_iterable;
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
    if (currentState.current_frame_index < 0) {
        pallas_log(DebugLevel::Debug, "End of trace %d!\n", __LINE__);
        return false;
    }

    if (flags == PALLAS_READ_FLAG_NONE)
        flags = pallas_read_flag;

    pallas_assert(getCurIterable().isIterable());

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
    switch (current_token.type) {
    case TypeEvent:
        currentState.currentFrame->referential_timestamp = getEventSummary(current_token)->timestamps->at(currentState.currentFrame->tokenCount[current_token]);
        break;

    case TypeLoop: {
        auto loop = thread_trace->getLoop(current_token);
        auto loop_sequence = thread_trace->getSequence(loop->repeated_token);
        currentState.currentFrame->referential_timestamp = loop_sequence->timestamps->at(currentState.currentFrame->tokenCount[loop->repeated_token]);
        auto loopCount = loop->nb_iterations;
        for (size_t i = 0; i < loopCount; i++) {
            currentState.currentFrame->tokenCount += loop_sequence->getTokenCountReading(thread_trace, currentState.currentFrame->tokenCount);
            currentState.currentFrame->tokenCount[loop->repeated_token] += 1;
        }
        break;
    }

    case TypeSequence: {
        auto seq = thread_trace->getSequence(current_token);
        currentState.currentFrame->referential_timestamp = seq->timestamps->at(currentState.currentFrame->tokenCount[current_token]);
        currentState.currentFrame->tokenCount += seq->getTokenCountReading(thread_trace, currentState.currentFrame->tokenCount);
        break;
    }

    case TypeInvalid:
        pallas_error("Token is Invalid");
    }
    currentState.currentFrame->tokenCount[current_token]++;
    currentState.currentFrame->frame_index++;
    return true;
}

bool ThreadReader::moveToNextTokenInBlock() {
    return moveToNextToken(PALLAS_READ_FLAG_NO_UNROLL);
}

bool ThreadReader::moveToPrevToken(int flags) {
    // Check if we've reached the end of the trace
    if (currentState.current_frame_index < 0) {
        pallas_log(DebugLevel::Debug, "End of trace %d!\n", __LINE__);
        return false;
    }

    if (flags == PALLAS_READ_FLAG_NONE)
        flags = pallas_read_flag;

    pallas_assert(currentState.currentFrame->callstack_iterable.isIterable());

    if (currentState.currentFrame->frame_index == 0) {
        if (currentState.current_frame_index <= 1)
            return false;
        Token current_iterable_token = currentState.currentFrame->callstack_iterable;
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
            ntokens = thread_trace->getLoop(previous_token)->nb_iterations;
        } else {
            break;
        }
        // Save cursor before entering
        // TODO : Optimiser ça
        moveToPrevToken(PALLAS_READ_FLAG_NO_UNROLL);
        if (!moveToNextToken(PALLAS_READ_FLAG_NO_UNROLL)) {  // If we didn't come back, we come back manually
            Token current_token = pollCurToken();
            pallas_duration_t token_duration = 0;
            switch (current_token.type) {
            case TypeEvent:
                token_duration = getEventSummary(current_token)->timestamps->at(currentState.currentFrame->tokenCount[current_token]);
                break;

            case TypeLoop:
                token_duration = getLoopDuration(current_token);
                for (int i = 0; i < thread_trace->getLoop(current_token)->nb_iterations; i++) {
                    currentState.currentFrame->tokenCount += thread_trace->getSequence(thread_trace->getLoop(current_token)->repeated_token)
                                                               ->getTokenCountReading(thread_trace, currentState.currentFrame->tokenCount, true);
                    currentState.currentFrame->tokenCount[thread_trace->getLoop(current_token)->repeated_token]++;
                }
                break;

            case TypeSequence:
                token_duration = thread_trace->getSequence(current_token)->durations->at(currentState.currentFrame->tokenCount[current_token]);
                currentState.currentFrame->tokenCount += thread_trace->getSequence(current_token)->getTokenCountReading(thread_trace, currentState.currentFrame->tokenCount, true);
                break;

            case TypeInvalid:
                pallas_error("Token is Invalid");
            }

            currentState.currentFrame->referential_timestamp += token_duration;
            currentState.currentFrame->tokenCount[current_token]++;
            currentState.currentFrame->frame_index++;
        }
        currentState.currentFrame->tokenCount[previous_token]--;
        // Enter block
        currentState.current_frame_index++;
        currentState.currentFrame++;
        currentState.currentFrame->callstack_iterable = previous_token;
        currentState.currentFrame->frame_index = ntokens;
        previous_token = pollPrevToken(PALLAS_READ_FLAG_NO_UNROLL);
    }

    currentState.currentFrame->tokenCount[previous_token]--;
    currentState.currentFrame->frame_index--;

    switch (previous_token.type) {
    case TypeEvent:
        currentState.currentFrame->referential_timestamp -= getEventSummary(previous_token)->timestamps->at(currentState.currentFrame->tokenCount[previous_token]);
        break;

    case TypeLoop: {
        auto prev_loop = thread_trace->getLoop(previous_token);
        for (int i = 0; i < prev_loop->nb_iterations; i++) {
            currentState.currentFrame->tokenCount -=
              thread_trace->getSequence(prev_loop->repeated_token)->getTokenCountReading(thread_trace, currentState.currentFrame->tokenCount, true);
            currentState.currentFrame->tokenCount[prev_loop->repeated_token]--;
        }
        currentState.currentFrame->referential_timestamp -= getLoopDuration(previous_token);
        break;
    }
    case TypeSequence: {
        auto prev_sequence = thread_trace->getSequence(previous_token);
        currentState.currentFrame->tokenCount -= prev_sequence->getTokenCountReading(thread_trace, currentState.currentFrame->tokenCount, true);
        currentState.currentFrame->referential_timestamp -= prev_sequence->durations->at(currentState.currentFrame->tokenCount[previous_token]);
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
        pallas_log(DebugLevel::Debug, "[%d] Enter Block ", currentState.current_frame_index);
        printCurToken();
        printf("\n");
    }

    currentState.current_frame_index++;
    currentState.currentFrame++;
    currentState.currentFrame->frame_index = 0;
    struct timespec t1, t2;
    clock_gettime(CLOCK_MONOTONIC, &t1);
    currentState.currentFrame->referential_timestamp = currentState.callstack[currentState.current_frame_index - 1].referential_timestamp;
    clock_gettime(CLOCK_MONOTONIC, &t2);
    update_duration(&durations[ENTER_BLOCK], t1, t2);
    currentState.currentFrame->callstack_iterable = new_block;
    currentState.currentFrame->tokenCount = (currentState.currentFrame - 1)->tokenCount;
#ifdef DEBUG
    if (new_block.type == TypeSequence) {
        auto current_timestamp = currentState.currentFrame->referential_timestamp;
        auto seq = thread_trace->getSequence(new_block);
        auto theorical_timestamp = seq->timestamps->at(currentState.currentFrame->tokenCount[new_block]);
        if (theorical_timestamp != current_timestamp) {
            int a = 1;
        }
        // pallas_assert(theorical_timestamp == current_timestamp);
    }
#endif
}

void ThreadReader::leaveBlock() {
    if (debugLevel >= DebugLevel::Debug) {
        pallas_log(DebugLevel::Debug, "[%d] Leave \n", currentState.current_frame_index);
    }

    pallas_assert(currentState.current_frame_index > 0);

    currentState.current_frame_index--;
    currentState.currentFrame--;

    if (debugLevel >= DebugLevel::Debug && currentState.current_frame_index >= 0) {
        pallas_assert(getCurIterable().isIterable());
    }
}

bool ThreadReader::exitIfEndOfBlock(int flags) {
    if (flags == PALLAS_READ_FLAG_NONE)
        flags = pallas_read_flag;

    if (currentState.current_frame_index == 0)
        return false;
    int current_index = currentState.currentFrame->frame_index;
    auto current_iterable_token = currentState.currentFrame->callstack_iterable;
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
        enterBlock();
        return true;
    }
    if (current_token.type == TypeLoop && flags & PALLAS_READ_FLAG_UNROLL_LOOP) {
        enterBlock();
        return true;
    }
    return false;
}

Cursor ThreadReader::createCheckpoint() const {
    return Cursor(this->currentState);
}
void ThreadReader::loadCheckpoint(Cursor* checkpoint) {
    currentState = *checkpoint;
}

ThreadReader::~ThreadReader() {
    if (archive)
        archive->freeThread(thread_trace->id);
}

ThreadReader::ThreadReader(const ThreadReader& other) = default;

ThreadReader::ThreadReader(ThreadReader&& other) noexcept {
    archive = other.archive;
    thread_trace = other.thread_trace;
    currentState = other.currentState;
    pallas_read_flag = other.pallas_read_flag;
    // Set other to 0 for everything
    other.archive = nullptr;
    other.thread_trace = nullptr;
    other.currentState = Cursor();
    other.pallas_read_flag = 0;
}

ThreadReader& ThreadReader::operator=(const ThreadReader& other) {
    archive = other.archive;
    thread_trace = other.thread_trace;
    currentState = other.currentState;
    pallas_read_flag = other.pallas_read_flag;
    return *this;
}

ThreadReader& ThreadReader::operator=(ThreadReader&& other) noexcept {
    archive = other.archive;
    thread_trace = other.thread_trace;
    currentState = other.currentState;
    pallas_read_flag = other.pallas_read_flag;
    // Set other to 0 for everything
    other.archive = nullptr;
    other.thread_trace = nullptr;
    other.currentState = Cursor();
    other.pallas_read_flag = 0;
    return *this;
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
SequenceOccurence pallasGetSequenceOccurence(ThreadReader* thread_reader, Token sequence_id, size_t occurence_id) {
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
Cursor pallasCreateCheckpoint(ThreadReader* thread_reader) {
    return thread_reader->createCheckpoint();
}
void pallasLoadCheckpoint(ThreadReader* thread_reader, Cursor* checkpoint) {
    thread_reader->loadCheckpoint(checkpoint);
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

/*
 * Copyright (C) Telecom SudParis
 * See LICENSE in top-level directory.
 */
/** @file
 * Everything needed to read a trace.
 */
#pragma once

#include "pallas.h"
#include "pallas_attribute.h"
#include "pallas_timestamp.h"

namespace pallas {
/** Maximum Callstack Size. */
#define MAX_CALLSTACK_DEPTH 100

/** getNextToken flags */
#define PALLAS_READ_NO_UNROLL 0
#define PALLAS_READ_UNROLL_SEQUENCE 1
#define PALLAS_READ_UNROLL_LOOP 2
#define PALLAS_READ_UNROLL_ALL 3


/**
 * Options for the ThreadReader. Values have to be powers of 2.
 */
enum ThreadReaderOptions {
  None = 0,          /**< No specific options. */
  ShowStructure = 1, /**< Read and return the structure (Sequences and Loop). */
  NoTimestamps = 2,  /**< Do not load the timestamps / durations. */
};

/** Represents one occurence of an Event. */
typedef struct EventOccurence {
  struct Event* event;          /**< Pointer to the Event.*/
  pallas_timestamp_t timestamp; /**< Timestamp for that occurence.*/
  pallas_duration_t duration;   /**< Duration of that occurence.*/
  AttributeList* attributes;    /**< Attributes for that occurence.*/
} EventOccurence;

/**
 * Represents one occurence of a Sequence.
 */
typedef struct SequenceOccurence {
  struct Sequence* sequence;            /**< Pointer to the Sequence.*/
  pallas_timestamp_t timestamp;         /**< Timestamp for that occurence.*/
  pallas_duration_t duration;           /**< Duration of that occurence.*/
  struct TokenOccurence* full_sequence; /** Array of the occurrences in this sequence. */
} SequenceOccurence;

/**
 * Represents one occurence of a Loop.
 */
typedef struct LoopOccurence {
  struct Loop* loop;                     /**< Pointer to the Loop.*/
  unsigned int nb_iterations;            /**< Number of iterations for that occurence.*/
  pallas_timestamp_t timestamp;          /**< Timestamp for that occurence.*/
  pallas_duration_t duration;            /**< Duration for that occurence.*/
  struct SequenceOccurence* full_loop;   /**< Array of the Sequences in this loop.*/
  struct SequenceOccurence loop_summary; /**< False SequenceOccurence that represents a summary of all the
                                          * occurrences in full_loop. */
} LoopOccurence;

/**
 * Represents any kind of Occurrence.
 */
typedef union Occurence {
  struct LoopOccurence loop_occurence;         /**< Occurence for a Loop.*/
  struct SequenceOccurence sequence_occurence; /**< Occurence for a Sequence.*/
  struct EventOccurence event_occurence;       /**< Occurence for an Event.*/
} Occurence;

/**
 * Tuple containing a Token and its corresponding Occurence.
 */
typedef struct TokenOccurence {
  /** Token for the occurence. */
  const Token* token;
  /** Occurence corresponding to the Token. */
  Occurence* occurence;

  ~TokenOccurence();
} TokenOccurence;


/**
 * Reads one thread from an Pallas trace.
 */
typedef struct ThreadReader {
  /** Archive being read by this reader. */
  const struct Archive* archive;
  /** Thread being read. */
  struct Thread* thread_trace;
  /** The current referential timestamp. */
  pallas_timestamp_t referential_timestamp;

  /** Stack containing the sequences/loops being read. */
  Token callstack_iterable[MAX_CALLSTACK_DEPTH];

  /** Stack containing the index in the sequence or the loop iteration. */
  int callstack_index[MAX_CALLSTACK_DEPTH];

  /** Current frame = index of the event/loop being read in the callstacks.
   * You can view this as the "depth" of the callstack. */
  int current_frame;

  /** At any point, a token t has been seen tokenCount[t] times. */
  DEFINE_TokenCountMap(tokenCount);

  /**
   * Options as defined in pallas::ThreadReaderOptions.
   */
  int options;

  /**
   * Make a new ThreadReader from an Archive and a threadId.
   * @param archive Archive to read.
   * @param threadId Id of the thread to read.
   * @param options Options as defined in ThreadReaderOptions.
   */
  ThreadReader(const Archive* archive, ThreadId threadId, int options);

 private:
  /** Returns the Sequence being run at the given frame. */
  [[nodiscard]] const Token& getFrameInCallstack(int frame_number) const;
  /** Returns the token being run at the given frame. */
  [[nodiscard]] const Token& getTokenInCallstack(int frame_number) const;
  /** Prints the current Token. */
  void printCurToken() const;
  /** Gets the current Iterable. */
  [[nodiscard]] const Token& getCurIterable() const;
  /** Prints the current Sequence. */
  void printCurSequence() const;
  /** Prints the whole current callstack. */
  void printCallstack() const;
  /** Returns the EventSummary of the given Event. */
  [[nodiscard]] EventSummary* getEventSummary(Token event) const;
  /** Returns the timestamp of the given event occurring at the given index. */
  [[nodiscard]] pallas_timestamp_t getEventTimestamp(Token event, int occurence_id) const;
  /** Returns whether the given sequence still has more Tokens after the given current_index. */
  [[nodiscard]] bool isEndOfSequence(int current_index, Token sequence_id) const;
  /** Returns whether the given loop still has more Tokens after the given current_index. */
  [[nodiscard]] bool isEndOfLoop(int current_index, Token loop_id) const;
  /** Returns the duration of the given Loop. */
  [[nodiscard]] pallas_duration_t getLoopDuration(Token loop_id) const;

  /** Returns an EventOccurence for the given Token appearing at the given occurence_id.
   * Timestamp is set to Reader's referential timestamp.*/
  [[nodiscard]] EventOccurence getEventOccurence(Token event_id, size_t occurence_id) const;
  /** Returns an SequenceOccurence for the given Token appearing at the given occurence_id.
   * Timestamp is set to Reader's referential timestamp.*/
  [[nodiscard]] SequenceOccurence getSequenceOccurence(Token sequence_id,
                                                       size_t occurence_id) const;
  /** Returns an LoopOccurence for the given Token appearing at the given occurence_id.
   * Timestamp is set to Reader's referential timestamp.*/
  [[nodiscard]] LoopOccurence getLoopOccurence(Token loop_id, size_t occurence_id) const;

  /** Returns a pointer to the AttributeList for the given occurence of the given Event. */
  [[nodiscard]] AttributeList* getEventAttributeList(Token event_id, size_t occurence_id) const;


 public:
  /** Gets the current Token. */
  [[nodiscard]] const Token& pollCurToken() const;
  /** Peeks at and return the next token without actually updating the state */
  [[nodiscard]] std::optional<Token> pollNextToken() const;
  /** Updates the internal state */
  void moveToNextToken();
  /** Gets the next token and updates the reader's state if it returns a value.
   * It is more or less equivalent to `moveToNextToken()` then `pollCurToken()` */
  [[nodiscard]] std::optional<Token> getNextToken(int flags);
  /** Enters a block */
  void enterBlock(Token new_block);
  /** Leaves the current block */
  void leaveBlock();

  ~ThreadReader();
  ThreadReader(ThreadReader&& other) noexcept ;
} ThreadReader;

}; /* namespace pallas */

/* -*-
   mode: c;
   c-file-style: "k&r";
   c-basic-offset 2;
   tab-width 2 ;
   indent-tabs-mode nil
   -*- */

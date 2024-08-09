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

#ifdef __cplusplus
#include <optional>

namespace pallas {
#endif

/** Maximum Callstack Size. */
#define MAX_CALLSTACK_DEPTH 100

/** getNextToken flags */
#define PALLAS_READ_FLAG_NONE            0
#define PALLAS_READ_FLAG_NO_UNROLL       (1 << 0)
#define PALLAS_READ_FLAG_UNROLL_SEQUENCE (1 << 2)
#define PALLAS_READ_FLAG_UNROLL_LOOP     (1 << 3)
#define PALLAS_READ_FLAG_UNROLL_ALL (PALLAS_READ_FLAG_UNROLL_SEQUENCE|PALLAS_READ_FLAG_UNROLL_LOOP)


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
  struct Cursor *checkpoint;
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

typedef struct Cursor {
  /** The current referential timestamp. */
  pallas_timestamp_t referential_timestamp;

  /** Stack containing the sequences/loops being read. */
  Token callstack_iterable[MAX_CALLSTACK_DEPTH];

  /** Stack containing the index in the sequence or the loop iteration. */
  int callstack_index[MAX_CALLSTACK_DEPTH];

  /** Current frame = index of the event/loop being read in the callstacks.
   * You can view this as the "depth" of the callstack. */
  int current_frame;

  DEFINE_TokenCountMap(tokenCount);
#ifdef __cplusplus
  /** Creates a savestate of the given reader.
   * @param reader Reader whose state of reading we want to take a screenshot. */
  Cursor(const struct ThreadReader* reader);
  Cursor() = default;
  ~Cursor();
#endif
} Cursor;

/**
 * Reads one thread from an Pallas trace.
 */
typedef struct ThreadReader {
  /** Archive being read by this reader. */
  struct Archive* archive;
  /** Thread being read. */
  struct Thread* thread_trace;

  Cursor currentState;

  /** Stack containing the checkpoint in the sequence or the loop iteration. */
  Cursor callstack_cursors[MAX_CALLSTACK_DEPTH];

  /**
   * Options as defined in pallas::ThreadReaderOptions.
   */
  int pallas_read_flag;
#ifdef __cplusplus
  /**
   * Make a new ThreadReader from an Archive and a threadId.
   * @param archive Archive to read.
   * @param threadId Id of the thread to read.
   * @param options Options as defined in ThreadReaderOptions.
   */
  ThreadReader(Archive* archive, ThreadId threadId, int pallas_read_flag);

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
  /** Returns whether the given iterable token still has more Tokens after the given current_index. */
  [[nodiscard]] bool isEndOfBlock(int index, Token iterable_token) const;
  /** Returns whether the cursor is at the end of the current block. */
  [[nodiscard]] bool isEndOfCurrentBlock() const;
  /** Returns whether the cursor is at the end of the trace. */
  [[nodiscard]] bool isEndOfTrace() const;
  /** Returns the duration of the given Loop. */
  [[nodiscard]] pallas_duration_t getLoopDuration(Token loop_id) const;

  /** Returns an EventOccurence for the given Token appearing at the given occurence_id.
   * Timestamp is set to Reader's referential timestamp.*/
  [[nodiscard]] EventOccurence getEventOccurence(Token event_id, size_t occurence_id) const;
  /** Returns an SequenceOccurence for the given Token appearing at the given occurence_id.
   * Timestamp is set to Reader's referential timestamp.*/
  [[nodiscard]] SequenceOccurence getSequenceOccurence(Token sequence_id,
                                                       size_t occurence_id,
                                                       bool create_checkpoint = false) const;
  /** Returns an LoopOccurence for the given Token appearing at the given occurence_id.
   * Timestamp is set to Reader's referential timestamp.*/
  [[nodiscard]] LoopOccurence getLoopOccurence(Token loop_id, size_t occurence_id) const;

  /** Returns a pointer to the AttributeList for the given occurence of the given Event. */
  [[nodiscard]] AttributeList* getEventAttributeList(Token event_id, size_t occurence_id) const;

  void loadCheckpoint(Cursor *checkpoint);

  //******************* EXPLORATION FUNCTIONS ********************

  /** Gets the current Token. */
  [[nodiscard]] const Token& pollCurToken() const;

  /** Peeks at and return the next token without actually updating the state */
  [[nodiscard]] std::optional<Token> pollNextToken(int flags=PALLAS_READ_FLAG_NONE) const;
  /** Updates the internal state */
  bool moveToNextToken(int flags = PALLAS_READ_FLAG_NONE);
  /** Equivalent to moveToNextToken(PALLAS_READ_FLAG_NO_UNROLL) */
  void moveToNextTokenInBlock();
  /** Gets the next token and updates the reader's state if it returns a value.
   * It is exactly equivalent to `moveToNextToken()` then `pollCurToken()` */
  std::optional<Token> getNextToken(int flags = PALLAS_READ_FLAG_NONE);

  /** Peeks at and return the previous token without actually updating the state */
  [[nodiscard]] std::optional<Token> pollPrevToken(int flags=PALLAS_READ_FLAG_NONE) const;
  /** Updates the internal state */
  bool moveToPrevToken(int flags = PALLAS_READ_FLAG_NONE);
  /** Updates the internal state */
  void moveToPrevTokenInBlock();
  /** Gets the previous token and updates the reader's state if it returns a value.
   * It is exactly equivalent to `moveToPrevToken()` then `pollCurToken()` */
  std::optional<Token> getPrevToken(int flags = PALLAS_READ_FLAG_NONE);

  /** Enters a block */
  void enterBlock();
  /** Leaves the current block */
  void leaveBlock();
  /** Exits a block if at the end of it and flags allow it, returns a boolean representing if the rader actually exited a block */
  bool exitIfEndOfBlock(int flags = PALLAS_READ_FLAG_UNROLL_ALL);
  /** Enter a block if the current token starts a block, returns a boolean representing if the rader actually entered a block */
  bool enterIfStartOfBlock(int flags = PALLAS_READ_FLAG_UNROLL_ALL);

  ~ThreadReader();

  ThreadReader(const ThreadReader &) = default;
  ThreadReader(ThreadReader&& other) noexcept ;
#endif
} ThreadReader;

/* C bindings */

/**
 * Make a new ThreadReader from an Archive and a threadId.
 * @param archive Archive to read.
 * @param threadId Id of the thread to read.
 * @param options Options as defined in ThreadReaderOptions.
 */
ThreadReader pallasCreateThreadReader(Archive* archive, ThreadId threadId, int options);
/** Prints the current Token. */
void pallasPrintCurToken(ThreadReader *thread_reader);
/** Gets the current Iterable. */
const Token* pallasGetCurIterable(ThreadReader *thread_reader);
/** Prints the current Sequence. */
void pallasPrintCurSequence(ThreadReader *thread_reader);
/** Prints the whole current callstack. */
void pallasPrintCallstack(ThreadReader *thread_reader);
/** Returns the EventSummary of the given Event. */
EventSummary* pallasGetEventSummary(ThreadReader *thread_reader, Token event);
/** Returns the timestamp of the given event occurring at the given index. */
pallas_timestamp_t pallasGetEventTimestamp(ThreadReader *thread_reader, Token event, int occurence_id);
/** Returns whether the given sequence still has more Tokens after the given current_index. */
bool pallasIsEndOfSequence(ThreadReader *thread_reader, int current_index, Token sequence_id);
/** Returns whether the given loop still has more Tokens after the given current_index. */
bool pallasIsEndOfLoop(ThreadReader *thread_reader, int current_index, Token loop_id);
/** Returns whether the cursor is at the end of the current block. */
bool pallasIsEndOfCurrentBlock(ThreadReader *thread_reader);
/** Returns whether the cursor is at the end of the trace. */
bool pallasIsEndOfTrace(ThreadReader *thread_reader);
/** Returns the duration of the given Loop. */
pallas_duration_t pallasGetLoopDuration(ThreadReader *thread_reader, Token loop_id);

/** Returns an EventOccurence for the given Token appearing at the given occurence_id.
 * Timestamp is set to Reader's referential timestamp.*/
EventOccurence pallasGetEventOccurence(ThreadReader *thread_reader, Token event_id, size_t occurence_id);
/** Returns an SequenceOccurence for the given Token appearing at the given occurence_id.
 * Timestamp is set to Reader's referential timestamp.*/
SequenceOccurence pallasGetSequenceOccurence(ThreadReader *thread_reader,
                                             Token sequence_id,
                                             size_t occurence_id,
                                             bool create_checkpoint = false);
/** Returns an LoopOccurence for the given Token appearing at the given occurence_id.
 * Timestamp is set to Reader's referential timestamp.*/
LoopOccurence pallasGetLoopOccurence(ThreadReader *thread_reader, Token loop_id, size_t occurence_id);

/** Returns a pointer to the AttributeList for the given occurence of the given Event. */
AttributeList* pallasGetEventAttributeList(ThreadReader *thread_reader, Token event_id, size_t occurence_id);

void pallasLoadCheckpoint(ThreadReader *thread_reader, Cursor *checkpoint);

//******************* EXPLORATION FUNCTIONS ********************

/** Gets the current Token. */
const Token* pallasPollCurToken(ThreadReader *thread_reader);
/** Peeks at and return the next token without actually updating the state */
const Token* pallasPollNextToken(ThreadReader *thread_reader);
/** Peeks at and return the previous token without actually updating the state */
const Token* pallasPollPrevToken(ThreadReader *thread_reader);
/** Updates the internal state */
void pallasMoveToNextToken(ThreadReader *thread_reader);
/** Updates the internal state */
void pallasMoveToPrevToken(ThreadReader *thread_reader);
/** Gets the next token and updates the reader's state if it returns a value.
 * It is more or less equivalent to `moveToNextToken()` then `pollCurToken()` */
Token* pallasGetNextToken(ThreadReader *thread_reader, int flags);
/** Enters a block */
void pallasEnterBlock(ThreadReader *thread_reader);
/** Leaves the current block */
void pallasLeaveBlock(ThreadReader *thread_reader);
/** Exits a block if at the end of it and flags allow it, returns a boolean representing if the reader actually exited a block */
bool pallasExitIfEndOfBlock(ThreadReader *thread_reader);

Cursor pallasCreateCheckpoint(ThreadReader *thread_reader);

#ifdef __cplusplus
}; /* namespace pallas */
#endif
/* -*-
   mode: c;
   c-file-style: "k&r";
   c-basic-offset 2;
   tab-width 2 ;
   indent-tabs-mode nil
   -*- */

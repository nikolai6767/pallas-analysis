/*
 * Copyright (C) Telecom SudParis
 * See LICENSE in top-level directory.
 */
/** @file
 * Everything that has to do with writing a trace during an execution.
 */
#pragma once
#include "pallas.h"
#include "pallas_archive.h"
#include "pallas_attribute.h"
#ifdef __cplusplus
namespace pallas {
#endif
/**
 * Writes one thread to the Pallas trace format.
 */
typedef struct ThreadWriter {
  Thread thread_trace; /**< Thread being written. */
  C_CXX(void, std::vector<Token>) *
    sequence_stack; /**< Stack of all the *incomplete* sequences the writer is currently in. */
  int cur_depth;    /**< Current depth in the callstack. */
  int max_depth;    /**< Maximum depth in the callstack. */
  int thread_rank;  /**< Rank of this thread. todo: MPI rank ? */

  pallas_timestamp_t last_timestamp; /**< Timestamp of the last encountered event */

  pallas_duration_t* last_duration; /**< Pointer to the last event duration (to be updated when the timestamp of the
                                    next event is known) */

  pallas_timestamp_t*
    sequence_start_timestamp; /**< Start date of each ongoing sequence (used for computing the sequence duration) */

  /**
   * This is a vector of durations that are incomplete.
   * Those are durations from sequence whose last duration was still a timestamp.
   * Next time a timestamp is registered, and thus the duration for the last event in those sequences is computed,
   * they'll be updated.
   */
  DEFINE_Vector(pallas_duration_t*, incompleteDurations);
  /** The first recorded timestamp for this thread.*/
  C_CXX(uint8_t firstTimestamp[TIMEPOINT_SIZE], Timepoint firstTimestamp = {});
#ifdef __cplusplus

 private:
  void findLoopBasic(size_t maxLoopLength);
  void findLoopFilter();
  /** Tries to find a Loop in the current array of tokens.  */
  void findLoop();
  /** Creates a Loop in the trace, and returns a pointer to it.
   * Does not change the current array of tokens.
   * @param start_index Starting index of the loop (first token in the loop).
   * @param loop_len Lenght of the sequence repeated in the loop.
   * */
  Loop* createLoop(size_t start_index, size_t loop_len);
  /** Create a Loop and change the current array of token to reflect that.
   *
   * For example, replaces `[E1, E2, E3, E4, E1, E2, E3, E4]` with `[L1]`,
   * where L1 contains 2 * S1 = `[E1, E2, E3, E4]`
   * @param loop_len Length of the sequence repeated in the loop.
   * @param index_first_iteration Starting index of the first iteration of the loop.
   * @param index_second_iteration Starting index of the second iteration of the loop.
   */
  void replaceTokensInLoop(int loop_len, size_t index_first_iteration, size_t index_second_iteration);
  /** Returns a reference to the current sequence of Tokens being written. */
  [[nodiscard]] std::vector<Token>& getCurrentTokenSequence() const { return sequence_stack[cur_depth]; };
  /** Stores the timestamp in the given EventSummary. */
  void storeTimestamp(EventSummary* es, pallas_timestamp_t ts);
  /** Stores the attribute list in the given EventSummary. */
  void storeAttributeList(EventSummary* es, AttributeList* attribute_list, size_t occurence_index);
  /** Stores the tokens in that Sequence's array of Tokens, then tries to find a Loop.*/
  void storeToken(std::vector<Token>& tokenSeq, Token t);
  /** Move up the callstack and create a new Sequence. */
  void recordEnterFunction();
  /** Close a Sequence and move down the callstack. */
  void recordExitFunction();
  /** Returns the current timestamp. */
  pallas_timestamp_t getTimestamp();
  /** Returns t if it's valid, of the current timestamp. */
  pallas_timestamp_t timestamp(pallas_timestamp_t t);
  /** Adds the given duration to all the stored addresses to complete.*/
  void completeDurations(pallas_duration_t duration);
  /** Adds the given address to a list of duration to complete. */
  void addDurationToComplete(pallas_duration_t* duration);

 public:
  void open(Archive* archive, ThreadId thread_id);
  void threadClose();
  /** Creates the new Event and stores it. Returns the occurence index of that new Event. */
  size_t storeEvent(enum EventType event_type,
                    TokenId event_id,
                    pallas_timestamp_t ts,
                    struct AttributeList* attribute_list);
#endif
} ThreadWriter;
#ifdef __cplusplus
}
extern "C" {
#endif

/* C Callbacks */
/**
 * Allocates a new ThreadWriter and returns a pointer to that allocated memory.
 * @return Pointer to ThreadWriter
 */
extern PALLAS(ThreadWriter) * pallas_thread_writer_new();
extern void pallas_write_global_archive_open(PALLAS(Archive) * archive, const char* dir_name, const char* trace_name);
extern void pallas_write_global_archive_close(PALLAS(Archive) * archive);

extern void pallas_write_thread_open(PALLAS(Archive) * archive,
                                     PALLAS(ThreadWriter) * thread_writer,
                                     PALLAS(ThreadId) thread_id);

extern void pallas_write_thread_close(PALLAS(ThreadWriter) * thread_writer);

extern void pallas_write_define_location_group(PALLAS(Archive) * archive,
                                               PALLAS(LocationGroupId) id,
                                               PALLAS(StringRef) name,
                                               PALLAS(LocationGroupId) parent);

extern void pallas_write_define_location(PALLAS(Archive) * archive,
                                         PALLAS(ThreadId) id,
                                         PALLAS(StringRef) name,
                                         PALLAS(LocationGroupId) parent);

extern void pallas_write_archive_open(PALLAS(Archive) * archive,
                                      const char* dir_name,
                                      const char* trace_name,
                                      PALLAS(LocationGroupId) location_group);

extern void pallas_write_archive_close(PALLAS(Archive) * archive);

extern void pallas_store_event(PALLAS(ThreadWriter) * thread_writer,
                               enum PALLAS(EventType) event_type,
                               PALLAS(TokenId) id,
                               pallas_timestamp_t ts,
                               PALLAS(AttributeList) * attribute_list);

/* Event Records */

extern void pallas_record_enter(PALLAS(ThreadWriter) * thread_writer,
                                PALLAS(AttributeList) * attributeList,
                                pallas_timestamp_t time,
                                PALLAS(RegionRef) region_ref);

extern void pallas_record_leave(PALLAS(ThreadWriter) * thread_writer,
                                PALLAS(AttributeList) * attributeList,
                                pallas_timestamp_t time,
                                PALLAS(RegionRef) region_ref);

extern void pallas_record_thread_begin(PALLAS(ThreadWriter) * thread_writer,
                                       PALLAS(AttributeList) * attributeList,
                                       pallas_timestamp_t time);

extern void pallas_record_thread_end(PALLAS(ThreadWriter) * thread_writer,
                                     PALLAS(AttributeList) * attributeList,
                                     pallas_timestamp_t time);

extern void pallas_record_thread_team_begin(PALLAS(ThreadWriter) * thread_writer,
                                            PALLAS(AttributeList) * attributeList,
                                            pallas_timestamp_t time);

extern void pallas_record_thread_team_end(PALLAS(ThreadWriter) * thread_writer,
                                          PALLAS(AttributeList) * attributeList,
                                          pallas_timestamp_t time);

extern void pallas_record_mpi_send(PALLAS(ThreadWriter) * thread_writer,
                                   PALLAS(AttributeList) * attributeList,
                                   pallas_timestamp_t time,
                                   uint32_t receiver,
                                   uint32_t communicator,
                                   uint32_t msgTag,
                                   uint64_t msgLength);

extern void pallas_record_mpi_isend(PALLAS(ThreadWriter) * thread_writer,
                                    PALLAS(AttributeList) * attribute_list,
                                    pallas_timestamp_t time,
                                    uint32_t receiver,
                                    uint32_t communicator,
                                    uint32_t msgTag,
                                    uint64_t msgLength,
                                    uint64_t requestID);

extern void pallas_record_mpi_isend_complete(PALLAS(ThreadWriter) * thread_writer,
                                             PALLAS(AttributeList) * attribute_list,
                                             pallas_timestamp_t time,
                                             uint64_t requestID);

extern void pallas_record_mpi_irecv_request(PALLAS(ThreadWriter) * thread_writer,
                                            PALLAS(AttributeList) * attribute_list,
                                            pallas_timestamp_t time,
                                            uint64_t requestID);

extern void pallas_record_mpi_recv(PALLAS(ThreadWriter) * thread_writer,
                                   PALLAS(AttributeList) * attributeList,
                                   pallas_timestamp_t time,
                                   uint32_t sender,
                                   uint32_t communicator,
                                   uint32_t msgTag,
                                   uint64_t msgLength);

extern void pallas_record_mpi_irecv(PALLAS(ThreadWriter) * thread_writer,
                                    PALLAS(AttributeList) * attribute_list,
                                    pallas_timestamp_t time,
                                    uint32_t sender,
                                    uint32_t communicator,
                                    uint32_t msgTag,
                                    uint64_t msgLength,
                                    uint64_t requestID);

extern void pallas_record_mpi_collective_begin(PALLAS(ThreadWriter) * thread_writer,
                                               PALLAS(AttributeList) * attribute_list,
                                               pallas_timestamp_t time);

extern void pallas_record_mpi_collective_end(PALLAS(ThreadWriter) * thread_writer,
                                             PALLAS(AttributeList) * attribute_list,
                                             pallas_timestamp_t time,
                                             uint32_t collectiveOp,
                                             uint32_t communicator,
                                             uint32_t root,
                                             uint64_t sizeSent,
                                             uint64_t sizeReceived);

extern void pallas_record_generic(PALLAS(ThreadWriter *) thread_writer,
                                PALLAS(AttributeList) * attribute_list,
                                pallas_timestamp_t time,
                                PALLAS(StringRef) event_name);

#ifdef __cplusplus
};
#endif

extern C_CXX(_Thread_local, thread_local) int pallas_recursion_shield;

/* -*-
   mode: c;
   c-file-style: "k&r";
   c-basic-offset 2;
   tab-width 2 ;
   indent-tabs-mode nil
   -*- */

/*
 * Copyright (C) Telecom SudParis
 * See LICENSE in top-level directory.
 */
#include <gtest/gtest.h>
#include <filesystem>

#if __cplusplus >= 202002L && (__GNUC__ >= 13 || __clang__ >= 14 || _MSC_VER >= 1929)
#include <format>
#define HAS_FORMAT
#endif
#include "pallas/pallas.h"
#include "pallas/pallas_record.h"
#include "pallas/pallas_write.h"

using namespace pallas;
static pallas_timestamp_t ts = 0;
static pallas_timestamp_t step = 1;
static pallas_timestamp_t get_timestamp() {
  ts += step;
  return ts;
}

static inline std::string dummyTraceName = "unit_test_trace";

void testFindLoop(int MAX_EVENT, int NUM_LOOPS) {
  /* Make a dummy archive and a dummy thread writer. */
  auto archive = GlobalArchive();
  archive.open(dummyTraceName.c_str(), dummyTraceName.c_str());
  auto a = Archive();
  a.open(dummyTraceName.c_str(), dummyTraceName.c_str(), 0);
  ThreadWriter thread_writer;
  thread_writer.open(&a, 0);

  /* Start recording some events.*/
  for (int eid = 0; eid < MAX_EVENT; eid++) {
#ifdef HAS_FORMAT
    archive.addString(eid, std::format("dummyEvent{}", eid).c_str());
#else
    archive.addString(eid, "dummyEvent");
#endif
    pallas_record_generic(&thread_writer, nullptr, get_timestamp(), eid);
  }
  /* Check they've been correctly registered. */
  EXPECT_EQ(thread_writer.cur_depth, 0);
  EXPECT_EQ(thread_writer.sequence_stack[0].size(), MAX_EVENT);
  for (int eid = 0; eid < MAX_EVENT; eid++) {
    EXPECT_EQ(thread_writer.sequence_stack[0][eid].type, TypeEvent);
    EXPECT_EQ(thread_writer.sequence_stack[0][eid].id, eid);
  }

  /* Start recording some more events. This should make a first loop. */
  for (int eid = 0; eid < MAX_EVENT; eid++)
    pallas_record_generic(&thread_writer, nullptr, get_timestamp(), eid);

  /* This should have been recognized as a loop, so now there should be some changes. */
  EXPECT_EQ(thread_writer.cur_depth, 0);
  EXPECT_EQ(thread_writer.sequence_stack[0].size(), 1);
  EXPECT_EQ(thread_writer.sequence_stack[0][0].type, TypeLoop);
  /* Check that the loop is correct */
  auto& firstLoop = thread_writer.thread_trace.loops[0];
  EXPECT_EQ(firstLoop.nb_iterations, 2);

  /* Check that the sequence inside that loop is correct */
  struct Sequence* s = thread_writer.thread_trace.getSequence(firstLoop.repeated_token);

  EXPECT_EQ(s->size(), MAX_EVENT);

  for (int eid = 0; eid < MAX_EVENT; eid++) {
    EXPECT_EQ(s->tokens[eid].type, TypeEvent);
    EXPECT_EQ(s->tokens[eid].id, eid);
  }

  /* Start recording even more events. The first loop happens 3 times now.*/
  for (int eid = 0; eid < MAX_EVENT; eid++)
    pallas_record_generic(&thread_writer, nullptr, get_timestamp(), eid);
  EXPECT_EQ(thread_writer.cur_depth, 0);
  EXPECT_EQ(thread_writer.sequence_stack[0].size(), 1);
  EXPECT_EQ(firstLoop.nb_iterations, 3);

  /* Now start recording one more event and then loop again. */
#ifdef HAS_FORMAT
  archive.addString(MAX_EVENT, std::format("dummyEvent{}", MAX_EVENT).c_str());
#else
  archive.addString(MAX_EVENT, "dummyEvent");
#endif
  pallas_record_generic(&thread_writer, nullptr, get_timestamp(), MAX_EVENT);

  DOFOR(loop_number, NUM_LOOPS) {
    DOFOR(eid, MAX_EVENT) {
      pallas_record_generic(&thread_writer, nullptr, get_timestamp(), eid);
    }
  }
  EXPECT_EQ(thread_writer.cur_depth, 0);
  EXPECT_EQ(thread_writer.sequence_stack[0].size(), 3);  // L0 E L1
  EXPECT_EQ(firstLoop.nb_iterations, 3);

  auto& secondLoop = thread_writer.thread_trace.loops[1];
  EXPECT_EQ(secondLoop.nb_iterations, NUM_LOOPS);

  thread_writer.threadClose();
  archive.close();
  // TODO Find a way for the test to clean this trace
  //      Because somehow the following does not remove the folder
  //      Only is content
  //      std::filesystem::remove_all(dummyTraceName);
}

static inline void check_event_allocation(Thread* thread_trace, unsigned id) {
  while (id > thread_trace->nb_allocated_events) {
    DOUBLE_MEMORY_SPACE_CONSTRUCTOR(thread_trace->events, thread_trace->nb_allocated_events, struct EventSummary);
  }
  if (thread_trace->nb_events < id + 1) {
    thread_trace->nb_events = id + 1;
  }
}

void testSequenceDuration() {
  /* Make a dummy archive and a dummy thread writer. */
  Archive archive;
  archive.open("sequence_duration_trace", "sequence_duration_trace", 0);

  ThreadWriter thread_writer;

  thread_writer.open(&archive, 0);

  /* Here's what we're going to do: we'll define some sequences as the following:
   * S1 = E1 E2
   * S2 = E2 E3 E4
   * S3 = E3 E4 E5 E6
   * and L_i will be a repetition of S_i+1 (as S0 is always taken)
   * What we then need to do is a final repetition of
   * S_n = L1 L2 ... L_n-1
   * And then we'll check that the durations of all the sequences is oki doki
   */

  int OUTER_LOOP_SIZE = 2;
  int INNER_LOOP_SIZE = 10;
  int MAX_SUBSEQUENCE_NUMBER = 10;

  for (int outer_loop_number = 1; outer_loop_number <= OUTER_LOOP_SIZE; outer_loop_number++) {
    // Outer loop for S_n
    for (int sequence_number = 1; sequence_number <= MAX_SUBSEQUENCE_NUMBER; sequence_number++) {
      // Doing all the L_i containing the S_i
      for (int loop = 0; loop < INNER_LOOP_SIZE; loop++) {
        // Finally, doing the sequence
        for (int eid = 0; eid <= sequence_number; eid++)
          pallas_record_generic(&thread_writer, nullptr, get_timestamp(),
                                sequence_number * MAX_SUBSEQUENCE_NUMBER + eid);
      }
    }
  }
  pallas_record_generic(&thread_writer, nullptr, get_timestamp(), 0);
  thread_writer.thread_trace.events[0].durations->at(0) = 0;
  thread_writer.threadClose();
  archive.close();

  for (int sequence_number = 0; sequence_number <= MAX_SUBSEQUENCE_NUMBER; sequence_number++) {
    Sequence* s = thread_writer.thread_trace.sequences[sequence_number];
    if (sequence_number > 0) {
      EXPECT_EQ(s->tokens.size(), sequence_number + 1);
      EXPECT_EQ(s->durations->size, INNER_LOOP_SIZE * OUTER_LOOP_SIZE);
      for (auto t : *s->durations) {
        EXPECT_EQ(t, s->size());
      }
    }
  }
}

void testVector(size_t TEST_SIZE) {
  LinkedDurationVector vector = LinkedDurationVector();
  for (size_t i = 0; i < TEST_SIZE; i++) {
    vector.add(i);
  }
  vector.finalUpdateStats();
  EXPECT_EQ(vector.size, TEST_SIZE);
  EXPECT_EQ(vector.min, 0);
  EXPECT_EQ(vector.max, TEST_SIZE - 1);
}

TEST(UnitTests, Vector) {
  testVector(500);
}

TEST(UnitTests, FindLoop) {
  testFindLoop(50, 100);
}

TEST(UnitTests, SequenceDuration) {
  testSequenceDuration();
}

/* -*-
   mode: c;
   c-file-style: "k&r";
   c-basic-offset 2;
   tab-width 2 ;
   indent-tabs-mode nil
   -*- */

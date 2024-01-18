/*
 * Copyright (C) Telecom SudParis
 * See LICENSE in top-level directory.
 */
#include <filesystem>
#include "pallas/pallas.h"
#include "pallas/pallas_write.h"
using namespace pallas;

static pallas_timestamp_t ts = 0;
static pallas_timestamp_t step = 1;
static pallas_timestamp_t get_timestamp() {
  ts += step;
  return ts;
}

static inline std::string dummyTraceName = "dummy_trace";

static inline void check_event_allocation(Thread* thread_trace, unsigned id) {
  pallas_log(DebugLevel::Max, "Searching for event {.id=%d}\n", id);

  while (id > thread_trace->nb_allocated_events) {
    pallas_warn("Doubling mem space of events for thread trace %p\n", (void*)thread_trace);
    DOUBLE_MEMORY_SPACE_CONSTRUCTOR(thread_trace->events, thread_trace->nb_allocated_events, struct EventSummary);
  }
  if (thread_trace->nb_events < id + 1) {
    thread_trace->nb_events = id + 1;
  }
}

static void init_dummy_event(ThreadWriter* thread_writer, int id) {
  check_event_allocation(&thread_writer->thread_trace, id);
  thread_writer->storeEvent(PALLAS_SINGLETON, id, get_timestamp(), nullptr);
}

int main(int argc, char** argv __attribute__((unused))) {
  if (argc < 2) {
    pallas_error("Not enough arguments ! 2 argument required.\n");
  }
  if (argc > 3) {
    pallas_error("Too many arguments ! 3 argument required.\n");
  }
  int MAX_EVENT = std::stoi(argv[1]);
  int NUM_LOOPS = std::stoi(argv[2]);

  /* Make a dummy archive and a dummy thread writer. */
  struct Archive archive;
  archive.open(dummyTraceName.c_str(), dummyTraceName.c_str(), 0);
  struct ThreadWriter thread_writer;
  thread_writer.open(&archive, 0);

  /* Start recording some events.*/
  for (int eid = 0; eid < MAX_EVENT; eid++)
    init_dummy_event(&thread_writer, (Record)eid);

  /* Check they've been correctly registered. */
  pallas_assert_always(thread_writer.cur_depth == 0);
  pallas_assert_always(thread_writer.og_seq[0]->size() == (unsigned int)MAX_EVENT);
  for (int eid = 0; eid < MAX_EVENT; eid++) {
    pallas_assert_always(thread_writer.og_seq[0]->tokens[eid].type == TypeEvent);
    pallas_assert_always(thread_writer.og_seq[0]->tokens[eid].id == eid);
  }

  /* Start recording some more events. This should make a first loop. */
  for (int eid = 0; eid < MAX_EVENT; eid++)
    init_dummy_event(&thread_writer, eid);

  /* This should have been recognized as a loop, so now there should be some changes. */
  pallas_assert_always(thread_writer.cur_depth == 0);
  pallas_assert_always(thread_writer.og_seq[0]->size() == 1);
  pallas_assert_always(thread_writer.og_seq[0]->tokens[0].type == TypeLoop);
  /* Check that the loop is correct */
  struct Loop* l = thread_writer.thread_trace.getLoop(thread_writer.og_seq[0]->tokens[0]);
  pallas_assert_always(l->nb_iterations[0] == 2);

  /* Check that the sequence inside that loop is correct */
  struct Sequence* s = thread_writer.thread_trace.getSequence(l->repeated_token);

  pallas_assert_always(s->size() == (unsigned int)MAX_EVENT);

  for (int eid = 0; eid < MAX_EVENT; eid++) {
    pallas_assert_always(s->tokens[eid].type == TypeEvent);
    pallas_assert_always(s->tokens[eid].id == eid);
  }

  /* Start recording even more events. The first loop happens 3 times now.*/
  for (int eid = 0; eid < MAX_EVENT; eid++)
    init_dummy_event(&thread_writer, eid);
  pallas_assert_always(thread_writer.cur_depth == 0);
  pallas_assert_always(thread_writer.og_seq[0]->size() == 1);
  pallas_assert_always(l->nb_iterations[0] == 3);

  /* Now start recording one more event and then loop again. */
  init_dummy_event(&thread_writer, MAX_EVENT);
  DOFOR(loop_number, NUM_LOOPS) {
    DOFOR(eid, MAX_EVENT)
    init_dummy_event(&thread_writer, eid);
  }
  pallas_assert_always(thread_writer.cur_depth == 0);
  pallas_assert_always(thread_writer.og_seq[0]->size() == 3);  // L0 E L0
  pallas_assert_always(l->nb_iterations[0] == 3);
  pallas_assert_always(l->nb_iterations[1] == (size_t)NUM_LOOPS);

  thread_writer.threadClose();
  archive.close();
  // TODO Find a way for the test to clean this trace
  //      Because somehow the following does not remove the folder
  //      Only is content
  //      std::filesystem::remove_all(dummyTraceName);
  return 0;
}

/* -*-
   mode: c;
   c-file-style: "k&r";
   c-basic-offset 2;
   tab-width 2 ;
   indent-tabs-mode nil
   -*- */

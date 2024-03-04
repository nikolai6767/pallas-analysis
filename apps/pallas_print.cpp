/*
 * Copyright (C) Telecom SudParis
 * See LICENSE in top-level directory.
 */
#include <algorithm>
#include <iostream>
#include <string>
#include "pallas/pallas.h"
#include "pallas/pallas_archive.h"
#include "pallas/pallas_read.h"
#include "pallas/pallas_storage.h"

static bool show_structure = false;
static bool per_thread = false;
static std::string structure_indent[MAX_CALLSTACK_DEPTH];
static short store_timestamps = 1;

static bool print_timestamp = true;
static bool print_duration = false;
static bool unroll_loops = true;
static bool explore_loop_sequences = true;
static bool verbose = false;

static void _print_timestamp(pallas_timestamp_t ts) {
  if (print_timestamp) {
    printf("%21.9lf\t", ts / 1e9);
  }
}
static void _print_timestamp_header() {
  if (print_timestamp) {
    printf("%21s\t", "Timestamp");
  }
}

static void _print_duration(pallas_timestamp_t d) {
  if (print_duration) {
    printf("%21.9lf\t", d / 1e9);
  }
}

static void _print_duration_header() {
  if (print_duration) {
    printf("%21s\t", "Duration");
  }
}

static void _print_indent(const std::string& current_indent) {
  std::cout << current_indent;
}

/* Print one event */
static void print_event(const std::string& current_indent,
                        const pallas::Thread* thread,
                        const pallas::Token token,
                        const pallas::EventOccurence* e) {
  _print_timestamp(e->timestamp);
  _print_duration(e->duration);
  _print_indent(current_indent);

  if (!per_thread)
    std::cout << thread->getName() << "\t";
  if (verbose) {
    thread->printToken(token);
    std::cout << "\t";
  }
  thread->printEvent(e->event);
  thread->printEventAttribute(e);
  std::cout << std::endl;
}

static void print_sequence(const std::string& current_indent,
                           const pallas::Thread* thread,
                           const pallas::Token token,
                           const pallas::SequenceOccurence* sequenceOccurence,
                           const pallas::LoopOccurence* containingLoopOccurence = nullptr) {
  auto* sequence = sequenceOccurence->sequence;
  pallas_timestamp_t ts = sequenceOccurence->timestamp;
  pallas_timestamp_t duration = sequenceOccurence->duration;

  _print_timestamp(ts);
  _print_duration(duration);
  _print_indent(current_indent);

  if (!per_thread)
    std::cout << thread->getName() << "\t";
  if (show_structure) {
    std::cout << "Sequence ";
    thread->printToken(token);
  }

  if (verbose) {
    thread->printToken(token);
    printf("\t");
    for (unsigned i = 0; i < sequence->size(); i++) {
      thread->printToken(sequence->tokens[i]);
      std::cout << " ";
    }
  }

  if (containingLoopOccurence && (unroll_loops || explore_loop_sequences)) {
    //    pallas_timestamp_t mean = containingLoopOccurence->duration / containingLoopOccurence->nb_iterations;
    //    std::cout << ((mean < duration) ? "-" : "+");
    //    uint64_t diff = (mean < duration) ? duration - mean : mean - duration;
    //    float percentile = (float)diff / (float)mean * 100;
    //    printf("%5.2f%%", percentile);
  }
  std::cout << std::endl;
}

static void print_loop(const std::string& current_indent,
                       const pallas::Thread* thread,
                       const pallas::Token token,
                       const pallas::LoopOccurence* loopOccurence) {
  _print_timestamp(loopOccurence->timestamp);
  _print_duration(loopOccurence->duration);
  _print_indent(current_indent);

  if (!per_thread)
    std::cout << thread->getName() << "\t";

  if (show_structure) {
    std::cout << "Loop ";
  }

  auto* loop = loopOccurence->loop;

  thread->printToken(token);
  std::cout << "\t" << loopOccurence->nb_iterations << " * ";
  thread->printToken(loop->repeated_token);
  std::cout << std::endl;
}

static void print_token(const pallas::Thread* thread,
                        const pallas::Token* t,
                        const pallas::Occurence* e,
                        int depth = 0,
                        int last_one = 0,
                        const pallas::LoopOccurence* containing_loop = nullptr) {
  pallas_log(pallas::DebugLevel::Verbose, "Reading repeated_token(%d.%d) for thread %s\n", t->type, t->id, thread->getName());
  // Prints the structure of the sequences and the loops
  std::string current_indent;
  if (show_structure && depth >= 1) {
    structure_indent[depth - 1] = (last_one ? "╰" : "├");
    DOFOR(i, depth) {
      current_indent += structure_indent[i];
    }
    if (t->type != pallas::TypeEvent) {
      current_indent += (containing_loop && !explore_loop_sequences) ? "─" : "┬";
    } else {
      current_indent += "─";
    }
    structure_indent[depth - 1] = last_one ? " " : "│";
  }

  // Prints the repeated_token we first started with
  switch (t->type) {
  case pallas::TypeInvalid:
    pallas_error("Type is invalid\n");
    break;
  case pallas::TypeEvent:
    print_event(current_indent, thread, *t, &e->event_occurence);
    break;
  case pallas::TypeSequence: {
    if (show_structure)
      print_sequence(current_indent, thread, *t, &e->sequence_occurence, containing_loop);
    break;
  }
  case pallas::TypeLoop: {
    if (show_structure)
      print_loop(current_indent, thread, *t, &e->loop_occurence);
    break;
  }
  }
}

static void display_sequence(pallas::ThreadReader* reader,
                             pallas::Token token,
                             pallas::SequenceOccurence* occurence,
                             int depth) {
  if (occurence) {
    load_savestate(reader, occurence->savestate);
    reader->enterBlock(token);
  }
  auto current_level = reader->readCurrentLevel();
  for (const auto& tokenOccurence : current_level) {
    print_token(reader->thread_trace, tokenOccurence.token, tokenOccurence.occurence, depth,
                &tokenOccurence == &current_level.back());
    if (tokenOccurence.token->type == pallas::TypeSequence) {
      display_sequence(reader, *tokenOccurence.token, &tokenOccurence.occurence->sequence_occurence, depth + 1);
    }
    if (tokenOccurence.token->type == pallas::TypeLoop) {
      pallas::LoopOccurence& loop = tokenOccurence.occurence->loop_occurence;
      // The printing of loops is a bit convoluted, because there's no right way to do it
      // Here, we offer four ways to print loops:
      //    - Print only the Sequence inside once, with its mean/median duration
      //    - Print only the Sequence inside once, and also print what's inside of it, with their mean/median duration
      //    - Print the Sequence inside as much time as needed, but don't unroll it
      //    - Print the Sequence inside as much time as needed, and unroll it.
      // So we basically need two booleans: unroll_loops, which we use to determine if we need to print each Sequence
      // inside the loop, and explore_loop_sequences, which we use to determine if we need to print the inside of
      // each Sequence.
      // We'll do each option one after another
      if (!unroll_loops && !explore_loop_sequences) {
        loop.loop_summary = pallas::SequenceOccurence();
        loop.loop_summary.sequence = loop.full_loop[0].sequence;
        for (uint j = 0; j < loop.nb_iterations; j++) {
          loop.loop_summary.duration += loop.full_loop[j].duration / loop.nb_iterations;
        }
        // Don't do this at home kids
        print_token(reader->thread_trace, &loop.loop->repeated_token, (pallas::Occurence*)&loop.loop_summary, depth + 1,
                    true, &loop);
      }
      if (!unroll_loops && explore_loop_sequences) {
        pallas_error("Not implemented yet ! Sorry\n");
      }
      if (unroll_loops) {
        for (uint j = 0; j < loop.nb_iterations; j++) {
          pallas::SequenceOccurence* seq = &loop.full_loop[j];
          print_token(reader->thread_trace, &loop.loop->repeated_token, (pallas::Occurence*)seq, depth + 1,
                      j == loop.nb_iterations - 1, &loop);
          if (explore_loop_sequences) {
            display_sequence(reader, loop.loop->repeated_token, seq, depth + 2);
          }
        }
      }
    }
  }
  if (occurence) {
    occurence->full_sequence = new pallas::TokenOccurence[current_level.size()];
    memcpy(occurence->full_sequence, current_level.data(), current_level.size() * sizeof(pallas::TokenOccurence));
  }
  reader->leaveBlock();
}

/* Print all the events of a thread */
static void print_thread(pallas::Archive& trace, pallas::Thread* thread) {
  printf("Reading events for thread %u (%s):\n", thread->id, thread->getName());
  _print_timestamp_header();
  _print_duration_header();

  printf("Event\n");

  int reader_options = pallas::ThreadReaderOptions::None;
  if (show_structure)
    reader_options |= pallas::ThreadReaderOptions::ShowStructure;
  if (!store_timestamps || trace.store_timestamps == 0)
    reader_options |= pallas::ThreadReaderOptions::NoTimestamps;

  auto* reader = new pallas::ThreadReader(&trace, thread->id, reader_options);

  display_sequence(reader, pallas::Token(pallas::TypeSequence, 0), nullptr, 0);
}

/**
 * Compare the timestamps of the current token on each thread and select the smallest timestamp.
 * @returns Tuple containing the ThreadId and a TokenOccurence.
 *          You are responsible for the memory of the TokenOccurence.
 */
static std::tuple<pallas::ThreadId, pallas::TokenOccurence> getNextToken(std::vector<pallas::ThreadReader>& threadReaders) {
  // Find the earliest threadReader
  pallas::ThreadReader* earliestReader = nullptr;
  for (auto& reader : threadReaders) {
    // Check if reader has finished reading its trace
    if (reader.current_frame < 0)
      continue;
    if (!earliestReader || earliestReader->referential_timestamp > reader.referential_timestamp) {
      earliestReader = &reader;
    }
  }

  // If no reader was available
  if (!earliestReader) {
    return {PALLAS_THREAD_ID_INVALID, {nullptr, nullptr}};
  }

  // Grab the interesting information
  auto& token = earliestReader->getCurToken();
  auto* occurrence = earliestReader->getOccurence(token, earliestReader->tokenCount[token]);
  auto threadId = earliestReader->thread_trace->id;

  // Update the reader
  earliestReader->updateReadCurToken();
  if (token.type == pallas::TypeEvent)
    earliestReader->moveToNextToken();

  return {threadId, {&token, occurrence}};
}

/** Print all the events of all the threads sorted by timestamp*/
void printTrace(pallas::Archive& trace) {
  auto readers = std::vector<pallas::ThreadReader>();
  int reader_options = pallas::ThreadReaderOptions::None;
  //  if (show_structure)
  //    reader_options |= pallas::ThreadReaderOptions::ShowStructure;

  for (int i = 0; i < trace.nb_threads; i++) {
    readers.emplace_back(&trace, trace.threads[i]->id, reader_options);
  }

  _print_timestamp_header();
  _print_duration_header();
  printf("Event\n");

  pallas::ThreadId threadId;
  pallas::TokenOccurence tokenOccurence;
  std::tie(threadId, tokenOccurence) = getNextToken(readers);
  while (threadId != PALLAS_THREAD_ID_INVALID) {
    auto currentReader = std::find_if(readers.begin(), readers.end(), [&threadId](const pallas::ThreadReader& reader) {
      return reader.thread_trace->id == threadId;
    });
    print_token(currentReader->thread_trace, tokenOccurence.token, tokenOccurence.occurence);
    // If you read the doc, you'll know that the memory of tokenOccurence is ours to manage
    std::tie(threadId, tokenOccurence) = getNextToken(readers);
  }
}

void usage(const char* prog_name) {
  printf("Usage: %s [OPTION] trace_file\n", prog_name);
  printf("\t-T          Print events per thread\n");
  printf("\t-S          Structure mode\n");
  printf("\t-v          Verbose mode\n");
  printf("\t-d          Print duration of events\n");
  printf("\t-t          Don't print timestamps\n");
  printf("\t-u          Unroll loops\n");
  printf("\t-e          Explore sequences inside of loops\n");
  printf("\t-?  -h      Display this help and exit\n");
}

int main(int argc, char** argv) {
  int nb_opts = 0;
  char* trace_name = NULL;

  for (int i = 1; i < argc; i++) {
    if (!strcmp(argv[i], "-v")) {
      pallas_debug_level_set(pallas::DebugLevel::Debug);
      nb_opts++;
    } else if (!strcmp(argv[i], "-T")) {
      per_thread = true;
      nb_opts++;
    } else if (!strcmp(argv[i], "-d")) {
      print_duration = true;
      nb_opts++;
    } else if (!strcmp(argv[i], "-t")) {
      print_timestamp = false;
      nb_opts++;
    } else if (!strcmp(argv[i], "-S")) {
      per_thread = true;
      show_structure = true;
      unroll_loops = false;
      explore_loop_sequences = false;
      nb_opts++;
    } else if (show_structure && !strcmp(argv[i], "-u")) {
      unroll_loops = true;
      nb_opts++;
    } else if (show_structure && !strcmp(argv[i], "-e")) {
      explore_loop_sequences = true;
      nb_opts++;
    } else if (!strcmp(argv[i], "--no-timestamps")) {
      setenv("STORE_TIMESTAMPS", "FALSE", 0);
      print_timestamp = true;
      store_timestamps = 0;
      nb_opts++;
    } else if (!strcmp(argv[i], "-h") || !strcmp(argv[i], "-?")) {
      usage(argv[0]);
      return EXIT_SUCCESS;
    } else {
      /* Unknown parameter name. It's probably the trace's path name. We can stop
       * parsing the parameter list.
       */
      break;
    }
  }

  trace_name = argv[nb_opts + 1];
  if (trace_name == NULL) {
    usage(argv[0]);
    return EXIT_SUCCESS;
  }

  auto trace = pallas::Archive();
  pallas_read_archive(&trace, trace_name);
  store_timestamps = trace.store_timestamps;

  if (per_thread) {
    for (int i = 0; i < trace.nb_threads; i++) {
      printf("\n");
      print_thread(trace, trace.threads[i]);
    }
  } else {
    printTrace(trace);
  }

  return EXIT_SUCCESS;
}

/* -*-
   mode: c;
   c-file-style: "k&r";
   c-basic-offset 2;
   tab-width 2 ;
   indent-tabs-mode nil
   -*- */

/*
 * Copyright (C) Telecom SudParis
 * See LICENSE in top-level directory.
 */
#include <algorithm>
#include <iostream>
#include <list>
#include <string>
#include "pallas/pallas.h"
#include "pallas/pallas_archive.h"
#include "pallas/pallas_read.h"
#include "pallas/pallas_storage.h"


bool verbose = false;
bool per_thread = false;
bool show_durations = false;
bool show_timestamps = true;

static void _print_timestamp(pallas_timestamp_t ts) {
  if (show_timestamps) {
    printf("%21.9lf\t", ts / 1e9);
  }
}
static void _print_timestamp_header() {
  if (show_timestamps) {
    printf("%21s\t", "Timestamp");
  }
}

static void _print_duration(pallas_timestamp_t d) {
  if (show_durations) {
    printf("%21.9lf\t", d / 1e9);
  }
}

static void _print_duration_header() {
  if (show_durations) {
    printf("%21s\t", "Duration");
  }
}

/* Print one event */
static void printEvent(const pallas::Thread* thread,
                       const pallas::Token token,
                       const pallas::EventOccurence e) {
  _print_timestamp(e.timestamp);
  _print_duration(e.duration);

  if (!per_thread)
    std::cout << thread->getName() << "\t";
  if (verbose) {
    thread->printToken(token);
    std::cout << "\t";
  }
  thread->printEvent(e.event);
  thread->printEventAttribute(&e);
  std::cout << std::endl;
}

bool isReadingOver(const std::vector<pallas::ThreadReader>& readers) {
  for (const auto& reader : readers) {
    if (!reader.isEndOfTrace()) {
      return false;
    }
  }
  return true;
}

void printTrace(const pallas::GlobalArchive &trace) {
  auto readers = std::vector<pallas::ThreadReader>();
  int reader_options = pallas::ThreadReaderOptions::None;
  for (int i = 0; i < trace.nb_archives; i++) {
    for (int j = 0; j < trace.archive_list[i]->nb_threads; j ++)
      if (trace.archive_list[i]->threads[j])
        readers.emplace_back(trace.archive_list[i], trace.archive_list[i]->threads[j]->id, reader_options);
  }

  _print_timestamp_header();
  _print_duration_header();
  std::cout << std::endl;

  while (!isReadingOver(readers)) {
    pallas::ThreadReader *min_reader = &readers[0];
    pallas_timestamp_t min_timestamp = ULONG_MAX;
    for (int i = 1; i < readers.size(); i++) {
      if (!readers[i].isEndOfTrace() && readers[i].referential_timestamp < min_timestamp) {
        min_reader = &readers[i];
        min_timestamp = readers[i].referential_timestamp;
      }
    }

    auto token = min_reader->pollCurToken();
    if (token.type == pallas::TypeEvent) {
      printEvent(min_reader->thread_trace, token, min_reader->getEventOccurence(token, min_reader->tokenCount[token]));
    }

    if (!min_reader->getNextToken(PALLAS_READ_UNROLL_ALL).has_value()) {
      pallas_assert(min_reader->isEndOfTrace());
    }
  }
}

void printStructure(const int flags, const pallas::GlobalArchive& trace) {
  constexpr int reader_options = pallas::ThreadReaderOptions::None;
  for (int i = 0; i < trace.nb_archives; i++) {
    for (int j = 0; j < trace.archive_list[i]->nb_threads; j ++) {
      if (!trace.archive_list[i]->threads[j])
        continue;
      pallas::ThreadReader tr = pallas::ThreadReader(trace.archive_list[i], trace.archive_list[i]->threads[j]->id, reader_options);
      std::cout << "--- Thread " << j << " ---" << std::endl;
      auto current_token = tr.pollCurToken();
      while (true) {
        for (int k = 0; k < tr.current_frame - 1; k++)
          std::cout << "  ";
        tr.thread_trace->printToken(current_token);
        for (int k = 10; k > tr.current_frame - 1; k--)
          std::cout << "  ";
        if (current_token.type == pallas::TypeEvent) {
          auto occ = tr.getEventOccurence(current_token, tr.tokenCount[current_token]);
          std::cout << " : ";
          printEvent(tr.thread_trace, current_token, occ);
        } else if (current_token.type == pallas::TypeSequence) {
          printf("%lu ", tr.tokenCount[current_token]);
          printf("%21.9lf\n", tr.thread_trace->getSequence(current_token)->durations->at(tr.tokenCount[current_token]) / 1e9);
        } else if (current_token.type == pallas::TypeLoop) {
          printf("%lu ", tr.tokenCount[current_token]);
          printf("%21.9lf\n", tr.getLoopDuration(current_token) / 1e9);
        }
        auto next_token = tr.getNextToken(flags);
        if (!next_token.has_value())
          break;
        current_token = next_token.value();
      }
    }
  }
}


void usage(const char* prog_name) {
  std::cout << "Usage : " << prog_name << " [options] <trace file>" << std::endl;
  std::cout << "\t" << "-h" << "\t" << "Show this help and exit" << std::endl;
  std::cout << "\t" << "-v" << "\t" << "Enable verbose mode" << std::endl;
  std::cout << "\t" << "-T" << "\t" << "Enable per thread mode" << std::endl;
  std::cout << "\t" << "-d" << "\t" << "Show durations" << std::endl;
  std::cout << "\t" << "-t" << "\t" << "Hide timestamps" << std::endl;
  std::cout << "\t" << "-S" << "\t" << "Enable structure mode (per thread mode only)" << std::endl;
  std::cout << "\t" << "-s" << "\t" << "Do not unroll sequences (structure mode only)" << std::endl;
  std::cout << "\t" << "-l" << "\t" << "Do not unroll loops (structure mode only)" << std::endl;
}

int main(const int argc, char* argv[]) {
  int nb_opts = 0;
  int flags = PALLAS_READ_UNROLL_ALL;
  bool show_structure = false;

  for (nb_opts = 1; nb_opts < argc; nb_opts++) {
    if (!strcmp(argv[nb_opts], "-v")) {
      pallas_debug_level_set(pallas::DebugLevel::Debug);
    } else if (!strcmp(argv[nb_opts], "-h") || !strcmp(argv[nb_opts], "-?")) {
      usage(argv[0]);
      return EXIT_SUCCESS;
    } else if (!strcmp(argv[nb_opts], "-v")) {
      pallas_debug_level_set(pallas::DebugLevel::Debug);
    } else if (!strcmp(argv[nb_opts], "-T")) {
      per_thread = true;
    } else if (!strcmp(argv[nb_opts], "-d")) {
      show_durations = true;
    } else if (!strcmp(argv[nb_opts], "-t")) {
      show_timestamps = false;
    } else if (!strcmp(argv[nb_opts], "-S")) {
      show_structure = true;
    } else if (!strcmp(argv[nb_opts], "-s")) {
      flags &= ~PALLAS_READ_UNROLL_SEQUENCE;
    } else if (!strcmp(argv[nb_opts], "-l")) {
      flags &= ~PALLAS_READ_UNROLL_LOOP;
    } else {
      /* Unknown parameter name. It's probably the trace's path name. We can stop
       * parsing the parameter list.
       */
      break;
    }
  }

  char* trace_name = argv[nb_opts];
  if (trace_name == nullptr) {
    std::cout << "Missing trace file" << std::endl;
    usage(argv[0]);
    return EXIT_SUCCESS;
  }

  auto trace = pallas::GlobalArchive();
  pallasReadGlobalArchive(&trace, trace_name);

  if (show_structure)
    printStructure(flags, trace);
  else
    printTrace(trace);

  return EXIT_SUCCESS;
}


/* -*-
   mode: c;
   c-file-style: "k&r";
   c-basic-offset 2;
   tab-width 2 ;
   indent-tabs-mode nil
   -*- */

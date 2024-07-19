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

bool show_timestamps = false;
bool show_structure = false;

void usage(const char* prog_name) {
  std::cout << "Usage : " << prog_name << " [options] <trace file>" << std::endl;
  std::cout << "\t" << "-h" << "\t" << "Show this help and exit" << std::endl;
  std::cout << "\t" << "-v" << "\t" << "Be verbose" << std::endl;
  std::cout << "\t" << "-s" << "\t" << "Do not unroll sequences" << std::endl;
  std::cout << "\t" << "-l" << "\t" << "Do not unroll loops" << std::endl;
  std::cout << "\t" << "-t" << "\t" << "Show timestamps" << std::endl;
  std::cout << "\t" << "-S" << "\t" << "Show structure" << std::endl;
}

static void printEvent(const pallas::Thread* thread,
                       const pallas::EventOccurence* e) {
  thread->printEvent(e->event);
  thread->printEventAttribute(e);
  if (show_timestamps)
    printf(" (%.9lf)", e->timestamp / 1e9);
  std::cout << std::endl;
}

int main(const int argc, char* argv[]) {
  int nb_opts = 0;
  int flags = PALLAS_READ_UNROLL_ALL;

  for (nb_opts = 1; nb_opts < argc; nb_opts++) {
    if (!strcmp(argv[nb_opts], "-v")) {
      pallas_debug_level_set(pallas::DebugLevel::Debug);
    } else if (!strcmp(argv[nb_opts], "-h") || !strcmp(argv[nb_opts], "-?")) {
      usage(argv[0]);
      return EXIT_SUCCESS;
    } else if (!strcmp(argv[nb_opts], "-v")) {
      pallas_debug_level_set(pallas::DebugLevel::Debug);
    } else if (!strcmp(argv[nb_opts], "-s")) {
      flags &= ~PALLAS_READ_UNROLL_SEQUENCE;
    } else if (!strcmp(argv[nb_opts], "-l")) {
      flags &= ~PALLAS_READ_UNROLL_LOOP;
    } else if (!strcmp(argv[nb_opts], "-t")) {
      show_timestamps = true;
    } else if (!strcmp(argv[nb_opts], "-S")) {
      show_structure = true;
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

  constexpr int reader_options = pallas::ThreadReaderOptions::None;
  for (int i = 0; i < trace.nb_archives; i++) {
    for (int j = 0; j < trace.archive_list[i]->nb_threads; j ++) {
      std::cout << "--- Thread " << j << " ---" << std::endl;
      auto tr = pallas::ThreadReader(trace.archive_list[i], trace.archive_list[i]->threads[j]->id, reader_options);
      auto current_token = tr.pollCurToken();
      while (true) {
        if (show_structure) {
          for (int k = 0; k < tr.current_frame - 1; k++)
            std::cout << "  ";
          tr.thread_trace->printToken(current_token);
          if (current_token.type != pallas::TypeEvent) {
            std::cout << std::endl;
          } else {
            auto occ = tr.getEventOccurence(current_token, tr.tokenCount[current_token]);
            std::cout << " : ";
            printEvent(tr.thread_trace, &occ);
          }
        } else {
          if (current_token.type == pallas::TypeEvent) {
            auto occ = tr.getEventOccurence(current_token, tr.tokenCount[current_token]);
            printEvent(tr.thread_trace, &occ);
          }
        }
        auto next_token = tr.getNextToken(flags);
        if (!next_token.has_value())
          break;
        current_token = next_token.value();
      }
    }
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

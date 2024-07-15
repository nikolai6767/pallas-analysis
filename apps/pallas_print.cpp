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

void usage(const char* prog_name) {
  std::cout << "Usage : Go read source code you dum dum" << std::endl;
}

static void printEvent(const pallas::Thread* thread,
                       const pallas::EventOccurence* e) {
  thread->printEvent(e->event);
  thread->printEventAttribute(e);
  std::cout << std::endl;
}

int main(const int argc, char* argv[]) {
  int nb_opts = 0;

  for (nb_opts = 1; nb_opts < argc; nb_opts++) {
    if (!strcmp(argv[nb_opts], "-v")) {
      pallas_debug_level_set(pallas::DebugLevel::Debug);
    }else if (!strcmp(argv[nb_opts], "-h") || !strcmp(argv[nb_opts], "-?")) {
      usage(argv[0]);
      return EXIT_SUCCESS;
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

  auto trace = pallas::Archive();
  pallas_read_main_archive(&trace, trace_name);

  int reader_options = pallas::ThreadReaderOptions::None;
  for (int i = 0; i < trace.nb_archives; i++) {
    for (int j = 0; j < 1 /*trace.archive_list[i]->nb_threads*/; j ++) {
      std::cout << "--- Thread " << j << " ---" << std::endl;
      auto tr = pallas::ThreadReader(trace.archive_list[i], trace.archive_list[i]->threads[j]->id, reader_options);
      tr.thread_trace->printSequence(tr.callstack_iterable[1]);
      auto currentToken = tr.pollCurToken();
      auto nextToken = tr.getNextToken(PALLAS_READ_UNROLL_ALL);
      while (nextToken.has_value()) {
        for (int k = 0; k < tr.current_frame - 1; k++)
          std::cout << "  ";
        tr.thread_trace->printToken(currentToken);
        std::cout << std::endl;
        currentToken = nextToken.value();
        nextToken = tr.getNextToken(PALLAS_READ_UNROLL_ALL);
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

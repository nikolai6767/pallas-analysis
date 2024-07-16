/*
 * Copyright (C) Telecom SudParis
 * See LICENSE in top-level directory.
 */
#include <algorithm>
#include <iomanip>
#include <iostream>
#include <string>
#include "pallas/pallas.h"
#include "pallas/pallas_archive.h"
#include "pallas/pallas_read.h"
#include "pallas/pallas_storage.h"

static void testSequenceDuration(pallas::ThreadReader* reader,
                                 const pallas::Token token,
                                 pallas::SequenceOccurence* sequenceOccurence,
                                 const int depth,
                                 const pallas_duration_t givenDuration) {
  if (sequenceOccurence) {
    reader->loadSavestate(sequenceOccurence->savestate);
    reader->enterBlock(token);
  }
  const auto currentLevel = reader->readCurrentLevel();
  for (const auto& [token, occurence] : currentLevel) {
    switch (token->type) {
    case pallas::TypeSequence: {
      testSequenceDuration(reader, *token, &occurence->sequence_occurence, depth + 1,
                           occurence->sequence_occurence.duration);
      break;
    }
    case pallas::TypeLoop: {
      const pallas::LoopOccurence& loop = occurence->loop_occurence;
      for (uint j = 0; j < loop.nb_iterations; j++) {
        pallas::SequenceOccurence* seq = &loop.full_loop[j];
        testSequenceDuration(reader, loop.loop->repeated_token, seq, depth + 2, seq->duration);
      }
      break;
    }
    default: {
      break;
    };
    }
  }
  if (sequenceOccurence) {
    sequenceOccurence->full_sequence = new pallas::TokenOccurence[currentLevel.size()];
    // memcpy(sequenceOccurence->full_sequence, currentLevel.data(), currentLevel.size() * sizeof(pallas::TokenOccurence));
  }
  std::cout << "Sequence " << token.id << ":\n";
  long double actualDuration = 0;
  for (const auto& [token, occurence] : currentLevel) {
    switch (token->type) {
    case pallas::TypeEvent: {
      std::cout << "\tE" << std::setw(7) << std::left <<  token->id
                << std::setw(7) << std::right << occurence->event_occurence.duration << "\n";
      actualDuration += occurence->event_occurence.duration;
      break;
    }
    case pallas::TypeSequence: {
      std::cout << "\tS" << std::setw(7) << std::left <<  token->id
                << std::setw(7) << std::right << occurence->sequence_occurence.duration << "\n";
      actualDuration += occurence->sequence_occurence.duration;
      break;
    }
    case pallas::TypeLoop: {
      std::cout << "\tL" << std::setw(7) << std::left <<  token->id
                << std::setw(7) << std::right << occurence->loop_occurence.duration << "\n";
      actualDuration += occurence->loop_occurence.duration;
      break;
    }
    default:
      pallas_error("This should not have happened\n");
    }
  }
  std::cout << "\tSum   \t" << std::setw(7) << std::right << actualDuration << "\n";
  std::cout << "\tStored\t" << std::setw(7) << std::right << givenDuration
            << "\tDiff = " << (actualDuration - givenDuration) * 100 / (givenDuration) << "%" << std::endl;
  reader->leaveBlock();
}

/* Print all the events of a thread */
static void testThreadDuration(const pallas::Archive& trace, const pallas::Thread& thread) {
  printf("Testing durations for Thread %u (%s):\n", thread.id, thread.getName());

  constexpr int readerOptions = pallas::ThreadReaderOptions::None;
  auto* reader = new pallas::ThreadReader(&trace, thread.id, readerOptions);

  testSequenceDuration(reader, PALLAS_SEQUENCE_ID(0), nullptr, 0, reader->thread_trace->sequences[0]->durations->front());
}

void usage(const char* prog_name) {
  printf("Usage: %s [OPTION] trace_file\n", prog_name);
  printf("\t-v          Verbose mode\n");
  printf("\t-?  -h      Display this help and exit\n");
}

int main(const int argc, char* argv[]) {
  int nb_opts = 0;
  char* trace_name = nullptr;

  for (int i = 1; i < argc; i++) {
    if (!strcmp(argv[i], "-v")) {
      pallas_debug_level_set(pallas::DebugLevel::Debug);
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
  if (trace_name == nullptr) {
    usage(argv[0]);
    return EXIT_SUCCESS;
  }

  auto trace = pallas::GlobalArchive();
  pallasReadGlobalArchive(&trace, trace_name);

  for (int i = 0; i < trace.nb_archives; i++) {
    for (int j = 0; j < trace.archive_list[i]->nb_threads; j ++) {

      printf("\n");
      testThreadDuration(*trace.archive_list[i], *trace.archive_list[i]->threads[j]);
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

/*
 * Copyright (C) Telecom SudParis
 * See LICENSE in top-level directory.
 */
#include <cstdlib>
#include <cstring>
#include <iomanip>
#if __GNUC__ >= 13 || __clang__ >= 14 || _MSC_VER >= 1929
#include <format>
#define HAS_FORMAT
#endif
#include "pallas/pallas.h"
#include "pallas/pallas_archive.h"
#include "pallas/pallas_log.h"
#include "pallas/pallas_read.h"
#include "pallas/pallas_storage.h"

#define DURATION_WIDTH 15

using namespace pallas;
void print_sequence(const Sequence* s, const Thread* t) {
  std::cout << "{";
  for (unsigned i = 0; i < s->size(); i++) {
    const Token token = s->tokens[i];
    std::cout << t->getTokenString(token);
    if (i < s->size() - 1)
      printf(", ");
  }
  std::cout << "}" << std::endl;
}

void info_event(Thread* t, EventSummary* e) {
  pallas_print_event(t, &e->event);
  std::cout << "\t{.nb_events: " << e->durations->size << "}" << std::endl;
}

void info_sequence(Sequence* s) {
  std::cout << "\t{.size: " << s->size() << "}" << std::endl;
}

void info_loop(Loop* l, Thread* t) {
  std::cout << "{.nb_loops: " << l->nb_iterations.size()
            << ", .repeated_token: " << t->getTokenString(l->repeated_token) << ", .nb_iterations: [";
  for (const auto& i : l->nb_iterations) {
    std::cout << i;
    if (&i != &l->nb_iterations.back()) {
      std::cout << ", ";
    }
  }
  std::cout << "]}" << std::endl;
}

void info_thread(Thread* t) {
  printf("Thread %d {.archive: %d}\n", t->id, t->archive->id);
  printf("\tEvents {.nb_events: %d}\n", t->nb_events);
  for (unsigned i = 0; i < t->nb_events; i++) {
    printf("\t\tE%d\t", i);
    info_event(t, &t->events[i]);
  }

  printf("\tSequences {.nb_sequences: %d}\n", t->nb_sequences);
  for (unsigned i = 0; i < t->nb_sequences; i++) {
    std::cout << "\t\tS" << i << "\t" << t->sequences[i]->durations->size << " x ";
    print_sequence(t->sequences[i], t);
    if (t->sequences[i]->durations->size > 1) {
      std::cout.precision(9);
      std::cout << "\t\t\tMin:  " << std::fixed << std::setw(DURATION_WIDTH) << t->sequences[i]->durations->min / 1e9
                << "\n\t\t\tMax:  " << std::fixed << std::setw(DURATION_WIDTH) << t->sequences[i]->durations->max / 1e9
                << "\n\t\t\tMean: " << std::fixed << std::setw(DURATION_WIDTH) << t->sequences[i]->durations->mean / 1e9
                << std::endl;
    } else {
      std::cout << "\t\t\tDuration: " << std::fixed << std::setw(DURATION_WIDTH)
                << t->sequences[i]->durations->front() / 1e9 << std::endl;
    }
  }

  printf("\tLoops {.nb_loops: %d}\n", t->nb_loops);
  for (unsigned i = 0; i < t->nb_loops; i++) {
    printf("\t\tL%d\t", i);
    info_loop(&t->loops[i], t);
  }
}

void info_global_archive(GlobalArchive* archive) {
  printf("Main archive:\n");
  printf("\tdir_name:   %s\n", archive->dir_name);
  printf("\ttrace_name: %s\n", archive->trace_name);
  printf("\tfullpath:   %s\n", archive->fullpath);
  if (!archive->definitions.strings.empty())
    printf("\tStrings {.nb_strings: %zu } :\n", archive->definitions.strings.size());

  for (auto& [stringRef, string] : archive->definitions.strings) {
    printf("\t\t%d: '%s'\n", string.string_ref, string.str);
  }
  if (!archive->definitions.regions.empty())
    printf("\tRegions {.nb_regions: %zu } :\n", archive->definitions.regions.size());
  for (auto& [regionRef, region] : archive->definitions.regions) {
    printf("\t\t%d: %s\n", region.region_ref, archive->getString(region.string_ref)->str);
  }

  if (!archive->location_groups.empty())
    printf("\tLocation_groups {.nb_lg: %zu }:\n", archive->location_groups.size());
  for (auto& locationGroup : archive->location_groups) {
    printf("\t\t%d: %s", locationGroup.id, archive->getString(locationGroup.name)->str);
    if (locationGroup.parent != PALLAS_LOCATION_GROUP_ID_INVALID)
      printf(", parent: %d", locationGroup.parent);
    if (locationGroup.mainLoc != PALLAS_THREAD_ID_INVALID)
      printf(", mainLocation: %d", locationGroup.mainLoc);
    printf("\n");
  }

  if (!archive->locations.empty())
    printf("\tLocations {.nb_loc: %zu }:\n", archive->locations.size());
  for (auto location : archive->locations) {
    printf("\t\t%d: %s, parent: %d\n", location.id, archive->getString(location.name)->str, location.parent);
  }
  if (archive->nb_archives)
    printf("\tArchives {.nb_archives: %d}\n", archive->nb_archives);

  printf("\n");
}

void info_archive(Archive* archive) {
  printf("Archive %d:\n", archive->id);

  if (archive->nb_threads)
    printf("\tThreads {.nb_threads: %d}:\n", archive->nb_threads);
  if (archive->threads) {
    for (int i = 0; i < archive->nb_threads; i++) {
      auto thread = archive->getThreadAt(i);
      if (thread) {
        printf("\t\t%d: {.archive=%d, .nb_events=%d, .nb_sequences=%d, .nb_loops=%d}\n", thread->id,
               thread->archive->id, thread->nb_events, thread->nb_sequences, thread->nb_loops);
      }
    }
  }
  printf("\n");
}

void info_trace(GlobalArchive* trace) {
  info_global_archive(trace);
  for (int i = 0; i < trace->nb_archives; i++) {
    info_archive(trace->archive_list[i]);
  }

  for (int i = 0; i < trace->nb_archives; i++) {
    for (int j = 0; j < trace->archive_list[i]->nb_threads; j++) {
      auto thread = trace->archive_list[i]->getThreadAt(j);
      if (thread)
        info_thread(thread);
    }
  }
}

void usage(const char* prog_name) {
  printf("Usage: %s [OPTION] trace_file\n", prog_name);
  printf("\t-v          Verbose mode\n");
  printf("\t-?  -h --help     Display this help and exit\n");
}

int main(int argc, char** argv) {
  int nb_opts = 0;
  char* trace_name = nullptr;

  for (int i = 1; i < argc; i++) {
    if (!strcmp(argv[i], "-v")) {
      pallas_debug_level_set(DebugLevel::Debug);
      nb_opts++;
    } else if (!strcmp(argv[i], "-h") || !strcmp(argv[i], "-?") || !strcmp(argv[i], "--help")) {
      usage(argv[0]);
      return EXIT_SUCCESS;
    } else {
      /* Unknown parameter name. It's probably the program name. We can stop
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

  auto trace = GlobalArchive();
  pallasReadGlobalArchive(&trace, trace_name);
  info_trace(&trace);

  return EXIT_SUCCESS;
}

/* -*-
   mode: c;
   c-file-style: "k&r";
   c-basic-offset 2;
   tab-width 2 ;
   indent-tabs-mode nil
   -*- */

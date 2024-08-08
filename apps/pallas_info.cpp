/*
 * Copyright (C) Telecom SudParis
 * See LICENSE in top-level directory.
 */
#include <cstdlib>
#include <cstring>
#if __GNUC__ >= 13 || __clang__ >= 14 || _MSC_VER >= 1929
#include <format>
#define HAS_FORMAT
#endif
#include "pallas/pallas.h"
#include "pallas/pallas_archive.h"
#include "pallas/pallas_read.h"
#include "pallas/pallas_storage.h"
#include "pallas/pallas_log.h"

using namespace pallas;
void print_sequence(const Sequence* s) {
  printf("{");
  for (unsigned i = 0; i < s->size(); i++) {
    const Token token = s->tokens[i];
    printf("%c%d", PALLAS_TOKEN_TYPE_C(token), token.id);
    if (i < s->size() - 1)
      printf(", ");
  }
  printf("}\n");
}

void info_event(Thread* t, EventSummary* e) {
  pallas_print_event(t, &e->event);
  printf("\t{.nb_events: %zu}\n", e->durations->size);
}

void info_sequence(Sequence* s) {
  printf("{.size: %zu}\n", pallas_sequence_get_size(s));
}

void info_loop(Loop* l) {
  printf("{.nb_loops: %zu, .repeated_token: %c%d, .nb_iterations: ", l->nb_iterations.size(),
         PALLAS_TOKEN_TYPE_C(l->repeated_token), l->repeated_token.id);
  printf("[");
  for (auto& i : l->nb_iterations) {
    printf("%u", i);
    if (&i != &l->nb_iterations.back()) {
      printf(", ");
    }
  }
  printf("]}\n");
}

#ifdef HAS_FORMAT
#define UINT64_FILTER(d) \
  ((d == UINT64_MAX) ? "INVALID_MAX" : (d == 0) ? "INVALID_MIN" : std::format("{:>21.9}", d / 1e9))
#else
#define UINT64_FILTER(d) ((d == UINT64_MAX) ? "INVALID_MAX" : (d == 0) ? "INVALID_MIN" : std::to_string(d / 1e9))
#endif
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
    print_sequence(t->sequences[i]);
    if (t->sequences[i]->durations->size > 1) {
      std::cout << "\t\t\tMin:  " << UINT64_FILTER(t->sequences[i]->durations->min)
                << "\n\t\t\tMax:  " << UINT64_FILTER(t->sequences[i]->durations->max)
                << "\n\t\t\tMean: " << UINT64_FILTER(t->sequences[i]->durations->mean) << std::endl;
    } else {
      std::cout << "\t\t\tDuration: " << t->sequences[i]->durations->front() << std::endl;
    }
  }

  printf("\tLoops {.nb_loops: %d}\n", t->nb_loops);
  for (unsigned i = 0; i < t->nb_loops; i++) {
    printf("\t\tL%d\t", i);
    info_loop(&t->loops[i]);
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
  if (!archive->definitions.groups.empty())
    printf("\tGroups {.nb_groups: %zu } :\n", archive->definitions.groups.size());
  for (auto& [groupRef, group] : archive->definitions.groups) {
    printf("\t\t%d: '%s' [", group.group_ref, archive->getString(group.name)->str);
    for(uint32_t i = 0; i<group.numberOfMembers; i++) {
      printf("%s%lu", i>0?", ":"", group.members[i]);
    }
    printf("]\n");
  }
  if (!archive->definitions.comms.empty())
    printf("\tComms {.nb_comms: %zu } :\n", archive->definitions.comms.size());
  for (auto& [commRef, comm] : archive->definitions.comms) {
    printf("\t\t%d: '%s' (group, %d, parent: %d) \n", comm.comm_ref, archive->getString(comm.name)->str, comm.group, comm.parent);
  }

  if (!archive->location_groups.empty())
    printf("\tLocation_groups {.nb_lg: %zu }:\n", archive->location_groups.size());
  for (auto& locationGroup: archive->location_groups) {
    printf("\t\t%d: %s", locationGroup.id,
           archive->getString(locationGroup.name)->str);
    if (locationGroup.parent != PALLAS_LOCATION_GROUP_ID_INVALID)
      printf(", parent: %d", locationGroup.parent);
    if (locationGroup.mainLoc != PALLAS_THREAD_ID_INVALID)
      printf(", mainLocation: %d", locationGroup.mainLoc);
    printf("\n");
  }

  if (!archive->locations.empty())
    printf("\tLocations {.nb_loc: %zu }:\n", archive->locations.size());
  for (auto location: archive->locations) {
    printf("\t\t%d: %s, parent: %d\n", location.id,
           archive->getString(location.name)->str, location.parent);
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
      if (archive->threads[i]) {
        printf("\t\t%d: {.archive=%d, .nb_events=%d, .nb_sequences=%d, .nb_loops=%d}\n", archive->threads[i]->id,
               archive->threads[i]->archive->id, archive->threads[i]->nb_events, archive->threads[i]->nb_sequences,
               archive->threads[i]->nb_loops);
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
      if (trace->archive_list[i]->threads[j])
      info_thread(trace->archive_list[i]->threads[j]);
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

  auto trace = GlobalArchive ();
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

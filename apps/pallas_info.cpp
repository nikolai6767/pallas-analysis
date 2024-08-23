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

bool show_definitions = false;
bool show_details     = false;
int thread_to_print   = -1;

bool show_archives    = true;
bool show_threads     = true;

static bool _should_print_thread(int thread_id) {
  return thread_to_print <0 || thread_to_print == thread_id;
}

static double ns2s(uint64_t ns) {
  return ns*1.0/1e9;
}

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

  if(! _should_print_thread(t->id))
    return;

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
  if (show_details) {
    printf("\tdir_name:   %s\n", archive->dir_name);
    printf("\ttrace_name: %s\n", archive->trace_name);
  }

  printf("\tfullpath:    %s\n", archive->fullpath);
  printf("\tnb_archives: %d\n", archive->nb_archives);
  printf("\tnb_process: %lu\n", archive->location_groups.size());
  printf("\tnb_threads: %lu\n", archive->locations.size());

  if(show_definitions) {
    if (!archive->definitions.strings.empty()) {
      printf("\tStrings {.nb_strings: %zu } :\n", archive->definitions.strings.size());

      for (auto& [stringRef, string] : archive->definitions.strings) {
	printf("\t\t%d: '%s'\n", string.string_ref, string.str);
      }
    }

    if (!archive->definitions.regions.empty()) {
      printf("\tRegions {.nb_regions: %zu } :\n", archive->definitions.regions.size());
      for (auto& [regionRef, region] : archive->definitions.regions) {
	printf("\t\t%d: %s\n", region.region_ref, archive->getString(region.string_ref)->str);
      }
    }

    if (!archive->definitions.groups.empty()) {
      printf("\tGroups {.nb_groups: %zu } :\n", archive->definitions.groups.size());
      for (auto& [groupRef, group] : archive->definitions.groups) {
	printf("\t\t%d: '%s' [", group.group_ref, archive->getString(group.name)->str);
	for(uint32_t i = 0; i<group.numberOfMembers; i++) {
	  printf("%s%lu", i>0?", ":"", group.members[i]);
	}
	printf("]\n");
      }
    }

    if (!archive->definitions.comms.empty()) {
      printf("\tComms {.nb_comms: %zu } :\n", archive->definitions.comms.size());
      for (auto& [commRef, comm] : archive->definitions.comms) {
	printf("\t\t%d: '%s' (group, %d, parent: %d) \n", comm.comm_ref, archive->getString(comm.name)->str, comm.group, comm.parent);
      }
    }

    if (!archive->location_groups.empty()) {
      printf("\tLocation_groups {.nb_lg: %zu }:\n", archive->location_groups.size());
      for (auto& locationGroup : archive->location_groups) {
	printf("\t\t%d: %s", locationGroup.id, archive->getString(locationGroup.name)->str);
	if (locationGroup.parent != PALLAS_LOCATION_GROUP_ID_INVALID)
	  printf(", parent: %d", locationGroup.parent);
	if (locationGroup.mainLoc != PALLAS_THREAD_ID_INVALID)
	  printf(", mainLocation: %d", locationGroup.mainLoc);
	printf("\n");
      }
    }

    if (!archive->locations.empty()) {
      printf("\tLocations {.nb_loc: %zu }:\n", archive->locations.size());
      for (auto location : archive->locations) {
	printf("\t\t%d: %s, parent: %d\n", location.id, archive->getString(location.name)->str, location.parent);
      }
    }

    if (archive->nb_archives)
      printf("\tArchives {.nb_archives: %d}\n", archive->nb_archives);
  }

  printf("\n");
}

static bool _archiveContainsThread(Archive* archive, int thread_id) {
  for(int i=0; i< archive->nb_threads; i++) {
    auto thread = archive->getThreadAt(i);
    if(thread->id == thread_id)
      return true;
  }
  return false;
}

void info_archive_header() {
  std::cout<< std::endl;
  std::cout<< "#";
  std::cout << std::setw(14) <<"Archive_id";
  std::cout << std::setw(20) <<"Archive_name";
  std::cout << std::setw(15) <<"Nb_threads";
  std::cout << std::endl;
}

void info_archive(Archive* archive) {
  std::cout << std::setw(15) << archive->id;
  std::cout << std::setw(20) << archive->getName();
  std::cout << std::setw(15) << archive->nb_threads;
  std::cout << std::endl;
}

void info_threads_header() {
  std::cout<< std::endl;
  std::cout<< "#";
  std::cout << std::setw(19) <<"Thread_name";
  std::cout << std::setw(15) <<"Thread_id";
  std::cout << std::setw(15) <<"First_ts";
  std::cout << std::setw(15) <<"Last_ts";
  std::cout << std::setw(15) <<"Duration(s)";
  std::cout << std::setw(15) <<"Event_count";
  std::cout << std::setw(15) <<"Nb_events";
  std::cout << std::setw(15) <<"Nb_sequences";
  std::cout << std::setw(15) <<"Nb_loops";
  std::cout << std::setw(15) <<"Archive_id";
  std::cout << std::endl;
}

void info_threads(Archive* archive) {
  if(thread_to_print >= 0 && ! _archiveContainsThread(archive, thread_to_print)) {
    return;
  }

  if (archive->threads) {
    for (int i = 0; i < archive->nb_threads; i++) {
      auto thread = archive->getThreadAt(i);
      if (thread && _should_print_thread(thread->id)) {
	std::cout << std::setw(20) << thread->getName();
	std::cout << std::setw(15) << thread->id;
	std::cout << std::setw(15) << thread->getFirstTimestamp();
	std::cout << std::setw(15) << thread->getLastTimestamp();
	std::cout << std::setw(15) << ns2s(thread->getDuration());
	std::cout << std::setw(15) << thread->getEventCount();
	std::cout << std::setw(15) << thread->nb_events;
	std::cout << std::setw(15) << thread->nb_sequences;
	std::cout << std::setw(15) << thread->nb_loops;
	std::cout << std::setw(15) << archive->id;
	std::cout << std::endl;
      }
    }
  }

}

void info_trace(GlobalArchive* trace) {
  info_global_archive(trace);

  if(show_archives) {
    info_archive_header();
    for (int i = 0; i < trace->nb_archives; i++) {
      info_archive(trace->archive_list[i]);
    }
  }

  if(show_threads) {
    info_threads_header();
    for (int i = 0; i < trace->nb_archives; i++) {
      info_threads(trace->archive_list[i]);
    }
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
  printf("\t-v             Verbose mode\n");
  printf("\t-D             Show definitions\n");
  printf("\t--thread tid   Only print thread <tid>\n");
  printf("\t-?  -h --help     Display this help and exit\n");
}

int main(int argc, char** argv) {
  int nb_opts = 0;
  char* trace_name = nullptr;

  for (nb_opts = 1; nb_opts < argc; nb_opts++) {
    if (!strcmp(argv[nb_opts], "-v")) {
      pallas_debug_level_set(DebugLevel::Debug);
      show_details = true;
    } else if (!strcmp(argv[nb_opts], "-D")) {
      show_definitions = true;
    } else if (!strcmp(argv[nb_opts], "--thread")) {
      thread_to_print = atoi(argv[nb_opts+1]);
      printf("thread_to_print=%d\n", thread_to_print);
      nb_opts++;
    } else if (!strcmp(argv[nb_opts], "-h") || !strcmp(argv[nb_opts], "-?") || !strcmp(argv[nb_opts], "--help")) {
      printf("invalid option '%s'\n", argv[nb_opts]);
      usage(argv[0]);
      return EXIT_SUCCESS;
    } else {
      /* Unknown parameter name. It's probably the program name. We can stop
       * parsing the parameter list.
       */
      trace_name = argv[nb_opts];
      break;
    }
  }

  if (trace_name == nullptr) {
    printf("Missing trace file\n");
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

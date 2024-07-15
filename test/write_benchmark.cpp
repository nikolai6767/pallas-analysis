/*
 * Copyright (C) Telecom SudParis
 * See LICENSE in top-level directory.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <atomic>
#include <sstream>
#if __GNUC__ >= 13 || __clang__ >= 14 || _MSC_VER >= 1929
#include <format>
#define HAS_FORMAT
#endif

#include "pallas/pallas.h"
#include "pallas/pallas_archive.h"
#include "pallas/pallas_write.h"
#include "pallas/pallas_record.h"

using namespace pallas;

static Archive globalArchive;
static Archive mainProcess;
static LocationGroupId processID;
static StringRef processName;

static int nb_iter_default = 100000;
static int nb_functions_default = 2;
static int nb_threads_default = 4;
static int pattern_default = 0;
static int use_logical_clock_default = 0;

static int nb_iter;
static int nb_functions;
static int nb_threads;
static int pattern;
static int use_logical_clock;

std::vector<ThreadWriter> thread_writers;
std::vector<RegionRef> regions;
std::vector<std::string> region_names;
std::vector<StringRef> strings;

static pthread_barrier_t bench_start;
static pthread_barrier_t bench_stop;

#define TIME_DIFF(t1, t2) (((t2).tv_sec - (t1).tv_sec) + ((t2).tv_nsec - (t1).tv_nsec) / 1e9)

static StringRef _register_string(const char* str) {
  static std::atomic<StringRef> next_ref = 0;
  StringRef ref = next_ref++;
  globalArchive.addString(ref, str);
  return ref;
}

static LocationGroupId _new_location_group() {
  static std::atomic<LocationGroupId> next_id = 0;
  LocationGroupId id = next_id++;
  return id;
}


static ThreadId _new_thread() {
  static std::atomic<ThreadId> next_id = 0;
  ThreadId id = next_id++;
  return id;
}

static pallas_timestamp_t get_timestamp() {
  pallas_timestamp_t res = PALLAS_TIMESTAMP_INVALID;

  if(use_logical_clock) {
      static int next_ts = 1;
      res = next_ts++;
  }
  return res;
}

void* worker(void* arg __attribute__((unused))) {
  pthread_mutex_lock(&globalArchive.lock);
  ThreadId threadID = _new_thread();
  auto threadWriter = ThreadWriter();

#ifdef HAS_FORMAT
  StringRef threadNameRef = _register_string(std::format("thread_{}", threadID).c_str());
#else
  std::ostringstream os("thread_");
  os << threadID;
  StringRef threadNameRef = _register_string(os.str().c_str());
#endif
  globalArchive.defineLocation(threadID, threadNameRef, processID);
  threadWriter.open(&mainProcess, threadID);

  struct timespec t1, t2;
  pthread_barrier_wait(&bench_start);
  clock_gettime(CLOCK_MONOTONIC, &t1);
  for (int i = 0; i < nb_iter; i++) {
    switch (pattern) {
    case 0:
      /* each iteration is a sequence of functions:
       * E_f1 L_f1 E_f2 L_f2 E_f3 L_f3 ...
       */
      for (int j = 0; j < nb_functions; j++) {
        pallas_record_generic(&threadWriter, NULL, get_timestamp(), strings[j]);
      }
      break;

    case 1:
      /* each iteration contains recursive function calls::
       * E_f1 E_f2 E_f3 ... L_f3 L_f2 L_f1
       */
      for (int j = 0; j < nb_functions; j++) {
        pallas_record_enter(&threadWriter, NULL, get_timestamp(), regions[j]);
      }
      for (int j = nb_functions - 1; j >= 0; j--) {
        pallas_record_leave(&threadWriter, NULL, get_timestamp(), regions[j]);
      }
      break;
    default:
      fprintf(stderr, "invalid pattern: %d\n", pattern);
    }
  }
  clock_gettime(CLOCK_MONOTONIC, &t2);

  pthread_barrier_wait(&bench_stop);

  double duration = TIME_DIFF(t1, t2);
  int nb_event_per_iter = 2 * nb_functions;
  int nb_events = nb_iter * nb_event_per_iter;
  double duration_per_event = duration / nb_events;

  printf("T#%d: %d events in %lf s -> %lf ns per event\n", threadID, nb_events, duration, duration_per_event * 1e9);

  threadWriter.threadClose();
  return NULL;
}

void usage(const char* prog_name) {
  printf("Usage: %s [OPTION] program [arg1 arg2 ...]\n", prog_name);
  printf("\t-n X    Set the number of iterations (default: %d)\n", nb_iter_default);
  printf("\t-f X    Set the number of functions (default: %d)\n", nb_functions_default);
  printf("\t-t X    Set the number of threads (default: %d)\n", nb_threads_default);
  printf("\t-p X    Select the event pattern\n");
  printf("\t-l      Use a per-thread logical clock instead of the default clock (default: %d)\n", use_logical_clock_default);

  printf("\t-? -h   Display this help and exit\n");
}

int main(int argc, char** argv) {
  int nb_opts = 0;  // options

  nb_iter = nb_iter_default;
  nb_functions = nb_functions_default;
  nb_threads = nb_threads_default;
  pattern = pattern_default;

  for (int i = 1; i < argc; i++) {
    if (!strcmp(argv[i], "-n")) {
      nb_iter = atoi(argv[i + 1]);
      nb_opts += 2;
      i++;
    } else if (!strcmp(argv[i], "-f")) {
      nb_functions = atoi(argv[i + 1]);
      nb_opts += 2;
      i++;
    } else if (!strcmp(argv[i], "-t")) {
      nb_threads = atoi(argv[i + 1]);
      nb_opts += 2;
      i++;
    } else if (!strcmp(argv[i], "-p")) {
      pattern = atoi(argv[i + 1]);
      nb_opts += 2;
      i++;
    } else if (!strcmp(argv[i], "-l")) {
      use_logical_clock = 1;
      nb_opts += 1;
    } else if (!strcmp(argv[i], "-?") || !strcmp(argv[i], "-h")) {
      usage(argv[0]);
      return EXIT_SUCCESS;
    } else {
      fprintf(stderr, "invalid option: %s\n", argv[i]);
      usage(argv[0]);
      return EXIT_FAILURE;
    }
  }

  pthread_barrier_init(&bench_start, nullptr, nb_threads + 1);
  pthread_barrier_init(&bench_stop, nullptr, nb_threads + 1);

  printf("nb_iter = %d\n", nb_iter);
  printf("nb_functions = %d\n", nb_functions);
  printf("nb_threads = %d\n", nb_threads);
  printf("pattern = %d\n", pattern);
  printf("---------------------\n");

  globalArchive = Archive();
  mainProcess = Archive();
  mainProcess.global_archive = &globalArchive;
  globalArchive.globalOpen("write_benchmark_CPP_trace", "main");

  processID = _new_location_group();
  processName = _register_string("Main process");
  globalArchive.defineLocationGroup(processID, processName, 0);

  mainProcess.open("write_benchmark_CPP_trace", "main", 0);

  for (int i = 0; i < nb_functions; i++) {
    std::ostringstream os;
    os << "function_" << i;
    region_names.push_back(os.str());
    os.clear();
    strings.push_back(_register_string(region_names.back().c_str()));
    regions.push_back(strings.back());
    mainProcess.addRegion(regions.back(), strings.back());
  }

  std::vector<pthread_t> threadID;
  for (int i = 0; i < nb_threads; i++) {
    pthread_t tid;
    pthread_create(&tid, nullptr, worker, nullptr);
    threadID.push_back(tid);
  }

  struct timespec t1, t2;
  pthread_barrier_wait(&bench_start);

  clock_gettime(CLOCK_MONOTONIC, &t1);
  pthread_barrier_wait(&bench_stop);
  clock_gettime(CLOCK_MONOTONIC, &t2);

  for (int i = 0; i < nb_threads; i++)
    pthread_join(threadID[i], NULL);

  double duration = TIME_DIFF(t1, t2);
  int nb_event_per_iter = 2 * nb_functions;
  int nb_events = nb_iter * nb_event_per_iter * nb_threads;
  double events_per_second = nb_events / duration;

  printf("TOTAL: %d events in %lf s -> %lf Me/s \n", nb_events, duration, events_per_second / 1e6);

  mainProcess.close();
  globalArchive.close();
  return EXIT_SUCCESS;
}

/* -*-
   mode: c;
   c-file-style: "k&r";
   c-basic-offset 2;
   tab-width 2 ;
   indent-tabs-mode nil
   -*- */

/*
 * Copyright (C) Telecom SudParis
 * See LICENSE in top-level directory.
 */
/** @file
 * Debug macros, debug levels, all that stuff.
 */
#pragma once
#ifndef NDEBUG
/** This macro only exists in Debug mode.*/
#define DEBUG
#endif

#ifdef __cplusplus
/** Only exists in C++. */
#define CXX(cxx_name) cxx_name
/** Only exists in C. */
#define C(c_name)
#else
/** Only exists in C++. */
#define CXX(cxx_name)
/** Only exists in C. */
#define C(c_name) c_name
#endif

/** A macro to help naming conventions in C/C++. First argument is only kept in C, second is only kept in C++. */
#define C_CXX(c_name, cxx_name) C(c_name) CXX(cxx_name)
/** Adds pallas:: in front of the variables in C++. */
#define PALLAS(something) CXX(pallas::) something

#ifdef __cplusplus
#include <pthread.h>
#include <cstdio>
#include <cstdlib>
#else
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#endif
#ifdef __cplusplus
namespace pallas {
#endif

/**
 * Enum to handle the different level of logging.
 */
enum CXX(class) DebugLevel {
  Error,   /**< Only print errors. */
  Quiet,   /**< Only print important messages. */
  Normal,  /**< Default verbosity level. */
  Verbose, /**< Print additional information. */
  Debug,   /**< Print much information. */
  Help,    /**< Print the different verbosity level and exit. */
  Max,     /**< Flood stdout with debug messages. */
};
#ifdef __cplusplus
extern enum DebugLevel debugLevel;
}; /* namespace pallas */
extern "C" {
#endif
/** Initializes the DebugLevel using Env Variables. */
extern void pallas_debug_level_init(void);
/** Sets the DebugLevel to the given level. */
extern void pallas_debug_level_set(enum PALLAS(DebugLevel) lvl);
/** Returns the DebugLevel. */
extern enum PALLAS(DebugLevel) pallas_debug_level_get(void);
CXX(
};)

/* -*-
   mode: c;
   c-file-style: "k&r";
   c-basic-offset 2;
   tab-width 2 ;
   indent-tabs-mode nil
   -*- */


//Duration structure 
typedef struct {
	double total_d;
	double min_d;
	double max_d;
	int count;
} Duration;
//Initialises a duration (all to 0 and DBL_MAX for min)
void duration_init(Duration* d);
//Updates a duration with two timespecs
void update_duration(Duration* d, struct timespec t1, struct timespec t2);
//writes one duration
void duration_write_csv(const char* filename, const Duration* d);


//All tracked functions
enum FunctionIndex {
  PRINT_TIMESTAMP,
  PRINT_TIMESTAMP_HEADER,
  PRINT_DURATION,
  PRINT_DURATION_HEADER,
  PRINT_EVENT,
  PRINT_FLAME,
  PRINT_CSV,
  PRINT_CSV_BULK,
  PRINT_TRACE,
  GET_CURRENT_INDEX, 
  PRINT_THREAD_STRUCTURE,
  PRINT_STRUCTURE,
  POLL_CURR_TOKEN, 
  GET_NEXT_TOKEN,
  POLL2,
  PRINT_EVENT1,
  PRINT_EVENT2,
  PRINT_EVENT3,
  GET_EVENT,
  GET_EVENT_OCC,
  TOK,
  NB_FUNCTIONS
};


//Table of durations
extern Duration durations[NB_FUNCTIONS];
//Writes all the durations of duration[NB_FUNCTIONS] to a file
void duration_write_all_csv(const char* filename);
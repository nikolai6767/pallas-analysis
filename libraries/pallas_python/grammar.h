//
// Created by khatharsis on 10/02/25.
//
#pragma once
#include "pallas_python.h"
typedef struct {
  PyObject ob_base;
  pallas::Token token;
} TokenObject;
extern PyTypeObject TokenType;

typedef struct {
  PyObject ob_base;
  pallas::Sequence* sequence;
  pallas::Thread* thread;
} SequenceObject;
extern PyTypeObject SequenceType;

typedef struct {
  PyObject ob_base;
  pallas::Loop* loop;
  pallas::Thread* thread;
} LoopObject;
extern PyTypeObject LoopType;

typedef struct {
  PyObject ob_base;
  pallas::EventSummary* event_summary;
  pallas::Thread* thread;
} EventSummaryObject;
extern PyTypeObject EventSummaryType;


typedef struct {
  PyObject ob_base;
  pallas::Event* event;
  pallas::Thread* thread;
} EventObject;
extern PyTypeObject EventType;

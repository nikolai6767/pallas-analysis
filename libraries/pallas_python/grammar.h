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
} SequenceObject;
extern PyTypeObject SequenceType;

typedef struct {
  PyObject ob_base;
  pallas::Loop* loop;
} LoopObject;
extern PyTypeObject LoopType;

typedef struct {
  PyObject ob_base;
  pallas::EventSummary* event_summary;
} EventSummaryObject;
extern PyTypeObject EventSummaryType;

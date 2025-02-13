//
// Created by khatharsis on 10/02/25.
//
#pragma once
#include "pallas_python.h"

typedef struct {
  PyObject ob_base;
  pallas::Thread* thread;
} ThreadObject;
extern PyTypeObject ThreadType;

// Object for the Archives
typedef struct {
  PyObject ob_base;
  pallas::Archive* archive;  // We're using a pointer for less memory shenanigans
} ArchiveObject;
extern PyTypeObject ArchiveType;

// Creating the Python Object that'll match the trace
typedef struct {
  PyObject ob_base;
  pallas::GlobalArchive* trace;
} TraceObject;

extern PyTypeObject TraceType;

PyObject* open_trace(PyObject* self, PyObject* args);
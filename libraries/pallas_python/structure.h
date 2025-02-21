//
// Created by khatharsis on 10/02/25.
//
#pragma once
#include "pallas_python.h"

typedef struct LocationGroupObject {
  PyObject ob_base;
  const pallas::LocationGroup* locationGroup;
  const char* name;
  struct LocationGroupObject* parent;
} LocationGroupObject;
extern PyTypeObject LocationGroupType;

typedef struct {
  PyObject ob_base;
  const pallas::Location* location;
  const char* name;
  struct LocationGroupObject* parent;
} LocationObject;
extern PyTypeObject LocationType;


typedef struct {
  PyObject ob_base;
  pallas::Thread* thread;
} ThreadObject;
extern PyTypeObject ThreadType;

// Object for the Archives
typedef struct {
  PyObject ob_base;
  pallas::Archive* archive;
} ArchiveObject;
extern PyTypeObject ArchiveType;

// Creating the Python Object that'll match the trace
typedef struct {
  PyObject ob_base;
  pallas::GlobalArchive* trace;
} TraceObject;

extern PyTypeObject TraceType;

PyObject* open_trace(PyObject* self, PyObject* args);
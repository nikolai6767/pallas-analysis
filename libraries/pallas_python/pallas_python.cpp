//
// Created by khatharsis on 23/01/25.
//
#define PY_SSIZE_T_CLEAN
#include <Python.h>
// This HAS to be at the beginning of the file
// Do NOT modify it

// Handle additional fields introduced in Python 3.13+

#if PY_VERSION_HEX >= ((3 << 24) | (13 << 16))
#define PYTYPEOBJECT_EXTRA_FIELDS \
  0,   /* tp_vectorcall */        \
    0, /* tp_watched */           \
    0  /* tp_versions_used */
#elif PY_VERSION_HEX >= ((3 << 24) | (8 << 16))
#define PYTYPEOBJECT_EXTRA_FIELDS 0 /* tp_vectorcall */
#else
#define PYTYPEOBJECT_EXTRA_FIELDS
#endif

#include "pallas/pallas.h"
#include "pallas/pallas_archive.h"
#include "pallas/pallas_storage.h"

#include "global_archive.cpp"


static PyMethodDef PallasMethods[] = {
  {"open_trace", open_trace, METH_VARARGS, "Open a Pallas trace."},
  {nullptr, nullptr, 0, nullptr}, /* Sentinel */
};

static struct PyModuleDef pallasmodule = {
  PyModuleDef_HEAD_INIT,
  "pallas_python",                             /* name of module */
  "Python API for the Pallas Tracing Library", /* module documentation, may be NULL */
  -1,                                          /* size of per-interpreter state of the module, or -1 if the module keeps state in global variables. */
  PallasMethods,
  nullptr,
  nullptr,
  nullptr,
  nullptr,
};

PyMODINIT_FUNC PyInit_pallas_python(void) {
  PyObject* m;
  if (PyType_Ready(&TraceType) < 0)
    return NULL;
  if (PyType_Ready(&ArchiveType) < 0)
    return NULL;

  m = PyModule_Create(&pallasmodule);
  if (m == NULL)
    return NULL;

  if (PyModule_AddObjectRef(m, "Trace", (PyObject*)&TraceType) < 0) {
    Py_DECREF(m);
    return NULL;
  }

  if (PyModule_AddObjectRef(m, "Archive", (PyObject*)&ArchiveType) < 0) {
    Py_DECREF(m);
    return NULL;
  }

  return m;
}
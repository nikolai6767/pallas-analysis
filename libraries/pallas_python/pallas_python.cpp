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

#include "trace_structure.cpp"

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

#define PYTHON_CHECK_READY(TypeName)  if (PyType_Ready(& TypeName##Type) < 0) return NULL

#define ADD_PYTHON_TYPE(TypeName) \
if (PyModule_AddObjectRef(m, #TypeName, (PyObject*)& TypeName##Type) < 0) { \
Py_DECREF(m); \
return NULL; \
  }

PyMODINIT_FUNC PyInit_pallas_python(void) {
  PyObject* m;
  PYTHON_CHECK_READY(Token);
  PYTHON_CHECK_READY(Sequence);
  PYTHON_CHECK_READY(Loop);
  PYTHON_CHECK_READY(EventSummary);

  PYTHON_CHECK_READY(Thread);
  PYTHON_CHECK_READY(Trace);
  PYTHON_CHECK_READY(Archive);


  m = PyModule_Create(&pallasmodule);
  if (m == NULL)
    return NULL;

  ADD_PYTHON_TYPE(Token);
  ADD_PYTHON_TYPE(Sequence);
  ADD_PYTHON_TYPE(Loop);
  ADD_PYTHON_TYPE(EventSummary);

  ADD_PYTHON_TYPE(Thread);
  ADD_PYTHON_TYPE(Trace);
  ADD_PYTHON_TYPE(Archive);

  return m;
}
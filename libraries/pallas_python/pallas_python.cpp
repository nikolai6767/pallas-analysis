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

// Creating the Python Object that'll match the trace
typedef struct {
  PyObject ob_base;
  pallas::GlobalArchive trace;
} TraceObject;

// Defining some custom members for the
static PyMemberDef Custom_members[] = {
  {"dir_name", Py_T_STRING, offsetof(TraceObject, trace) + offsetof(pallas::GlobalArchive, dir_name), 0, "Directory name."},
  {"trace_name", Py_T_STRING, offsetof(TraceObject, trace) + offsetof(pallas::GlobalArchive, trace_name), 0, "Trace name."},
  {"fullpath", Py_T_STRING, offsetof(TraceObject, trace) + offsetof(pallas::GlobalArchive, fullpath), 0, "Full path."},
  {"nb_archives", Py_T_INT, offsetof(TraceObject, trace) + offsetof(pallas::GlobalArchive, nb_archives), 0, "Number of Archives."},
  {nullptr},  // Sentinel value, marks the end of the array
};

// Defining custom getters for the locations
static PyObject* Trace_get_locations(TraceObject* self, void* closure) {
  PyObject* list = PyList_New(self->trace.locations.size());
  for (size_t i = 0; i < self->trace.locations.size(); ++i) {
    PyObject* loc = PyLong_FromLong(self->trace.locations[i].id);  // Example: Expose `id`
    PyList_SetItem(list, i, loc);
  }
  return list;
}

static PyGetSetDef Custom_getsetters[] = {
  {"locations", (getter)Trace_get_locations, nullptr, "List of Locations", nullptr},
  {nullptr}  // Sentinel
};

// Define a new PyTypeObject
static PyTypeObject TraceType = {
  .ob_base = PyVarObject_HEAD_INIT(NULL, 0).tp_name = "pallas_python.Trace",
  .tp_basicsize = sizeof(TraceObject),
  .tp_flags = Py_TPFLAGS_DEFAULT,
  .tp_doc = PyDoc_STR("A Pallas trace object."),
  .tp_members = Custom_members,
  .tp_getset = Custom_getsetters,
  .tp_new = PyType_GenericNew,
};
static PyObject* open_trace(PyObject* self, PyObject* args) {
  const char* trace_name;
  if (!PyArg_ParseTuple(args, "s", &trace_name)) {
    return nullptr;
  }
  auto* temp = PyObject_New(TraceObject, &TraceType);
  if (!temp) {
    return nullptr;
  }
  // WTF
  std::memset(&temp->trace, 0, sizeof(temp->trace));
  /* OK so the Python library is just straight up lying to us now
   * Because it claims to initialize everything to NULL
   * WHICH IT CLEARLY DOES NOT
   * This is stupid and the person who designed this deserves to `git commit sepuku`
   */
  pallasReadGlobalArchive(&temp->trace, trace_name);
  return reinterpret_cast<PyObject*>(temp);
}

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

  m = PyModule_Create(&pallasmodule);
  if (m == NULL)
    return NULL;

  if (PyModule_AddObjectRef(m, "Custom", (PyObject*)&TraceType) < 0) {
    Py_DECREF(m);
    return NULL;
  }

  return m;
}
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

#define PYTHON_CHECK_READY(TypeName)     \
  if (PyType_Ready(&TypeName##Type) < 0) \
  return NULL

#define ADD_PYTHON_TYPE(TypeName)                                            \
  if (PyModule_AddObjectRef(m, #TypeName, (PyObject*)&TypeName##Type) < 0) { \
    Py_DECREF(m);                                                            \
    return NULL;                                                             \
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

  import_array();
  m = PyModule_Create(&pallasmodule);
  if (m == NULL)
    return NULL;


  // Import the enum module
  PyObject *enumModule = PyImport_ImportModule("enum");
  PyObject *enumClass = PyObject_GetAttrString(enumModule, "Enum");

  // Create some enums
  // First the TokenType
  PyObject* tokenTypeDict = PyDict_New();
  PyDict_SetItemString(tokenTypeDict, "INVALID", PyLong_FromLong(pallas::TypeInvalid));
  PyDict_SetItemString(tokenTypeDict, "EVENT", PyLong_FromLong(pallas::TypeEvent));
  PyDict_SetItemString(tokenTypeDict, "SEQUENCE", PyLong_FromLong(pallas::TypeSequence));
  PyDict_SetItemString(tokenTypeDict, "LOOP", PyLong_FromLong(pallas::TypeLoop));

  PyObject *tokenTypeEnum = PyObject_CallFunction(enumClass, "sO", "TokenType", tokenTypeDict);
  PyModule_AddObject(m, "TokenType", tokenTypeEnum);

  // Then the RecordType
  // Create the Enum class for EventRecord
  PyObject* eventRecordDict = PyDict_New();
  PyDict_SetItemString(eventRecordDict, "BUFFER_FLUSH", PyLong_FromLong(pallas::PALLAS_EVENT_BUFFER_FLUSH));
  PyDict_SetItemString(eventRecordDict, "MEASUREMENT_ON_OFF", PyLong_FromLong(pallas::PALLAS_EVENT_MEASUREMENT_ON_OFF));
  PyDict_SetItemString(eventRecordDict, "ENTER", PyLong_FromLong(pallas::PALLAS_EVENT_ENTER));
  PyDict_SetItemString(eventRecordDict, "LEAVE", PyLong_FromLong(pallas::PALLAS_EVENT_LEAVE));
  PyDict_SetItemString(eventRecordDict, "MPI_SEND", PyLong_FromLong(pallas::PALLAS_EVENT_MPI_SEND));
  PyDict_SetItemString(eventRecordDict, "MPI_ISEND", PyLong_FromLong(pallas::PALLAS_EVENT_MPI_ISEND));
  PyDict_SetItemString(eventRecordDict, "MPI_ISEND_COMPLETE", PyLong_FromLong(pallas::PALLAS_EVENT_MPI_ISEND_COMPLETE));
  PyDict_SetItemString(eventRecordDict, "MPI_IRECV_REQUEST", PyLong_FromLong(pallas::PALLAS_EVENT_MPI_IRECV_REQUEST));
  PyDict_SetItemString(eventRecordDict, "MPI_RECV", PyLong_FromLong(pallas::PALLAS_EVENT_MPI_RECV));
  PyDict_SetItemString(eventRecordDict, "MPI_IRECV", PyLong_FromLong(pallas::PALLAS_EVENT_MPI_IRECV));
  PyDict_SetItemString(eventRecordDict, "MPI_REQUEST_TEST", PyLong_FromLong(pallas::PALLAS_EVENT_MPI_REQUEST_TEST));
  PyDict_SetItemString(eventRecordDict, "MPI_REQUEST_CANCELLED", PyLong_FromLong(pallas::PALLAS_EVENT_MPI_REQUEST_CANCELLED));
  PyDict_SetItemString(eventRecordDict, "MPI_COLLECTIVE_BEGIN", PyLong_FromLong(pallas::PALLAS_EVENT_MPI_COLLECTIVE_BEGIN));
  PyDict_SetItemString(eventRecordDict, "MPI_COLLECTIVE_END", PyLong_FromLong(pallas::PALLAS_EVENT_MPI_COLLECTIVE_END));
  PyDict_SetItemString(eventRecordDict, "OMP_FORK", PyLong_FromLong(pallas::PALLAS_EVENT_OMP_FORK));
  PyDict_SetItemString(eventRecordDict, "OMP_JOIN", PyLong_FromLong(pallas::PALLAS_EVENT_OMP_JOIN));
  PyDict_SetItemString(eventRecordDict, "OMP_ACQUIRE_LOCK", PyLong_FromLong(pallas::PALLAS_EVENT_OMP_ACQUIRE_LOCK));
  PyDict_SetItemString(eventRecordDict, "OMP_RELEASE_LOCK", PyLong_FromLong(pallas::PALLAS_EVENT_OMP_RELEASE_LOCK));
  PyDict_SetItemString(eventRecordDict, "OMP_TASK_CREATE", PyLong_FromLong(pallas::PALLAS_EVENT_OMP_TASK_CREATE));
  PyDict_SetItemString(eventRecordDict, "OMP_TASK_SWITCH", PyLong_FromLong(pallas::PALLAS_EVENT_OMP_TASK_SWITCH));
  PyDict_SetItemString(eventRecordDict, "OMP_TASK_COMPLETE", PyLong_FromLong(pallas::PALLAS_EVENT_OMP_TASK_COMPLETE));
  PyDict_SetItemString(eventRecordDict, "METRIC", PyLong_FromLong(pallas::PALLAS_EVENT_METRIC));
  PyDict_SetItemString(eventRecordDict, "PARAMETER_STRING", PyLong_FromLong(pallas::PALLAS_EVENT_PARAMETER_STRING));
  PyDict_SetItemString(eventRecordDict, "PARAMETER_INT", PyLong_FromLong(pallas::PALLAS_EVENT_PARAMETER_INT));
  PyDict_SetItemString(eventRecordDict, "PARAMETER_UNSIGNED_INT", PyLong_FromLong(pallas::PALLAS_EVENT_PARAMETER_UNSIGNED_INT));
  PyDict_SetItemString(eventRecordDict, "THREAD_FORK", PyLong_FromLong(pallas::PALLAS_EVENT_THREAD_FORK));
  PyDict_SetItemString(eventRecordDict, "THREAD_JOIN", PyLong_FromLong(pallas::PALLAS_EVENT_THREAD_JOIN));
  PyDict_SetItemString(eventRecordDict, "THREAD_TEAM_BEGIN", PyLong_FromLong(pallas::PALLAS_EVENT_THREAD_TEAM_BEGIN));
  PyDict_SetItemString(eventRecordDict, "THREAD_TEAM_END", PyLong_FromLong(pallas::PALLAS_EVENT_THREAD_TEAM_END));
  PyDict_SetItemString(eventRecordDict, "THREAD_ACQUIRE_LOCK", PyLong_FromLong(pallas::PALLAS_EVENT_THREAD_ACQUIRE_LOCK));
  PyDict_SetItemString(eventRecordDict, "THREAD_RELEASE_LOCK", PyLong_FromLong(pallas::PALLAS_EVENT_THREAD_RELEASE_LOCK));
  PyDict_SetItemString(eventRecordDict, "THREAD_TASK_CREATE", PyLong_FromLong(pallas::PALLAS_EVENT_THREAD_TASK_CREATE));
  PyDict_SetItemString(eventRecordDict, "THREAD_TASK_SWITCH", PyLong_FromLong(pallas::PALLAS_EVENT_THREAD_TASK_SWITCH));
  PyDict_SetItemString(eventRecordDict, "THREAD_TASK_COMPLETE", PyLong_FromLong(pallas::PALLAS_EVENT_THREAD_TASK_COMPLETE));
  PyDict_SetItemString(eventRecordDict, "THREAD_CREATE", PyLong_FromLong(pallas::PALLAS_EVENT_THREAD_CREATE));
  PyDict_SetItemString(eventRecordDict, "THREAD_BEGIN", PyLong_FromLong(pallas::PALLAS_EVENT_THREAD_BEGIN));
  PyDict_SetItemString(eventRecordDict, "THREAD_WAIT", PyLong_FromLong(pallas::PALLAS_EVENT_THREAD_WAIT));
  PyDict_SetItemString(eventRecordDict, "THREAD_END", PyLong_FromLong(pallas::PALLAS_EVENT_THREAD_END));
  PyDict_SetItemString(eventRecordDict, "IO_CREATE_HANDLE", PyLong_FromLong(pallas::PALLAS_EVENT_IO_CREATE_HANDLE));
  PyDict_SetItemString(eventRecordDict, "IO_DESTROY_HANDLE", PyLong_FromLong(pallas::PALLAS_EVENT_IO_DESTROY_HANDLE));
  PyDict_SetItemString(eventRecordDict, "IO_DUPLICATE_HANDLE", PyLong_FromLong(pallas::PALLAS_EVENT_IO_DUPLICATE_HANDLE));
  PyDict_SetItemString(eventRecordDict, "IO_SEEK", PyLong_FromLong(pallas::PALLAS_EVENT_IO_SEEK));
  PyDict_SetItemString(eventRecordDict, "IO_CHANGE_STATUS_FLAGS", PyLong_FromLong(pallas::PALLAS_EVENT_IO_CHANGE_STATUS_FLAGS));
  PyDict_SetItemString(eventRecordDict, "IO_DELETE_FILE", PyLong_FromLong(pallas::PALLAS_EVENT_IO_DELETE_FILE));
  PyDict_SetItemString(eventRecordDict, "IO_OPERATION_BEGIN", PyLong_FromLong(pallas::PALLAS_EVENT_IO_OPERATION_BEGIN));
  PyDict_SetItemString(eventRecordDict, "IO_OPERATION_TEST", PyLong_FromLong(pallas::PALLAS_EVENT_IO_OPERATION_TEST));
  PyDict_SetItemString(eventRecordDict, "IO_OPERATION_ISSUED", PyLong_FromLong(pallas::PALLAS_EVENT_IO_OPERATION_ISSUED));
  PyDict_SetItemString(eventRecordDict, "IO_OPERATION_COMPLETE", PyLong_FromLong(pallas::PALLAS_EVENT_IO_OPERATION_COMPLETE));
  PyDict_SetItemString(eventRecordDict, "IO_OPERATION_CANCELLED", PyLong_FromLong(pallas::PALLAS_EVENT_IO_OPERATION_CANCELLED));
  PyDict_SetItemString(eventRecordDict, "IO_ACQUIRE_LOCK", PyLong_FromLong(pallas::PALLAS_EVENT_IO_ACQUIRE_LOCK));
  PyDict_SetItemString(eventRecordDict, "IO_RELEASE_LOCK", PyLong_FromLong(pallas::PALLAS_EVENT_IO_RELEASE_LOCK));
  PyDict_SetItemString(eventRecordDict, "IO_TRY_LOCK", PyLong_FromLong(pallas::PALLAS_EVENT_IO_TRY_LOCK));
  PyDict_SetItemString(eventRecordDict, "PROGRAM_BEGIN", PyLong_FromLong(pallas::PALLAS_EVENT_PROGRAM_BEGIN));
  PyDict_SetItemString(eventRecordDict, "PROGRAM_END", PyLong_FromLong(pallas::PALLAS_EVENT_PROGRAM_END));
  PyDict_SetItemString(eventRecordDict, "NON_BLOCKING_COLLECTIVE_REQUEST", PyLong_FromLong(pallas::PALLAS_EVENT_NON_BLOCKING_COLLECTIVE_REQUEST));
  PyDict_SetItemString(eventRecordDict, "NON_BLOCKING_COLLECTIVE_COMPLETE", PyLong_FromLong(pallas::PALLAS_EVENT_NON_BLOCKING_COLLECTIVE_COMPLETE));
  PyDict_SetItemString(eventRecordDict, "COMM_CREATE", PyLong_FromLong(pallas::PALLAS_EVENT_COMM_CREATE));
  PyDict_SetItemString(eventRecordDict, "COMM_DESTROY", PyLong_FromLong(pallas::PALLAS_EVENT_COMM_DESTROY));
  PyDict_SetItemString(eventRecordDict, "GENERIC", PyLong_FromLong(pallas::PALLAS_EVENT_GENERIC));

  // Add EventRecord enum to module
  PyObject *eventRecordEnum = PyObject_CallFunction(enumClass, "sO", "EventRecord", eventRecordDict);
  PyModule_AddObject(m, "EventRecord", eventRecordEnum);

  ADD_PYTHON_TYPE(Token);
  ADD_PYTHON_TYPE(Sequence);
  ADD_PYTHON_TYPE(Loop);
  ADD_PYTHON_TYPE(EventSummary);

  ADD_PYTHON_TYPE(Thread);
  ADD_PYTHON_TYPE(Trace);
  ADD_PYTHON_TYPE(Archive);

  return m;
}
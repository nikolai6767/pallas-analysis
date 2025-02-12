//
// Created by khatharsis on 31/01/25.
//
#define NO_IMPORT_ARRAY
#define PY_ARRAY_UNIQUE_SYMBOL PallasPython
#include "structure.h"
#include "grammar.h"

PyObject* Thread_get_id(ThreadObject* self, void*) {
  return PyLong_FromLong(self->thread->id);
}

PyObject* Thread_get_event_summaries(ThreadObject* self, void*) {
  PyObject* list = PyList_New(self->thread->nb_events);
  for (int eid = 0; eid < self->thread->nb_events; eid++) {
      auto* eventSummary = new EventSummaryObject{
        .ob_base = PyObject_HEAD_INIT(&EventSummaryType)  //
                     .event_summary = &self->thread->events[eid],
      };
      PyList_SET_ITEM(list, eid, reinterpret_cast<PyObject*>(eventSummary));
  }
  return list;
}

PyObject* Thread_get_sequences(ThreadObject* self, void*) {
  PyObject* list = PyList_New(self->thread->nb_sequences);
  for (int sid = 0; sid < self->thread->nb_sequences; sid++) {
    auto* sequence = new SequenceObject{
      .ob_base = PyObject_HEAD_INIT(&SequenceType)  //
                   .sequence = self->thread->sequences[sid],
    };
    PyList_SET_ITEM(list, sid, reinterpret_cast<PyObject*>(sequence));
  }
  return list;
}

PyObject* Thread_get_loops(ThreadObject* self, void*) {
  PyObject* list = PyList_New(self->thread->nb_loops);
  for (int lid = 0; lid < self->thread->nb_loops; lid++) {
    auto* loop = new LoopObject {
      .ob_base = PyObject_HEAD_INIT(&LoopType)  //
                   .loop = &self->thread->loops[lid],
    };
    PyList_SET_ITEM(list, lid, reinterpret_cast<PyObject*>(loop));
  }
  return list;
}

PyGetSetDef Thread_getset[] = {
  {"id", (getter)Thread_get_id, nullptr, "Thread ID", nullptr},
  {"events", (getter)Thread_get_event_summaries, nullptr, "List of events", nullptr},
  {"sequences", (getter)Thread_get_sequences, nullptr, "List of sequences", nullptr},
  {"loops", (getter)Thread_get_loops, nullptr, "List of loops", nullptr},
  {nullptr}  // Sentinel
};

PyTypeObject ThreadType = {
  .ob_base = PyVarObject_HEAD_INIT(nullptr, 0).tp_name = "pallas_python.Thread",
  .tp_basicsize = sizeof(ThreadObject),
  .tp_flags = Py_TPFLAGS_DEFAULT,
  .tp_doc = PyDoc_STR("A Pallas thread object."),
  .tp_getset = Thread_getset,
};

// So we have to do a whole bunch of getters
PyObject* Archive_get_dir_name(ArchiveObject* self, void*) {
  return PyUnicode_FromString(self->archive->dir_name);
}

PyObject* Archive_get_trace_name(ArchiveObject* self, void*) {
  return PyUnicode_FromString(self->archive->trace_name);
}

PyObject* Archive_get_fullpath(ArchiveObject* self, void*) {
  return PyUnicode_FromString(self->archive->fullpath);
}

PyObject* Archive_get_threads(ArchiveObject* self, void* closure) {
  PyObject* list = PyList_New(0);
  for (size_t i = 0; i < self->archive->nb_threads; ++i) {
    if (self->archive->getThreadAt(i)) {
      auto* thread = PyObject_New(ThreadObject, &ThreadType);
      PyObject_Init(reinterpret_cast<PyObject*>(thread), &ThreadType);
      thread->thread = self->archive->getThreadAt(i);
      PyList_Append(list, reinterpret_cast<PyObject*>(thread));
    }
  }
  // This is disgusting
  // Blame EZTrace for giving us false threads !!!
  return list;
}

PyGetSetDef Archive_getset[] = {
  {"dir_name", (getter)Archive_get_dir_name, nullptr, "Directory name of the archive.", nullptr},
  {"trace_name", (getter)Archive_get_trace_name, nullptr, "Trace name of the archive.", nullptr},
  {"fullpath", (getter)Archive_get_fullpath, nullptr, "Full path of the archive.", nullptr},
  {"threads", (getter)Archive_get_threads, nullptr, "List of threads in the archive.", nullptr},

  {nullptr}  // Sentinel
};

// Define a new PyTypeObject
PyTypeObject ArchiveType = {
  .ob_base = PyVarObject_HEAD_INIT(nullptr, 0).tp_name = "pallas_python.Archive",
  .tp_basicsize = sizeof(ArchiveObject),
  .tp_flags = Py_TPFLAGS_DEFAULT,
  .tp_doc = PyDoc_STR("A Pallas archive object."),
  .tp_members = nullptr,
  .tp_getset = Archive_getset,
};

// Defining some custom members
PyMemberDef Trace_members[] = {
  {"dir_name", Py_T_STRING, offsetof(TraceObject, trace) + offsetof(pallas::GlobalArchive, dir_name), 0, "Directory name."},
  {"trace_name", Py_T_STRING, offsetof(TraceObject, trace) + offsetof(pallas::GlobalArchive, trace_name), 0, "Trace name."},
  {"fullpath", Py_T_STRING, offsetof(TraceObject, trace) + offsetof(pallas::GlobalArchive, fullpath), 0, "Full path."},
  {"nb_archives", Py_T_INT, offsetof(TraceObject, trace) + offsetof(pallas::GlobalArchive, nb_archives), 0, "Number of Archives."},
  {nullptr},  // Sentinel value, marks the end of the array
};

// Defining custom getters for the Locations / Locations Groups
PyObject* Trace_get_locations(TraceObject* self, void* closure) {
  PyObject* list = PyList_New(self->trace.locations.size());
  for (size_t i = 0; i < self->trace.locations.size(); ++i) {
    PyObject* loc = PyLong_FromLong(self->trace.locations[i].id);
    PyList_SetItem(list, i, loc);
  }
  return list;
}
PyObject* Trace_get_location_groups(TraceObject* self, void* closure) {
  PyObject* list = PyList_New(self->trace.location_groups.size());
  for (size_t i = 0; i < self->trace.location_groups.size(); ++i) {
    PyObject* loc = PyLong_FromLong(self->trace.location_groups[i].id);
    PyList_SetItem(list, i, loc);
  }
  return list;
}

PyObject* Trace_get_archives(TraceObject* self, void* closure) {
  PyObject* list = PyList_New(self->trace.location_groups.size());
  int i = 0;
  for (auto& locationGroup : self->trace.location_groups) {
    auto* archive = PyObject_New(ArchiveObject, &ArchiveType);
    PyObject_Init(reinterpret_cast<PyObject*>(archive), &ArchiveType);
    if (locationGroup.mainLoc == PALLAS_THREAD_ID_INVALID)
      archive->archive = self->trace.getArchive(locationGroup.id);
    else
      archive->archive = self->trace.getArchive(locationGroup.mainLoc);

    PyList_SetItem(list, i++, reinterpret_cast<PyObject*>(archive));
  }

  // This is disgusting code but that's how it works in the readGlobalArchive
  // so I don't see why we shouldn't use it here
  // Blame EZTrace for giving us wrongly formatted Location Groups !!!
  return list;
}

PyObject* Trace_get_strings(TraceObject* self, void* ) {
  PyObject* map = PyDict_New();
  for (auto& [key, value] : self->trace.definitions.strings) {
    PyDict_SetItem(map, PyLong_FromLong(key), PyUnicode_FromStringAndSize(value.str, value.length - 1));
  }
  return map;
}

PyGetSetDef Trace_getsetters[] = {
  {"locations", (getter)Trace_get_locations, nullptr, "List of Locations", nullptr},
  {"location_groups", (getter)Trace_get_location_groups, nullptr, "List of Location Groups", nullptr},
  {"archives", (getter)Trace_get_archives, nullptr, "List of Archives", nullptr},
   {"strings", (getter) Trace_get_strings, nullptr, "Array of Strings", nullptr},
  {nullptr}  // Sentinel
};

PyTypeObject TraceType = {
  .ob_base = PyVarObject_HEAD_INIT(nullptr, 0).tp_name = "pallas_python.Trace",
  .tp_basicsize = sizeof(TraceObject),
  .tp_flags = Py_TPFLAGS_DEFAULT,
  .tp_doc = PyDoc_STR("A Pallas trace object."),
  .tp_members = Trace_members,
  .tp_getset = Trace_getsetters,
};


PyObject* open_trace(PyObject* self, PyObject* args) {
  const char* trace_name;
  if (!PyArg_ParseTuple(args, "s", &trace_name)) {
    return nullptr;
  }
  auto* temp = PyObject_New(TraceObject, &TraceType);
  PyObject_Init(reinterpret_cast<PyObject*>(temp), &TraceType);
  if (!temp) {
    return nullptr;
  }
  // WTF
  std::memset(&temp->trace, 0, sizeof(temp->trace));
  /*
   * OK so the Python library is just straight up lying to us now
   * Because it claims to initialize everything to nullptr
   * WHICH IT CLEARLY DOES NOT
   * This is stupid and the person who designed this deserves to `git commit sepuku`
   *
   * Edit: OK so actually idk what the "tp_alloc" field is used for
   * Because this https://docs.python.org/3.13/c-api/type.html#c.PyType_GenericAlloc is actually true
   * But so is this https://docs.python.org/3.13/c-api/allocation.html#c.PyObject_New
   * And the second one CLAIMS that it doesn't set anything to nullptr, it only allocates.
   * ...
   * What ? Why ? What ?
   * They have an allocator that claims to initialize everything to nullptr
   * but it's actually not what's used for anything other than the Python header ?!?!?!
   * Are you shitting me ?!?! Python was a mistake.
   */
  pallasReadGlobalArchive(&temp->trace, trace_name);
  return reinterpret_cast<PyObject*>(temp);
}
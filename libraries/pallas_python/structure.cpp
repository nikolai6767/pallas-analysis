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
    auto* loop = new LoopObject{
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

LocationGroupObject* getLocationGroup(const pallas::LocationGroup* loc_group, pallas::GlobalArchive* trace) {
  LocationGroupObject* locationObject = PyObject_New(LocationGroupObject, &LocationGroupType);
  locationObject->locationGroup = loc_group;
  locationObject->name = trace->definitions.getString(loc_group->name)->str;
  if (loc_group->parent != PALLAS_LOCATION_GROUP_ID_INVALID) {
    auto parent = trace->getLocationGroup(loc_group->parent);
    locationObject->parent = getLocationGroup(parent, trace);
  } else {
    locationObject->parent = nullptr;
  }
  return locationObject;
}

PyObject* LocationGroup_get_id(LocationGroupObject* self, void* closure) {
  return PyLong_FromLong(self->locationGroup->id);
}
PyObject* LocationGroup_get_name(LocationGroupObject* self, void* closure) {
  return PyUnicode_FromString(self->name);
}

PyObject* Location_get_parent(LocationObject* self, void* closure) {
  if (self->parent == nullptr) {
    return Py_None;
  }
  return reinterpret_cast<PyObject*>(self->parent);
}
PyObject* LocationGroup_get_parent(LocationGroupObject* self, void* closure) {
  if (self->parent == nullptr) {
    return Py_None;
  }
  return reinterpret_cast<PyObject*>(self->parent);
}

PyGetSetDef LocationGroup_getset[] = {
  {"id", (getter)LocationGroup_get_id, nullptr, "ID of that LocationGroup.", nullptr},
  {"name", (getter)LocationGroup_get_name, nullptr, "Name of that LocationGroup.", nullptr},
  {"parent", (getter)LocationGroup_get_parent, nullptr, "Parent group of that LocationGroup. Maybe be None.", nullptr},
  {nullptr}  // Sentinel
};

PyObject* LocationGroup_to_string(PyObject* self) {
  auto* trueSelf = reinterpret_cast<LocationGroupObject*>(self);
  return PyUnicode_FromFormat("<LocationGroup id=%d, name=%s>", trueSelf->locationGroup->id, trueSelf->name);
}

// Define a new PyTypeObject
PyTypeObject LocationGroupType = {
  .ob_base = PyVarObject_HEAD_INIT(nullptr, 0).tp_name = "pallas_python.LocationGroup",
  .tp_basicsize = sizeof(LocationGroupObject),
  .tp_repr = LocationGroup_to_string,
  .tp_str = LocationGroup_to_string,
  .tp_flags = Py_TPFLAGS_DEFAULT,
  .tp_doc = PyDoc_STR("A Pallas LocationGroup object."),
  .tp_members = nullptr,
  .tp_getset = LocationGroup_getset,
};

LocationObject* getLocation(const pallas::Location* loc, pallas::GlobalArchive* trace) {
  LocationObject* locationObject = PyObject_New(LocationObject, &LocationType);
  locationObject->location = loc;
  locationObject->name = trace->definitions.getString(loc->name)->str;
  if (loc->parent != PALLAS_LOCATION_GROUP_ID_INVALID) {
    auto parent = trace->getLocationGroup(loc->parent);
    locationObject->parent = getLocationGroup(parent, trace);
  } else {
    locationObject->parent = nullptr;
  }
  return locationObject;
}

PyObject* Location_get_id(LocationObject* self, void* closure) {
  return PyLong_FromLong(self->location->id);
}
PyObject* Location_get_name(LocationObject* self, void* closure) {
  return PyUnicode_FromString(self->name);
}
PyGetSetDef Location_getset[] = {
  {"id", (getter)Location_get_id, nullptr, "ID of that Location.", nullptr},
  {"name", (getter)Location_get_name, nullptr, "Name of that Location.", nullptr},
  {"parent", (getter)Location_get_parent, nullptr, "Parent group of that location. Should not be None.", nullptr},
  {nullptr}  // Sentinel
};

PyObject* Location_to_string(PyObject* self) {
  auto* trueSelf = reinterpret_cast<LocationObject*>(self);
  return PyUnicode_FromFormat("<Location id=%d, name=%s>", trueSelf->location->id, trueSelf->name);
}

// Define a new PyTypeObject
PyTypeObject LocationType = {
  .ob_base = PyVarObject_HEAD_INIT(nullptr, 0).tp_name = "pallas_python.Location",
  .tp_basicsize = sizeof(LocationObject),
  .tp_repr = Location_to_string,
  .tp_str = Location_to_string,
  .tp_flags = Py_TPFLAGS_DEFAULT,
  .tp_doc = PyDoc_STR("A Pallas location object."),
  .tp_members = nullptr,
  .tp_getset = Location_getset,
};

// So we have to do a whole bunch of getters
PyObject* Trace_get_dir_name(TraceObject* self, void*) {
  return PyUnicode_FromString(self->trace->dir_name);
}

PyObject* Trace_get_trace_name(TraceObject* self, void*) {
  return PyUnicode_FromString(self->trace->trace_name);
}

PyObject* Trace_get_fullpath(TraceObject* self, void*) {
  return PyUnicode_FromString(self->trace->fullpath);
}

// Defining custom getters for the Locations / Locations Groups
PyObject* Trace_get_locations(TraceObject* self, void* closure) {
  PyObject* dict = PyDict_New();
  for (size_t i = 0; i < self->trace->locations.size(); ++i) {
    int j = 0;
    auto* loc = &self->trace->locations[i];
    LocationObject* location_object = getLocation(loc, self->trace);
    PyDict_SetItem(dict, PyLong_FromLong(loc->id), reinterpret_cast<PyObject*>(location_object));
  }
  return dict;
}

PyObject* Trace_get_location_groups(TraceObject* self, void* closure) {
  PyObject* dict = PyDict_New();
  for (size_t i = 0; i < self->trace->location_groups.size(); ++i) {
    auto* loc = &self->trace->location_groups[i];
    LocationGroupObject* location_object = getLocationGroup(loc, self->trace);
    PyDict_SetItem(dict, PyLong_FromLong(loc->id), reinterpret_cast<PyObject*>(location_object));
  }
  return dict;
}

PyObject* Trace_get_archives(TraceObject* self, void* closure) {
  PyObject* list = PyList_New(self->trace->location_groups.size());
  int i = 0;
  for (auto& locationGroup : self->trace->location_groups) {
    auto* archive = PyObject_New(ArchiveObject, &ArchiveType);
    PyObject_Init(reinterpret_cast<PyObject*>(archive), &ArchiveType);
    if (locationGroup.mainLoc == PALLAS_THREAD_ID_INVALID)
      archive->archive = self->trace->getArchive(locationGroup.id);
    else
      archive->archive = self->trace->getArchive(locationGroup.mainLoc);

    PyList_SetItem(list, i++, reinterpret_cast<PyObject*>(archive));
  }

  // This is disgusting code but that's how it works in the readGlobalArchive
  // so I don't see why we shouldn't use it here
  // Blame EZTrace for giving us wrongly formatted Location Groups !!!
  return list;
}

PyObject* Trace_get_strings(TraceObject* self, void*) {
  PyObject* map = PyDict_New();
  for (auto& [key, value] : self->trace->definitions.strings) {
    PyDict_SetItem(map, PyLong_FromLong(key), PyUnicode_FromStringAndSize(value.str, value.length - 1));
  }
  return map;
}
PyObject* Region_toDict(const pallas::Region& r, const pallas::Definition& d) {
  PyObject* map = PyDict_New();
  PyDict_SetItem(map, PyUnicode_FromString("region_ref"), PyLong_FromLong(r.region_ref));
  PyDict_SetItem(map, PyUnicode_FromString("string"), PyUnicode_FromStringAndSize(d.strings.find(r.string_ref)->second.str, d.strings.find(r.string_ref)->second.length - 1));
  return map;
}

PyObject* Trace_get_regions(TraceObject* self, void*) {
  PyObject* map = PyDict_New();
  for (auto& [key, value] : self->trace->definitions.regions) {
    PyDict_SetItem(map, PyLong_FromLong(key), Region_toDict(value, self->trace->definitions));
  }
  return map;
}

void Trace_dealloc(TraceObject* self) {
  delete self->trace;
  Py_TYPE(self)->tp_free((PyObject*)self);
}

PyGetSetDef Trace_getsetters[] = {
  {"dir_name", (getter)Trace_get_dir_name, nullptr, "Directory name of the trace.", nullptr},
  {"trace_name", (getter)Trace_get_trace_name, nullptr, "Trace name of the trace.", nullptr},
  {"fullpath", (getter)Trace_get_fullpath, nullptr, "Full path of the trace.", nullptr},
  {"locations", (getter)Trace_get_locations, nullptr, "List of Locations", nullptr},
  {"location_groups", (getter)Trace_get_location_groups, nullptr, "List of Location Groups", nullptr},
  {"archives", (getter)Trace_get_archives, nullptr, "List of Archives", nullptr},
  {"strings", (getter)Trace_get_strings, nullptr, "Dict of Strings", nullptr},
  {"regions", (getter)Trace_get_regions, nullptr, "Dict of Regions", nullptr},
  {nullptr}  // Sentinel
};

PyTypeObject TraceType = {
  .ob_base = PyVarObject_HEAD_INIT(nullptr, 0).tp_name = "pallas_python.Trace",
  .tp_basicsize = sizeof(TraceObject),
  .tp_dealloc = (destructor)Trace_dealloc,
  .tp_flags = Py_TPFLAGS_DEFAULT,
  .tp_doc = PyDoc_STR("A Pallas trace object."),
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
  temp->trace = new pallas::GlobalArchive;
  pallasReadGlobalArchive(temp->trace, trace_name);
  return reinterpret_cast<PyObject*>(temp);
}
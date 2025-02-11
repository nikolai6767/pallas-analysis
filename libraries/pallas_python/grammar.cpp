//
// Created by khatharsis on 05/02/25.
//
#define NO_IMPORT_ARRAY
#define PY_ARRAY_UNIQUE_SYMBOL PallasPython
#include "grammar.h"

PyObject* Token_get_type(TokenObject* self, void*) {
  return PyObject_CallFunction(tokenTypeEnum, "i", self->token.type);
}

PyObject* Token_get_id(TokenObject* self, void*) {
  return PyLong_FromLong(self->token.id);
}

PyObject* Token_repr(PyObject* self) {
  auto* tokenObject = reinterpret_cast<TokenObject*>(self);
  std::ostringstream tempString;
  switch (tokenObject->token.type) {
  case pallas::TypeInvalid:
    tempString << "U";
    break;
  case pallas::TypeEvent:
    tempString << "E";
    break;
  case pallas::TypeSequence:
    tempString << "S";
    break;
  case pallas::TypeLoop:
    tempString << "L";
    break;
  }
  tempString << tokenObject->token.id;
  return PyUnicode_FromString(tempString.str().c_str());
}

PyGetSetDef Token_getset[] = {
  {"type", (getter)Token_get_type, nullptr, "Type of the token.", nullptr},
  {"id", (getter)Token_get_id, nullptr, "ID of the token.", nullptr},
  {nullptr}  // Sentinel
};

PyTypeObject TokenType = {
  .ob_base = PyVarObject_HEAD_INIT(nullptr, 0).tp_name = "pallas_python.Token",
  .tp_basicsize = sizeof(TokenObject),
  .tp_repr = Token_repr,
  .tp_str = Token_repr,
  .tp_flags = Py_TPFLAGS_DEFAULT,
  .tp_doc = PyDoc_STR("A Pallas token object."),
  .tp_getset = Token_getset,
};

// More Grammar / Structure Stuff

PyObject* Sequence_get_tokens(SequenceObject* self, void*) {
  PyObject* list = PyList_New(self->sequence->tokens.size());
  for (size_t i = 0; i < self->sequence->tokens.size(); ++i) {
    auto* token = new TokenObject{
      .ob_base = PyObject_HEAD_INIT(&TokenType)  //
                   .token = self->sequence->tokens[i],
    };
    PyList_SET_ITEM(list, i, token);
  }
  return list;
}

PyObject* Sequence_get_durations(SequenceObject* self, void*) {
  if (!self->sequence->durations || self->sequence->durations->size == 0) {
    Py_RETURN_NONE;  // Return None if durations is empty or uninitialized
  }
  auto* durations = self->sequence->durations;

  npy_intp size = self->sequence->durations->size;
  PyObject* np_array = PyArray_SimpleNewFromData(1, &size, NPY_UINT64, &durations->front());
  if (!np_array)
    Py_RETURN_NONE;  // Check for allocation failure
  return np_array;
}

PyObject* Sequence_get_timestamps(SequenceObject* self, void*) {
  if (!self->sequence->timestamps || self->sequence->timestamps->size == 0) {
    Py_RETURN_NONE;  // Return None if timestamps is empty or uninitialized
  }
  auto* timestamps = self->sequence->timestamps;

  npy_intp size = self->sequence->timestamps->size;
  PyObject* np_array = PyArray_SimpleNewFromData(1, &size, NPY_UINT64, &timestamps->front());
  if (!np_array)
    Py_RETURN_NONE;  // Check for allocation failure
  return np_array;
}

PyObject* Sequence_get_id(SequenceObject* self, void*) {
  return PyLong_FromLong(self->sequence->id);
}

PyGetSetDef Sequence_getset[] = {
  {"id", (getter)Sequence_get_id, nullptr, "Token identifying that Sequence", nullptr},
  {"tokens", (getter)Sequence_get_tokens, nullptr, "List of tokens in the sequence", nullptr},
  {"durations", (getter)Sequence_get_durations, nullptr, "Numpy Array of durations of the sequence", nullptr},
  {"timestamps", (getter)Sequence_get_timestamps, nullptr, "Numpy Array of timestamps of the sequence", nullptr},
  {nullptr}  // Sentinel
};

PyObject* Loop_get_repeated_token(LoopObject* self, void*) {
  return PyLong_FromUnsignedLong(self->loop->repeated_token.id);
}

PyObject* Loop_get_self_id(LoopObject* self, void*) {
  return PyLong_FromUnsignedLong(self->loop->self_id.id);
}

PyObject* Loop_get_iterations(LoopObject* self, void*) {
  return PyLong_FromUnsignedLong(self->loop->nb_iterations);
}

PyGetSetDef Loop_getset[] = {
  {"repeated_token", (getter)Loop_get_repeated_token, nullptr, "Token of the Sequence being repeated", nullptr},
  {"id", (getter)Loop_get_self_id, nullptr, "Token identifying that Loop", nullptr},
  {"iterations", (getter)Loop_get_iterations, nullptr, "Number of iterations of the loop", nullptr},
  {nullptr}  // Sentinel
};

PyTypeObject SequenceType = {
  .ob_base = PyVarObject_HEAD_INIT(nullptr, 0).tp_name = "pallas_python.Sequence",
  .tp_basicsize = sizeof(SequenceObject),
  .tp_flags = Py_TPFLAGS_DEFAULT,
  .tp_doc = PyDoc_STR("A Pallas sequence object."),
  .tp_getset = Sequence_getset,
};

PyTypeObject LoopType = {
  .ob_base = PyVarObject_HEAD_INIT(nullptr, 0).tp_name = "pallas_python.Loop",
  .tp_basicsize = sizeof(LoopObject),
  .tp_flags = Py_TPFLAGS_DEFAULT,
  .tp_doc = PyDoc_STR("A Pallas Loop object."),
  .tp_getset = Loop_getset,
};

PyObject* EventSummary_to_string(PyObject* self) {
  return PyUnicode_FromFormat("<EventSummary id=%d>", reinterpret_cast<EventSummaryObject*>(self)->event_summary->id);
}

PyObject* EventSummary_get_id(EventSummaryObject* self, void*) {
  return PyLong_FromUnsignedLong(self->event_summary->id);
}

PyObject* EventSummary_get_event(EventSummaryObject* self, void*) {
  auto* event = new EventObject{
    .ob_base = PyObject_HEAD_INIT(&EventType)  //
                 .event = &self->event_summary->event,
  };
  return reinterpret_cast<PyObject*>(event);
}

PyObject* EventSummary_get_durations(EventSummaryObject* self, void*) {
  if (!self->event_summary->durations || self->event_summary->durations->size == 0) {
    Py_RETURN_NONE;  // Return None if durations is empty or uninitialized
  }
  auto* durations = self->event_summary->durations;

  npy_intp size = self->event_summary->durations->size;
  PyObject* np_array = PyArray_SimpleNewFromData(1, &size, NPY_UINT64, &durations->front());
  if (!np_array)
    Py_RETURN_NONE;  // Check for allocation failure
  return np_array;
}

PyObject* EventSummary_get_mean_duration(EventSummaryObject* self, void*) {
  return PyLong_FromLong(self->event_summary->durations->mean);
}
PyObject* EventSummary_get_min_duration(EventSummaryObject* self, void*) {
  return PyLong_FromLong(self->event_summary->durations->min);
}
PyObject* EventSummary_get_max_duration(EventSummaryObject* self, void*) {
  return PyLong_FromLong(self->event_summary->durations->max);
}
PyObject* EventSummary_get_n_occurrences(EventSummaryObject* self, void*) {
  return PyLong_FromLong(self->event_summary->durations->size);
}

PyGetSetDef EventSummary_getset[] = {
  {"id", (getter)EventSummary_get_id, nullptr, "ID of the Event", nullptr},
  {"event", (getter)EventSummary_get_event, nullptr, "The Event being summarized", nullptr},
  {"durations", (getter)EventSummary_get_durations, nullptr, "Durations for each occurrence of that Event", nullptr},
  {"min_duration", (getter)EventSummary_get_min_duration, nullptr, "Minimum duration for that Event", nullptr},
  {"max_duration", (getter)EventSummary_get_max_duration, nullptr, "Maximum duration for that Event", nullptr},
  {"mean_duration", (getter)EventSummary_get_mean_duration, nullptr, "Average duration for that Event", nullptr},
  {"number_occurrence", (getter)EventSummary_get_n_occurrences, nullptr, "Number of repetitions for that Event", nullptr},

  // {"attributes", (getter)EventSummary_get_attributes, nullptr, "List of attributes.", nullptr},
  {nullptr}  // Sentinel
};

PyTypeObject EventSummaryType = {
  .ob_base = PyVarObject_HEAD_INIT(nullptr, 0).tp_name = "pallas_python.EventSummary",
  .tp_basicsize = sizeof(EventSummaryObject),
  .tp_repr = EventSummary_to_string,
  .tp_str = EventSummary_to_string,
  .tp_flags = Py_TPFLAGS_DEFAULT,
  .tp_doc = PyDoc_STR("A Pallas EventSummary object."),
  .tp_getset = EventSummary_getset,
};

PyObject* Event_get_record(EventObject* self, void*) {
  return PyObject_CallFunction(eventRecordEnum, "l", self->event->record);
}

static inline void pop_data(pallas::Event* e, void* data, size_t data_size, byte*& cursor) {
  if (cursor == nullptr) {
    cursor = &e->event_data[0];
  }
  memcpy(data, cursor, data_size);
  cursor += data_size;
}

#define READ_LONG(type, name)                                 \
  {                                                           \
    type name;                                                \
    pop_data(e, &name, sizeof(type), cursor);                 \
    PyDict_SetItemString(dict, #name, PyLong_FromLong(name)); \
  }

static PyObject* Event_get_data(EventObject* self, void*) {
  PyObject* dict = PyDict_New();
  if (!dict)
    return NULL;
  auto* e = self->event;
  byte* cursor = nullptr;

  switch (e->record) {
  case pallas::PALLAS_EVENT_ENTER:
  case pallas::PALLAS_EVENT_LEAVE: {
    READ_LONG(pallas::RegionRef, region_ref);
    break;
  }
  case pallas::PALLAS_EVENT_THREAD_BEGIN:
  case pallas::PALLAS_EVENT_THREAD_END:
  case pallas::PALLAS_EVENT_THREAD_TEAM_BEGIN:
  case pallas::PALLAS_EVENT_THREAD_TEAM_END:
  case pallas::PALLAS_EVENT_THREAD_JOIN:
  case pallas::PALLAS_EVENT_MPI_COLLECTIVE_BEGIN:
  case pallas::PALLAS_EVENT_THREAD_TASK_CREATE:
  case pallas::PALLAS_EVENT_THREAD_TASK_COMPLETE:
  case pallas::PALLAS_EVENT_THREAD_TASK_SWITCH:
    // No additional data for these events
    break;
  case pallas::PALLAS_EVENT_THREAD_FORK:
  case pallas::PALLAS_EVENT_OMP_FORK: {
    READ_LONG(uint32_t, numberOfRequestedThreads);
  }
  case pallas::PALLAS_EVENT_MPI_SEND:
  case pallas::PALLAS_EVENT_MPI_RECV: {
    READ_LONG(uint32_t, receiver);
    READ_LONG(uint32_t, communicator);
    READ_LONG(uint32_t, msgTag);
    READ_LONG(uint64_t, msgLength);
    break;
  }
  case pallas::PALLAS_EVENT_MPI_ISEND:
  case pallas::PALLAS_EVENT_MPI_IRECV: {
    READ_LONG(uint32_t, receiver);
    READ_LONG(uint32_t, communicator);
    READ_LONG(uint32_t, msgTag);
    READ_LONG(uint64_t, msgLength);
    READ_LONG(uint64_t, requestID);
    break;
  }
  case pallas::PALLAS_EVENT_MPI_ISEND_COMPLETE:
  case pallas::PALLAS_EVENT_MPI_IRECV_REQUEST: {
    READ_LONG(uint64_t, requestID);
    break;
  }
  case pallas::PALLAS_EVENT_THREAD_ACQUIRE_LOCK:
  case pallas::PALLAS_EVENT_THREAD_RELEASE_LOCK:
  case pallas::PALLAS_EVENT_OMP_ACQUIRE_LOCK:
  case pallas::PALLAS_EVENT_OMP_RELEASE_LOCK: {
    READ_LONG(uint32_t, lockID);
    READ_LONG(uint32_t, acquisitionOrder);
    break;
  }
  case pallas::PALLAS_EVENT_MPI_COLLECTIVE_END: {
    READ_LONG(uint32_t, collectiveOp);
    READ_LONG(uint32_t, communicator);
    READ_LONG(uint32_t, root);
    READ_LONG(uint64_t, sizeSent);
    READ_LONG(uint64_t, sizeReceived);
    break;
  }
  case pallas::PALLAS_EVENT_OMP_TASK_CREATE:
  case pallas::PALLAS_EVENT_OMP_TASK_SWITCH:
  case pallas::PALLAS_EVENT_OMP_TASK_COMPLETE: {
    READ_LONG(uint64_t, taskID);
    break;
  }
  case pallas::PALLAS_EVENT_GENERIC: {
    pallas::StringRef event_name;
    pop_data(e, &event_name, sizeof(event_name), cursor);
    PyDict_SetItemString(dict, "event_name", PyUnicode_FromString(reinterpret_cast<const char*>(&event_name)));
    break;
  }

  default:
    Py_RETURN_NONE;
    break;
  }

  return dict;
}

PyGetSetDef Event_getset[] = {
  {"record", (getter)Event_get_record, nullptr, "Event record", nullptr},
  {"data", (getter)Event_get_data, nullptr, "That event's data", nullptr},
  {nullptr}  // Sentinel
};

PyTypeObject EventType = {
  .ob_base = PyVarObject_HEAD_INIT(nullptr, 0).tp_name = "pallas_python.Event",
  .tp_basicsize = sizeof(EventObject),
  .tp_flags = Py_TPFLAGS_DEFAULT,
  .tp_doc = PyDoc_STR("A Pallas Event object."),
  .tp_getset = Event_getset,
};

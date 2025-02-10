//
// Created by khatharsis on 05/02/25.
//
#include <numpy/arrayobject.h>  // Required for NumPy integration

typedef struct {
  PyObject ob_base;
  pallas::Token token;
} TokenObject;
static PyObject* Token_get_type(TokenObject* self, void*) {
  return PyLong_FromLong(self->token.type);
}

static PyObject* Token_get_id(TokenObject* self, void*) {
  return PyLong_FromLong(self->token.id);
}

static PyObject* Token_repr(PyObject* self) {
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

static PyGetSetDef Token_getset[] = {
  {"type", (getter)Token_get_type, nullptr, "Type of the token.", nullptr},
  {"id", (getter)Token_get_id, nullptr, "ID of the token.", nullptr},
  {nullptr}  // Sentinel
};
static PyTypeObject TokenType = {
  .ob_base = PyVarObject_HEAD_INIT(nullptr, 0).tp_name = "pallas_python.Token",
  .tp_basicsize = sizeof(TokenObject),
  .tp_repr = Token_repr,
  .tp_str = Token_repr,
  .tp_flags = Py_TPFLAGS_DEFAULT,
  .tp_doc = PyDoc_STR("A Pallas token object."),
  .tp_getset = Token_getset,
};

// More Grammar / Structure Stuff

typedef struct {
  PyObject ob_base;
  pallas::Sequence* sequence;
} SequenceObject;
typedef struct {
  PyObject ob_base;
  pallas::Loop* loop;
} LoopObject;
typedef struct {
  PyObject ob_base;
  pallas::EventSummary* event_summary;
} EventSummaryObject;

static PyObject* Sequence_get_tokens(SequenceObject* self, void*) {
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

static PyObject* Sequence_get_durations(SequenceObject* self, void*) {
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

static PyObject* Sequence_get_timestamps(SequenceObject* self, void*) {
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

static PyObject* Sequence_get_id(SequenceObject* self, void*) {
  return PyLong_FromLong(self->sequence->id);
}

static PyGetSetDef Sequence_getset[] = {
  {"id", (getter)Sequence_get_id, nullptr, "Token identifying that Sequence", nullptr},
  {"tokens", (getter)Sequence_get_tokens, nullptr, "List of tokens in the sequence", nullptr},
  {"durations", (getter)Sequence_get_durations, nullptr, "Numpy Array of durations of the sequence", nullptr},
  {"timestamps", (getter)Sequence_get_timestamps, nullptr, "Numpy Array of timestamps of the sequence", nullptr},
  {nullptr}  // Sentinel
};

static PyObject* Loop_get_repeated_token(LoopObject* self, void*) {
  return PyLong_FromUnsignedLong(self->loop->repeated_token.id);
}

static PyObject* Loop_get_self_id(LoopObject* self, void*) {
  return PyLong_FromUnsignedLong(self->loop->self_id.id);
}

static PyObject* Loop_get_iterations(LoopObject* self, void*) {
  return PyLong_FromUnsignedLong(self->loop->nb_iterations);
}

static PyGetSetDef Loop_getset[] = {
  {"repeated_token", (getter)Loop_get_repeated_token, nullptr, "Token of the Sequence being repeated", nullptr},
  {"id", (getter)Loop_get_self_id, nullptr, "Token identifying that Loop", nullptr},
  {"iterations", (getter)Loop_get_iterations, nullptr, "Number of iterations of the loop", nullptr},
  {nullptr}  // Sentinel
};

static PyObject* EventSummary_to_string(PyObject* self) {
  return PyUnicode_FromFormat("<EventSummary id=%d>", reinterpret_cast<EventSummaryObject*>(self)->event_summary->id);
}

static PyObject* EventSummary_get_id(EventSummaryObject* self, void*) {
  return PyLong_FromUnsignedLong(self->event_summary->id);
}

static PyObject* EventSummary_get_event(EventSummaryObject* self, void*) {
  return nullptr;  // PyLong_FromUnsignedLong(self->event_summary->event);
}

static PyObject* EventSummary_get_durations(EventSummaryObject* self, void*) {
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

static PyGetSetDef EventSummary_getset[] = {
  {"id", (getter)EventSummary_get_id, nullptr, "ID of the Event", nullptr},
  {"event", (getter)EventSummary_get_event, nullptr, "The Event being summarized", nullptr},
  {"durations", (getter)EventSummary_get_durations, nullptr, "Durations for each occurrence of that Event", nullptr},
  // {"attribute_buffer", (getter)EventSummary_get_attribute_buffer, nullptr, "Storage for Attribute", nullptr},
  {nullptr}  // Sentinel
};

static PyTypeObject SequenceType = {
  .ob_base = PyVarObject_HEAD_INIT(nullptr, 0).tp_name = "pallas_python.Sequence",
  .tp_basicsize = sizeof(SequenceObject),
  .tp_flags = Py_TPFLAGS_DEFAULT,
  .tp_doc = PyDoc_STR("A Pallas sequence object."),
  .tp_getset = Sequence_getset,
};

static PyTypeObject LoopType = {
  .ob_base = PyVarObject_HEAD_INIT(nullptr, 0).tp_name = "pallas_python.Loop",
  .tp_basicsize = sizeof(LoopObject),
  .tp_flags = Py_TPFLAGS_DEFAULT,
  .tp_doc = PyDoc_STR("A Pallas Loop object."),
  .tp_getset = Loop_getset,
};

static PyTypeObject EventSummaryType = {
  .ob_base = PyVarObject_HEAD_INIT(nullptr, 0).tp_name = "pallas_python.EventSummary",
  .tp_basicsize = sizeof(EventSummaryObject),
  .tp_str = EventSummary_to_string,
  .tp_flags = Py_TPFLAGS_DEFAULT,
  .tp_doc = PyDoc_STR("A Pallas EventSummary object."),
  .tp_getset = EventSummary_getset,
};

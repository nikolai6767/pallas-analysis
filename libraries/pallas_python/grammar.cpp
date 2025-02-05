//
// Created by khatharsis on 05/02/25.
//

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
static PyGetSetDef Token_getset[] = {
  {"type", (getter)Token_get_type, NULL, "Type of the token.", NULL},
  {"id", (getter)Token_get_id, NULL, "ID of the token.", NULL},
  {NULL}  // Sentinel
};
static PyTypeObject TokenType = {
  .ob_base = PyVarObject_HEAD_INIT(nullptr, 0).tp_name = "pallas_python.Token",
  .tp_basicsize = sizeof(TokenObject),
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
  pallas::Loop* Loop;
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

static PyObject* Sequence_get_timestamps(SequenceObject* self, void*) {
  PyObject* list = PyList_New(self->sequence->timestamps->size);
  for (size_t i = 0; i < self->sequence->timestamps->size; ++i) {
    PyList_SET_ITEM(list, i, PyLong_FromUnsignedLong((*self->sequence->timestamps)[i]));
  }
  return list;
}

static PyObject* Sequence_get_id(SequenceObject* self, void*) {
  return PyLong_FromLong(self->sequence->id);
}

static PyGetSetDef Sequence_getset[] = {
  {"id", (getter)Sequence_get_id, NULL, "Sequence ID", NULL},
  {"tokens", (getter)Sequence_get_tokens, NULL, "List of tokens in the sequence", NULL},
  {"timestamps", (getter)Sequence_get_timestamps, NULL, "List of timestamps in the sequence", NULL},
  {NULL}  // Sentinel
};
static PyGetSetDef Loop_getset[] = {
  {NULL}  // Sentinel
};
static PyGetSetDef EventSummary_getset[] = {
  {NULL}  // Sentinel
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
  .tp_flags = Py_TPFLAGS_DEFAULT,
  .tp_doc = PyDoc_STR("A Pallas EventSummary object."),
  .tp_getset = EventSummary_getset,
};

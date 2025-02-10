//
// Created by khatharsis on 10/02/25.
//
#pragma once
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
#include <numpy/arrayobject.h>  // Required for NumPy integration
#include <pallas/pallas.h>
#include <pallas/pallas_archive.h>
#include <pallas/pallas_storage.h>

static PyObject* tokenTypeEnum;
static PyObject* eventRecordEnum;
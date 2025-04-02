/*
 * Copyright (C) Telecom SudParis
 * See LICENSE in top-level directory.
 */

#pragma once
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
namespace py = pybind11;

#include <pallas/pallas.h>
#include <pallas/pallas_archive.h>
#include <pallas/pallas_storage.h>

namespace PYBIND11_NAMESPACE {
namespace detail {
template <>
struct type_caster<pallas::String> {
  PYBIND11_TYPE_CASTER(pallas::String, const_name("pallas::String"));
  static handle cast(pallas::String src, return_value_policy /* policy */, handle /* parent */) { return PyUnicode_FromStringAndSize(src.str, src.length - 1); }
};
}  // namespace detail
}  // namespace PYBIND11_NAMESPACE


/* -*-
   mode: c;
   c-file-style: "k&r";
   c-basic-offset 2;
   tab-width 2 ;
   indent-tabs-mode nil
   -*- */

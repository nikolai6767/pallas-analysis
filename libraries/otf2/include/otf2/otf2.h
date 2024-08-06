/*
 * Copyright (C) Telecom SudParis
 * See LICENSE in top-level directory.
 */

#ifndef OTF2_H
#define OTF2_H

/**
 *  @file
 *
 *  @brief      Main include file for applications using OTF2.
 */

#include <otf2/OTF2_Reader.h>
//#include <otf2/otf2_compiler.h>

#ifndef __otf2_deprecated__
#define __otf2_deprecated__
#endif

#include "pallas/pallas.h"
#include "pallas/pallas_archive.h"
#include "pallas/pallas_write.h"
#include "pallas/pallas_read.h"

struct OTF2_GlobalDefWriter_struct {
  struct GlobalArchive * archive;
};

struct OTF2_DefWriter_struct {
  OTF2_LocationRef locationRef;
  struct Archive* archive;

  struct ThreadWriter* thread_writer;
};

struct OTF2_EvtWriter_struct {
  OTF2_LocationRef locationRef;
  struct Archive* archive;
  struct ThreadWriter* thread_writer;
};

struct OTF2_Archive_struct {
  struct Archive* archive;

  OTF2_GlobalDefWriter* globalDefWriter;

  OTF2_DefWriter** def_writers;
  OTF2_EvtWriter** evt_writers;
  int nb_locations;

  struct OTF2_LockingCallbacks lockingCallbacks;
  void* lockingData;

  struct OTF2_FlushCallbacks flushCallbacks;
  void* flushData;

  pthread_mutex_t lock;
};


struct OTF2_GlobalDefReaderCallbacks_struct {
  OTF2_GlobalDefReaderCallback_Unknown OTF2_GlobalDefReaderCallback_Unknown_callback;
  OTF2_GlobalDefReaderCallback_ClockProperties OTF2_GlobalDefReaderCallback_ClockProperties_callback;
  OTF2_GlobalDefReaderCallback_Paradigm OTF2_GlobalDefReaderCallback_Paradigm_callback;
  OTF2_GlobalDefReaderCallback_ParadigmProperty OTF2_GlobalDefReaderCallback_ParadigmProperty_callback;
  OTF2_GlobalDefReaderCallback_IoParadigm OTF2_GlobalDefReaderCallback_IoParadigm_callback;
  OTF2_GlobalDefReaderCallback_String OTF2_GlobalDefReaderCallback_String_callback;
  OTF2_GlobalDefReaderCallback_Attribute OTF2_GlobalDefReaderCallback_Attribute_callback;
  OTF2_GlobalDefReaderCallback_SystemTreeNode OTF2_GlobalDefReaderCallback_SystemTreeNode_callback;
  OTF2_GlobalDefReaderCallback_LocationGroup OTF2_GlobalDefReaderCallback_LocationGroup_callback;
  OTF2_GlobalDefReaderCallback_Location OTF2_GlobalDefReaderCallback_Location_callback;
  OTF2_GlobalDefReaderCallback_Region OTF2_GlobalDefReaderCallback_Region_callback;
  OTF2_GlobalDefReaderCallback_Callsite OTF2_GlobalDefReaderCallback_Callsite_callback;
  OTF2_GlobalDefReaderCallback_Callpath OTF2_GlobalDefReaderCallback_Callpath_callback;
  OTF2_GlobalDefReaderCallback_Group OTF2_GlobalDefReaderCallback_Group_callback;
  OTF2_GlobalDefReaderCallback_MetricMember OTF2_GlobalDefReaderCallback_MetricMember_callback;
  OTF2_GlobalDefReaderCallback_MetricClass OTF2_GlobalDefReaderCallback_MetricClass_callback;
  OTF2_GlobalDefReaderCallback_MetricInstance OTF2_GlobalDefReaderCallback_MetricInstance_callback;
  OTF2_GlobalDefReaderCallback_Comm OTF2_GlobalDefReaderCallback_Comm_callback;
  OTF2_GlobalDefReaderCallback_Parameter OTF2_GlobalDefReaderCallback_Parameter_callback;
  OTF2_GlobalDefReaderCallback_RmaWin OTF2_GlobalDefReaderCallback_RmaWin_callback;
  OTF2_GlobalDefReaderCallback_MetricClassRecorder OTF2_GlobalDefReaderCallback_MetricClassRecorder_callback;
  OTF2_GlobalDefReaderCallback_SystemTreeNodeProperty OTF2_GlobalDefReaderCallback_SystemTreeNodeProperty_callback;
  OTF2_GlobalDefReaderCallback_SystemTreeNodeDomain OTF2_GlobalDefReaderCallback_SystemTreeNodeDomain_callback;
  OTF2_GlobalDefReaderCallback_LocationGroupProperty OTF2_GlobalDefReaderCallback_LocationGroupProperty_callback;
  OTF2_GlobalDefReaderCallback_LocationProperty OTF2_GlobalDefReaderCallback_LocationProperty_callback;
  OTF2_GlobalDefReaderCallback_CartDimension OTF2_GlobalDefReaderCallback_CartDimension_callback;
  OTF2_GlobalDefReaderCallback_CartTopology OTF2_GlobalDefReaderCallback_CartTopology_callback;
  OTF2_GlobalDefReaderCallback_CartCoordinate OTF2_GlobalDefReaderCallback_CartCoordinate_callback;
  OTF2_GlobalDefReaderCallback_SourceCodeLocation OTF2_GlobalDefReaderCallback_SourceCodeLocation_callback;
  OTF2_GlobalDefReaderCallback_CallingContext OTF2_GlobalDefReaderCallback_CallingContext_callback;
  OTF2_GlobalDefReaderCallback_CallingContextProperty OTF2_GlobalDefReaderCallback_CallingContextProperty_callback;
  OTF2_GlobalDefReaderCallback_InterruptGenerator OTF2_GlobalDefReaderCallback_InterruptGenerator_callback;
  OTF2_GlobalDefReaderCallback_IoFileProperty OTF2_GlobalDefReaderCallback_IoFileProperty_callback;
  OTF2_GlobalDefReaderCallback_IoRegularFile OTF2_GlobalDefReaderCallback_IoRegularFile_callback;
  OTF2_GlobalDefReaderCallback_IoDirectory OTF2_GlobalDefReaderCallback_IoDirectory_callback;
  OTF2_GlobalDefReaderCallback_IoHandle OTF2_GlobalDefReaderCallback_IoHandle_callback;
  OTF2_GlobalDefReaderCallback_IoPreCreatedHandleState OTF2_GlobalDefReaderCallback_IoPreCreatedHandleState_callback;
  OTF2_GlobalDefReaderCallback_CallpathParameter OTF2_GlobalDefReaderCallback_CallpathParameter_callback;
  OTF2_GlobalDefReaderCallback_InterComm OTF2_GlobalDefReaderCallback_InterComm_callback;
};

struct OTF2_GlobalDefReader_struct {
  struct GlobalArchive *archive;

  OTF2_GlobalDefReaderCallbacks callbacks;
  void* user_data;
};

struct OTF2_DefReader_struct {
  OTF2_LocationRef location;
  PALLAS(ThreadReader)* thread_reader;

  OTF2_DefReaderCallbacks callbacks;
  void* user_data;
};

struct OTF2_EvtReader_struct {
  OTF2_LocationRef location;
  PALLAS(ThreadReader) * thread_reader;
};

struct OTF2_GlobalEvtReader_struct {
  struct OTF2_Reader_struct * otf2_reader;
};

struct OTF2_Reader_struct {
  struct GlobalArchive *archive;

  OTF2_GlobalEvtReader* global_evt_reader;
  OTF2_GlobalDefReader* global_def_reader;

  int nb_locations;
  PALLAS(ThreadId)* locations;
  int *selected_locations;

  struct OTF2_EvtReader_struct *evt_readers;
  struct OTF2_DefReader_struct *def_readers;
  PALLAS(ThreadReader) **thread_readers;

};

#endif /* !OTF2_H */

/* -*-
   mode: c;
   c-file-style: "k&r";
   c-basic-offset 2;
   tab-width 2 ;
   indent-tabs-mode nil
   -*- */

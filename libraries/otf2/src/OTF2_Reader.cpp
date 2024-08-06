#include <stdio.h>
#include <stdlib.h>

#include "pallas/pallas.h"
#include "pallas/pallas_log.h"
#include "pallas/pallas_storage.h"
#include "otf2/OTF2_Reader.h"
#include "otf2/otf2.h"

static void init_OTF2_GlobalDefReader(OTF2_Reader* reader, OTF2_GlobalDefReader* global_def_reader) {
  memset(global_def_reader, 0, sizeof(OTF2_GlobalDefReader));
  global_def_reader->archive = reader->archive;
}

static void init_OTF2_GlobalEvtReader(OTF2_Reader* reader, OTF2_GlobalEvtReader* global_evt_reader) {
  memset(global_evt_reader, 0, sizeof(OTF2_GlobalEvtReader));
  global_evt_reader->otf2_reader = reader;
}

OTF2_Reader* OTF2_Reader_Open(const char* anchorFilePath) {

  OTF2_Reader* reader = (OTF2_Reader*)malloc(sizeof(OTF2_Reader));
  memset(reader, 0, sizeof(OTF2_Reader));

  reader->archive = (struct GlobalArchive*) pallas_global_archive_new();
  PALLAS(GlobalArchive)* archive = (PALLAS(GlobalArchive)*)reader->archive;

  pallasReadGlobalArchive(archive, anchorFilePath);

  for (int i = 0; i < archive->nb_archives; i++) {
    printf("Archive %d: %d threads\n", i, archive->archive_list[i]->nb_threads);
    reader->nb_locations += archive->archive_list[i]->nb_threads;
    int nb_threads = archive->archive_list[i]->nb_threads;
    for(int j = 0; j<nb_threads; j++) {
      pallas::Thread* t = archive->archive_list[i]->threads[i];
      printf("\tthread %d\n", t->id);
    }
  }
  reader->locations = new PALLAS(ThreadId)[reader->nb_locations];
  reader->evt_readers = (struct OTF2_EvtReader_struct*) calloc(reader->nb_locations, sizeof(struct OTF2_EvtReader_struct));
  reader->def_readers = (struct OTF2_DefReader_struct*) calloc(reader->nb_locations, sizeof(struct OTF2_DefReader_struct));
  reader->thread_readers = (pallas::ThreadReader**) calloc(reader->nb_locations, sizeof(pallas::ThreadReader*));

  for(int i = 0; i< reader->nb_locations; i++) {
    reader->locations[i] =   archive->locations[i].id;
    reader->evt_readers->location = OTF2_UNDEFINED_LOCATION;
    reader->def_readers->location = OTF2_UNDEFINED_LOCATION;
  }


  reader->selected_locations = (int*) calloc(reader->nb_locations, sizeof(int));
  reader->global_def_reader = (OTF2_GlobalDefReader*)malloc(sizeof(OTF2_GlobalDefReader));
  init_OTF2_GlobalDefReader(reader, reader->global_def_reader);
  reader->global_evt_reader = (OTF2_GlobalEvtReader*)malloc(sizeof(OTF2_GlobalEvtReader));
  init_OTF2_GlobalEvtReader(reader, reader->global_evt_reader);
  
  return reader;
}

OTF2_ErrorCode OTF2_Reader_Close(OTF2_Reader* reader) {
  NOT_IMPLEMENTED;
}

OTF2_ErrorCode OTF2_Reader_SetHint(OTF2_Reader* reader, OTF2_Hint hint, void* value) {
  NOT_IMPLEMENTED;
}

OTF2_ErrorCode OTF2_Reader_SetCollectiveCallbacks(OTF2_Reader* reader,
                                                  const OTF2_CollectiveCallbacks* collectiveCallbacks,
                                                  void* collectiveData,
                                                  OTF2_CollectiveContext* globalCommContext,
                                                  OTF2_CollectiveContext* localCommContext) {
  NOT_IMPLEMENTED;
}

OTF2_ErrorCode OTF2_Reader_SetSerialCollectiveCallbacks(OTF2_Reader* reader) {
  NOT_IMPLEMENTED;
}

OTF2_ErrorCode OTF2_Reader_SetLockingCallbacks(OTF2_Reader* reader,
                                               const OTF2_LockingCallbacks* lockingCallbacks,
                                               void* lockingData) {
  NOT_IMPLEMENTED;
}

OTF2_ErrorCode OTF2_Reader_RegisterEvtCallbacks(OTF2_Reader* reader,
                                                OTF2_EvtReader* evtReader,
                                                const OTF2_EvtReaderCallbacks* callbacks,
                                                void* userData) {
  NOT_IMPLEMENTED;
}

OTF2_ErrorCode OTF2_Reader_RegisterGlobalEvtCallbacks(OTF2_Reader* reader,
                                                      OTF2_GlobalEvtReader* evtReader,
                                                      const OTF2_GlobalEvtReaderCallbacks* callbacks,
                                                      void* userData) {
  NOT_IMPLEMENTED;
}

OTF2_ErrorCode OTF2_Reader_RegisterDefCallbacks(OTF2_Reader* reader,
                                                OTF2_DefReader* defReader,
                                                const OTF2_DefReaderCallbacks* callbacks,
                                                void* userData) {
  memcpy(&defReader->callbacks, callbacks, sizeof(OTF2_DefReaderCallbacks));
  defReader->user_data = userData;
  return OTF2_SUCCESS;
}

OTF2_ErrorCode OTF2_Reader_RegisterGlobalDefCallbacks(OTF2_Reader* reader,
                                                      OTF2_GlobalDefReader* defReader,
                                                      const OTF2_GlobalDefReaderCallbacks* callbacks,
                                                      void* userData) {
  memcpy(&defReader->callbacks, callbacks, sizeof(OTF2_GlobalDefReaderCallbacks));
  defReader->user_data = userData;
  return OTF2_SUCCESS;
}

OTF2_ErrorCode OTF2_Reader_RegisterSnapCallbacks(OTF2_Reader* reader,
                                                 OTF2_SnapReader* snapReader,
                                                 const OTF2_SnapReaderCallbacks* callbacks,
                                                 void* userData) {
  NOT_IMPLEMENTED;
}

OTF2_ErrorCode OTF2_Reader_RegisterGlobalSnapCallbacks(OTF2_Reader* reader,
                                                       OTF2_GlobalSnapReader* evtReader,
                                                       const OTF2_GlobalSnapReaderCallbacks* callbacks,
                                                       void* userData) {
  NOT_IMPLEMENTED;
}

OTF2_ErrorCode OTF2_Reader_RegisterMarkerCallbacks(OTF2_Reader* reader,
                                                   OTF2_MarkerReader* markerReader,
                                                   const OTF2_MarkerReaderCallbacks* callbacks,
                                                   void* userData) {
  NOT_IMPLEMENTED;
}

OTF2_ErrorCode OTF2_Reader_ReadLocalEvents(OTF2_Reader* reader,
                                           OTF2_EvtReader* evtReader,
                                           uint64_t eventsToRead,
                                           uint64_t* eventsRead) {
  NOT_IMPLEMENTED;
}

OTF2_ErrorCode OTF2_Reader_ReadAllLocalEvents(OTF2_Reader* reader, OTF2_EvtReader* evtReader, uint64_t* eventsRead) {
  NOT_IMPLEMENTED;
}

OTF2_ErrorCode OTF2_Reader_ReadLocalEventsBackward(OTF2_Reader* reader,
                                                   OTF2_EvtReader* evtReader,
                                                   uint64_t eventsToRead,
                                                   uint64_t* eventsRead) {
  NOT_IMPLEMENTED;
}

OTF2_ErrorCode OTF2_Reader_ReadGlobalEvent(OTF2_Reader* reader, OTF2_GlobalEvtReader* evtReader) {
  NOT_IMPLEMENTED;
}

OTF2_ErrorCode OTF2_Reader_HasGlobalEvent(OTF2_Reader* reader, OTF2_GlobalEvtReader* evtReader, int* flag) {
  NOT_IMPLEMENTED;
}

OTF2_ErrorCode OTF2_Reader_ReadGlobalEvents(OTF2_Reader* reader,
                                            OTF2_GlobalEvtReader* evtReader,
                                            uint64_t eventsToRead,
                                            uint64_t* eventsRead) {
  NOT_IMPLEMENTED;
}

OTF2_ErrorCode OTF2_Reader_ReadAllGlobalEvents(OTF2_Reader* reader,
                                               OTF2_GlobalEvtReader* evtReader,
                                               uint64_t* eventsRead) {
  NOT_IMPLEMENTED;
}

OTF2_ErrorCode OTF2_Reader_ReadLocalDefinitions(OTF2_Reader* reader,
                                                OTF2_DefReader* defReader,
                                                uint64_t definitionsToRead,
                                                uint64_t* definitionsRead) {
  PALLAS(Archive)* archive = (PALLAS(Archive)*)defReader->thread_reader->archive;
  int nb_definition_read = 0;

#define CHECK_OTF2_CALLBACK_SUCCESS(_f_) do {			\
    if(nb_definition_read >= definitionsToRead) goto out;	\
    if(_f_ != OTF2_CALLBACK_SUCCESS) goto out;			\
    nb_definition_read++;					\
  } while(0)

  
  // TODO: implement local definition
  // out:
  *definitionsRead = nb_definition_read;
  pallas_assert(*definitionsRead <= definitionsToRead);
  return OTF2_SUCCESS;
}

OTF2_ErrorCode OTF2_Reader_ReadAllLocalDefinitions(OTF2_Reader* reader,
                                                   OTF2_DefReader* defReader,
                                                   uint64_t* definitionsRead) {
  return OTF2_Reader_ReadLocalDefinitions(reader, defReader, UINT64_MAX, definitionsRead);
}

OTF2_ErrorCode OTF2_Reader_ReadGlobalDefinitions(OTF2_Reader* reader,
                                                 OTF2_GlobalDefReader* defReader,
                                                 uint64_t definitionsToRead,
                                                 uint64_t* definitionsRead) {
  PALLAS(GlobalArchive)* archive = (PALLAS(GlobalArchive)*)reader->archive;
  int nb_definition_read = 0;

#define CHECK_OTF2_CALLBACK_SUCCESS(_f_) do {			  \
    if(nb_definition_read >= definitionsToRead) goto out; \
    if(_f_ != OTF2_CALLBACK_SUCCESS) goto out;		\
    nb_definition_read++;			\
  } while(0)

  if(defReader->callbacks.OTF2_GlobalDefReaderCallback_String_callback) {
    for ( const auto &string : archive->definitions.strings ) {
      CHECK_OTF2_CALLBACK_SUCCESS(defReader->callbacks.OTF2_GlobalDefReaderCallback_String_callback
				  (defReader->user_data,
				   string.first,
				   string.second.str));
    }
  }

  if(defReader->callbacks.OTF2_GlobalDefReaderCallback_Region_callback) {
    for ( const auto &region : archive->definitions.regions ) {
      CHECK_OTF2_CALLBACK_SUCCESS(defReader->callbacks.OTF2_GlobalDefReaderCallback_Region_callback
				  (defReader->user_data,
				   region.first,
				   region.second.string_ref, // name	
				   OTF2_UNDEFINED_STRING, // canonicalName
				   OTF2_UNDEFINED_STRING, // description
				   OTF2_REGION_ROLE_UNKNOWN, // regionRole
				   OTF2_PARADIGM_UNKNOWN,    // paradigm
				   OTF2_REGION_FLAG_NONE,    // regionFlags
				   OTF2_UNDEFINED_STRING, // sourceFile
				   0,			  // beginLineNumber
				   0));			  // endLineNumber
    }
  }

  if(defReader->callbacks.OTF2_GlobalDefReaderCallback_Attribute_callback) {
    for ( const auto &attr : archive->definitions.attributes ) {
      CHECK_OTF2_CALLBACK_SUCCESS(defReader->callbacks.OTF2_GlobalDefReaderCallback_Attribute_callback
				  (defReader->user_data,
				   attr.first,
				   attr.second.name,
				   attr.second.description,
				   attr.second.type));
    }
  }

  if(defReader->callbacks.OTF2_GlobalDefReaderCallback_LocationGroup_callback) {
    for ( const auto &lg : archive->location_groups ) {
      CHECK_OTF2_CALLBACK_SUCCESS(defReader->callbacks.OTF2_GlobalDefReaderCallback_LocationGroup_callback
				  (defReader->user_data,
				   lg.id, // self
				   lg.name, // name
				   OTF2_LOCATION_GROUP_TYPE_UNKNOWN, // locationGroupType
				   OTF2_UNDEFINED_SYSTEM_TREE_NODE, // systemTreeParent
				   lg.parent // creatingLocationGroup
				   ));
    }
  }

  if(defReader->callbacks.OTF2_GlobalDefReaderCallback_Location_callback) {
    for ( const auto &loc : archive->locations ) {
      CHECK_OTF2_CALLBACK_SUCCESS(defReader->callbacks.OTF2_GlobalDefReaderCallback_Location_callback
				  (defReader->user_data,
				   loc.id, // self
				   loc.name, // name
				   OTF2_LOCATION_TYPE_UNKNOWN, // locationType
				   0, // numberOfEvents
				   loc.parent // locationGroup
				   ));
    }
  }

  // todo: handle the other definitions

 out:
  *definitionsRead = nb_definition_read;
  pallas_assert(*definitionsRead <= definitionsToRead);
  return OTF2_SUCCESS;
}

OTF2_ErrorCode OTF2_Reader_ReadAllGlobalDefinitions(OTF2_Reader* reader,
                                                    OTF2_GlobalDefReader* defReader,
                                                    uint64_t* definitionsRead) {
  return OTF2_Reader_ReadGlobalDefinitions(reader, defReader, UINT64_MAX, definitionsRead);
}

OTF2_ErrorCode OTF2_Reader_ReadLocalSnapshots(OTF2_Reader* reader,
                                              OTF2_SnapReader* snapReader,
                                              uint64_t recordsToRead,
                                              uint64_t* recordsRead) {
  NOT_IMPLEMENTED;
}

OTF2_ErrorCode OTF2_Reader_ReadAllLocalSnapshots(OTF2_Reader* reader,
                                                 OTF2_SnapReader* snapReader,
                                                 uint64_t* recordsRead) {
  NOT_IMPLEMENTED;
}

OTF2_ErrorCode OTF2_Reader_ReadGlobalSnapshots(OTF2_Reader* reader,
                                               OTF2_GlobalSnapReader* snapReader,
                                               uint64_t recordsToRead,
                                               uint64_t* recordsRead) {
  NOT_IMPLEMENTED;
}

OTF2_ErrorCode OTF2_Reader_ReadAllGlobalSnapshots(OTF2_Reader* reader,
                                                  OTF2_GlobalSnapReader* snapReader,
                                                  uint64_t* recordsRead) {
  NOT_IMPLEMENTED;
}

OTF2_ErrorCode OTF2_Reader_ReadMarkers(OTF2_Reader* reader,
                                       OTF2_MarkerReader* markerReader,
                                       uint64_t markersToRead,
                                       uint64_t* markersRead) {
  NOT_IMPLEMENTED;
}

OTF2_ErrorCode OTF2_Reader_ReadAllMarkers(OTF2_Reader* reader, OTF2_MarkerReader* markerReader, uint64_t* markersRead) {
  NOT_IMPLEMENTED;
}

OTF2_EvtReader* OTF2_Reader_GetEvtReader(OTF2_Reader* reader, OTF2_LocationRef location) {
  for(int i = 0; i<reader->nb_locations; i++) {
    if(reader->locations[i] == location &&
       reader->selected_locations[i]) {

      if(reader->evt_readers[i].location == OTF2_UNDEFINED_LOCATION) {
	// This evt readers has not been open yet.
	reader->evt_readers[i].location = location;
      }
      if(! reader->thread_readers[i]) {
	PALLAS(GlobalArchive)* global_archive = (PALLAS(GlobalArchive)*)reader->archive;
	PALLAS(Archive)* thread_archive = global_archive->getArchiveFromLocation(location);
	reader->thread_readers[i] = new	pallas::ThreadReader(thread_archive, location, 0);
      }
      if(! reader->evt_readers[i].thread_reader)
	reader->evt_readers[i].thread_reader = reader->thread_readers[i];

      return &reader->evt_readers[i];
    }
  }
  return NULL;
}

OTF2_GlobalEvtReader* OTF2_Reader_GetGlobalEvtReader(OTF2_Reader* reader) {

  return reader->global_evt_reader;
}

OTF2_DefReader* OTF2_Reader_GetDefReader(OTF2_Reader* reader, OTF2_LocationRef location) {
  for(int i = 0; i<reader->nb_locations; i++) {
    if(reader->locations[i] == location &&
       reader->selected_locations[i]) {

      if(! reader->thread_readers[i]) {
	PALLAS(GlobalArchive)* global_archive = (PALLAS(GlobalArchive)*)reader->archive;
	PALLAS(Archive)* thread_archive = global_archive->getArchiveFromLocation(location);
	reader->thread_readers[i] = new	pallas::ThreadReader(thread_archive, location, 0);
      }

      if(reader->def_readers[i].location == OTF2_UNDEFINED_LOCATION) {
	// This def readers has not been open yet.
	memset(&reader->def_readers[i], 0, sizeof(struct OTF2_DefReader_struct));
	reader->def_readers[i].location = location;
	reader->def_readers[i].thread_reader = reader->thread_readers[i];
      }

      return &reader->def_readers[i];
    }
  }
  return NULL;
}

OTF2_GlobalDefReader* OTF2_Reader_GetGlobalDefReader(OTF2_Reader* reader) {

  return reader->global_def_reader;
}

OTF2_SnapReader* OTF2_Reader_GetSnapReader(OTF2_Reader* reader, OTF2_LocationRef location) {
  NOT_IMPLEMENTED;
}

OTF2_GlobalSnapReader* OTF2_Reader_GetGlobalSnapReader(OTF2_Reader* reader) {
  NOT_IMPLEMENTED;
}

OTF2_ThumbReader* OTF2_Reader_GetThumbReader(OTF2_Reader* reader, uint32_t number) {
  NOT_IMPLEMENTED;
}

OTF2_MarkerReader* OTF2_Reader_GetMarkerReader(OTF2_Reader* reader) {
  NOT_IMPLEMENTED;
}

OTF2_MarkerWriter* OTF2_Reader_GetMarkerWriter(OTF2_Reader* reader) {
  NOT_IMPLEMENTED;
}

OTF2_ErrorCode OTF2_Reader_CloseEvtReader(OTF2_Reader* reader, OTF2_EvtReader* evtReader) {
  NOT_IMPLEMENTED;
}

OTF2_ErrorCode OTF2_Reader_CloseGlobalEvtReader(OTF2_Reader* reader, OTF2_GlobalEvtReader* globalEvtReader) {
  NOT_IMPLEMENTED;
}

OTF2_ErrorCode OTF2_Reader_CloseDefReader(OTF2_Reader* reader, OTF2_DefReader* defReader) {
  // TODO: free/unitinialize def_reader?
  return OTF2_SUCCESS;
}

OTF2_ErrorCode OTF2_Reader_CloseGlobalDefReader(OTF2_Reader* reader, OTF2_GlobalDefReader* globalDefReader) {
  // TODO: free reader->global_def_reader?
  return OTF2_SUCCESS;
}

OTF2_ErrorCode OTF2_Reader_CloseSnapReader(OTF2_Reader* reader, OTF2_SnapReader* snapReader) {
  NOT_IMPLEMENTED;
}

OTF2_ErrorCode OTF2_Reader_CloseGlobalSnapReader(OTF2_Reader* reader, OTF2_GlobalSnapReader* globalSnapReader) {
  NOT_IMPLEMENTED;
}

OTF2_ErrorCode OTF2_Reader_CloseThumbReader(OTF2_Reader* reader, OTF2_ThumbReader* thumbReader) {
  NOT_IMPLEMENTED;
}

OTF2_ErrorCode OTF2_Reader_CloseMarkerReader(OTF2_Reader* reader, OTF2_MarkerReader* markerReader) {
  NOT_IMPLEMENTED;
}

OTF2_ErrorCode OTF2_Reader_CloseMarkerWriter(OTF2_Reader* reader, OTF2_MarkerWriter* markerWriter) {
  NOT_IMPLEMENTED;
}

OTF2_ErrorCode OTF2_Reader_GetVersion(OTF2_Reader* reader, uint8_t* major, uint8_t* minor, uint8_t* bugfix) {
  NOT_IMPLEMENTED;
}

OTF2_ErrorCode OTF2_Reader_GetChunkSize(OTF2_Reader* reader,
                                        uint64_t* chunkSizeEvents,
                                        uint64_t* chunkSizeDefinitions) {
  NOT_IMPLEMENTED;
}

OTF2_ErrorCode OTF2_Reader_GetFileSubstrate(OTF2_Reader* reader, OTF2_FileSubstrate* substrate) {
  NOT_IMPLEMENTED;
}

OTF2_ErrorCode OTF2_Reader_GetCompression(OTF2_Reader* reader, OTF2_Compression* compression) {
  NOT_IMPLEMENTED;
}

OTF2_ErrorCode OTF2_Reader_GetNumberOfLocations(OTF2_Reader* reader, uint64_t* numberOfLocations) {
  *numberOfLocations = reader->nb_locations;
  return OTF2_SUCCESS;
}

OTF2_ErrorCode OTF2_Reader_GetNumberOfGlobalDefinitions(OTF2_Reader* reader, uint64_t* numberOfDefinitions) {
  NOT_IMPLEMENTED;
}

OTF2_ErrorCode OTF2_Reader_GetMachineName(OTF2_Reader* reader, char** machineName) {
  NOT_IMPLEMENTED;
}

OTF2_ErrorCode OTF2_Reader_GetCreator(OTF2_Reader* reader, char** creator) {
  NOT_IMPLEMENTED;
}

OTF2_ErrorCode OTF2_Reader_GetDescription(OTF2_Reader* reader, char** description) {
  NOT_IMPLEMENTED;
}

OTF2_ErrorCode OTF2_Reader_GetPropertyNames(OTF2_Reader* reader, uint32_t* numberOfProperties, char*** names) {
  NOT_IMPLEMENTED;
}

OTF2_ErrorCode OTF2_Reader_GetProperty(OTF2_Reader* reader, const char* name, char** value) {
  NOT_IMPLEMENTED;
}

OTF2_ErrorCode OTF2_Reader_GetBoolProperty(OTF2_Reader* reader, const char* name, bool* value) {
  NOT_IMPLEMENTED;
}

OTF2_ErrorCode OTF2_Reader_GetTraceId(OTF2_Reader* reader, uint64_t* id) {
  NOT_IMPLEMENTED;
}

OTF2_ErrorCode OTF2_Reader_GetNumberOfSnapshots(OTF2_Reader* reader, uint32_t* number) {
  NOT_IMPLEMENTED;
}

OTF2_ErrorCode OTF2_Reader_GetNumberOfThumbnails(OTF2_Reader* reader, uint32_t* number) {
  NOT_IMPLEMENTED;
}

OTF2_ErrorCode OTF2_Reader_OpenEvtFiles(OTF2_Reader* reader) {
  return OTF2_SUCCESS;
}

OTF2_ErrorCode OTF2_Reader_CloseEvtFiles(OTF2_Reader* reader) {
  return OTF2_SUCCESS;
}

OTF2_ErrorCode OTF2_Reader_OpenDefFiles(OTF2_Reader* reader) {
  return OTF2_SUCCESS;
}

OTF2_ErrorCode OTF2_Reader_CloseDefFiles(OTF2_Reader* reader) {
  return OTF2_SUCCESS;
}

OTF2_ErrorCode OTF2_Reader_OpenSnapFiles(OTF2_Reader* reader) {
  return OTF2_SUCCESS;
}

OTF2_ErrorCode OTF2_Reader_CloseSnapFiles(OTF2_Reader* reader) {
  return OTF2_SUCCESS;
}

OTF2_ErrorCode OTF2_Reader_SelectLocation(OTF2_Reader* reader, OTF2_LocationRef location) {
  for(int i=0; i<reader->nb_locations; i++) {
    if(reader->locations[i] == location) {
      reader->selected_locations[i] = 1;
    }
  }

  return OTF2_SUCCESS;
}

/** @brief Determines if this reader is the primary archive which handles the
 *  global archive state (anchor file, global definitions, marker, ...).
 *
 *  @param reader       Reader handle.
 *  @param[out] result  Storage for the result.
 *
 *  @return @eref{OTF2_SUCCESS} if successful, an error code if an error occurs.
 */
OTF2_ErrorCode OTF2_Reader_IsPrimary(OTF2_Reader* reader, bool* result) {
  NOT_IMPLEMENTED;
}

OTF2_ErrorCode OTF2_Reader_IsMaster(OTF2_Reader* reader, bool* result) {
  NOT_IMPLEMENTED;
}

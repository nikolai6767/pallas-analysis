/*
 * Copyright (C) Telecom SudParis
 * See LICENSE in top-level directory.
 */

#include <stdio.h>
#include <stdlib.h>
#include <inttypes.h>
#include <string.h>
#include "otf2/otf2.h"
#include "pallas/pallas_log.h"

#define check_status( status, ...) if(status != OTF2_SUCCESS) {	\
    fprintf(stderr,  __VA_ARGS__);				\
    abort();							\
  }

#define check_pointer( ptr, ...) if(ptr == NULL) {		\
    fprintf(stderr,  __VA_ARGS__);				\
    abort();							\
  }

struct otf2_print_user_data {
  size_t nb_locations;
  uint64_t* locations;
};

void _set_callbacks(OTF2_Reader* reader) {

}

OTF2_CallbackCode print_global_def_string(void *userData, OTF2_StringRef self, const char *string) {
  printf("Global_def_string(userData=%p, self=%d, string='%s');\n", userData, self, string);
  return OTF2_CALLBACK_SUCCESS;
}


OTF2_CallbackCode print_global_def_region(void *userData,
					  OTF2_RegionRef self,
					  OTF2_StringRef name,
					  OTF2_StringRef canonicalName,
					  OTF2_StringRef description,
					  OTF2_RegionRole regionRole,
					  OTF2_Paradigm paradigm,
					  OTF2_RegionFlag regionFlags,
					  OTF2_StringRef sourceFile,
					  uint32_t beginLineNumber,
					  uint32_t endLineNumber) {
  printf("Global_def_region(userData=%p, self=%d, name=%d, canonicalName=%d, description=%d, regionRole=%d, paradigm=%d, regionFlags=%d, sourceFile=%d, beginLineNumber=%d, endLineNumber=%d)\n",
	 userData,
	 self,
	 name,
	 canonicalName,
	 description,
	 regionRole,
	 paradigm,
	 regionFlags,
	 sourceFile,
	 beginLineNumber,
	 endLineNumber);
  return OTF2_CALLBACK_SUCCESS;

}

OTF2_CallbackCode print_global_def_attribute(void *userData,
					     OTF2_AttributeRef self,
					     OTF2_StringRef name,
					     OTF2_StringRef description,
					     OTF2_Type type) {
  printf("Global_def_attribute(userData=%p, self=%d, name=%d, description=%d, type=%d)\n",
	 userData,
	 self,
	 name,
	 description,
	 type);
  return OTF2_CALLBACK_SUCCESS;
}

OTF2_CallbackCode print_global_def_location_group(void *userData,
						  OTF2_LocationGroupRef self,
						  OTF2_StringRef name,
						  OTF2_LocationGroupType locationGroupType,
						  OTF2_SystemTreeNodeRef systemTreeParent,
						  OTF2_LocationGroupRef creatingLocationGroup) {
  printf("Global_def_location_group(userData=%p, self=%d, name=%d, locationGroupType=%d, systemTreeParent=%d, creatingLocationGroup=%d)\n",
	 userData,
	 self,
	 name,
	 locationGroupType,
	 systemTreeParent,
	 creatingLocationGroup);
  return OTF2_CALLBACK_SUCCESS;
}

OTF2_CallbackCode print_global_def_location(void *userData,
					    OTF2_LocationRef self,
					    OTF2_StringRef name,
					    OTF2_LocationType locationType,
					    uint64_t numberOfEvents,
					    OTF2_LocationGroupRef locationGroup) {
  printf("Global_def_location(userData=%p, self=%ld, name=%d, locationType=%d, numberOfEvents=%ld, locationGroup=%d)\n",
	 userData,
	 self,
	 name,
	 locationType,
	 numberOfEvents,
	 locationGroup);

  struct otf2_print_user_data *user_data = userData;
  int index = user_data->nb_locations++;
  user_data->locations = realloc(user_data->locations, sizeof(uint64_t)*user_data->nb_locations);
  pallas_assert(user_data->locations);
  user_data->locations[index] = self;
  return OTF2_CALLBACK_SUCCESS;
  
}


OTF2_GlobalDefReaderCallbacks* otf2_print_define_global_def_callbacks(OTF2_Reader *reader) {
  OTF2_GlobalDefReaderCallbacks* def_callbacks = OTF2_GlobalDefReaderCallbacks_New();

  OTF2_GlobalDefReaderCallbacks_SetStringCallback( def_callbacks, print_global_def_string );
  OTF2_GlobalDefReaderCallbacks_SetAttributeCallback( def_callbacks, print_global_def_attribute );
//  OTF2_GlobalDefReaderCallbacks_SetSystemTreeNodeCallback( def_callbacks, print_global_def_system_tree_node );
  OTF2_GlobalDefReaderCallbacks_SetLocationGroupCallback( def_callbacks, print_global_def_location_group );
  OTF2_GlobalDefReaderCallbacks_SetLocationCallback( def_callbacks, print_global_def_location );
  OTF2_GlobalDefReaderCallbacks_SetRegionCallback( def_callbacks, print_global_def_region );
//  OTF2_GlobalDefReaderCallbacks_SetGroupCallback( def_callbacks, print_global_def_group );

  return def_callbacks;
}

OTF2_CallbackCode print_enter(OTF2_LocationRef locationID,
			      OTF2_TimeStamp time,
			      void *userData,
			      OTF2_AttributeList *attributeList,
			      OTF2_RegionRef region) {
  printf("ENTER(locationID=%d, time=%d, userData=%p, attributeList=%p, region=%d)\n",
	 locationID, time, userData, attributeList, region);
  return OTF2_CALLBACK_SUCCESS;
}

OTF2_CallbackCode print_leave(OTF2_LocationRef locationID,
			      OTF2_TimeStamp time,
			      void *userData,
			      OTF2_AttributeList *attributeList,
			      OTF2_RegionRef region) {
  printf("LEAVE(locationID=%d, time=%d, userData=%p, attributeList=%p, region=%d)\n",
	 locationID, time, userData, attributeList, region);
  return OTF2_CALLBACK_SUCCESS;
}


OTF2_GlobalEvtReaderCallbacks* otf2_print_create_global_evt_callbacks(OTF2_Reader *reader) {
  OTF2_GlobalEvtReaderCallbacks* evt_callbacks = OTF2_GlobalEvtReaderCallbacks_New();

  OTF2_GlobalEvtReaderCallbacks_SetEnterCallback( evt_callbacks, print_enter );
  OTF2_GlobalEvtReaderCallbacks_SetLeaveCallback( evt_callbacks, print_leave );
//  OTF2_GlobalEvtReaderCallbacks_SetMpiSendCallback( evt_callbacks, print_mpi_send );
//  OTF2_GlobalEvtReaderCallbacks_SetMpiIsendCallback( evt_callbacks, print_mpi_isend );
//  OTF2_GlobalEvtReaderCallbacks_SetMpiIsendCompleteCallback( evt_callbacks, print_mpi_isend_complete );
//  OTF2_GlobalEvtReaderCallbacks_SetMpiIrecvRequestCallback( evt_callbacks, print_mpi_irecv_request );
//  OTF2_GlobalEvtReaderCallbacks_SetMpiRecvCallback( evt_callbacks, print_mpi_recv );
//  OTF2_GlobalEvtReaderCallbacks_SetMpiIrecvCallback( evt_callbacks, print_mpi_irecv );
//  OTF2_GlobalEvtReaderCallbacks_SetMpiRequestTestCallback( evt_callbacks, print_mpi_request_test );
//  OTF2_GlobalEvtReaderCallbacks_SetMpiRequestCancelledCallback( evt_callbacks, print_mpi_request_cancelled );
//  OTF2_GlobalEvtReaderCallbacks_SetMpiCollectiveBeginCallback( evt_callbacks, print_mpi_collective_begin );
//  OTF2_GlobalEvtReaderCallbacks_SetMpiCollectiveEndCallback( evt_callbacks, print_mpi_collective_end );
//  OTF2_GlobalEvtReaderCallbacks_SetOmpForkCallback( evt_callbacks, print_omp_fork );
//  OTF2_GlobalEvtReaderCallbacks_SetOmpJoinCallback( evt_callbacks, print_omp_join );
//  OTF2_GlobalEvtReaderCallbacks_SetOmpAcquireLockCallback( evt_callbacks, print_omp_acquire_lock );
//  OTF2_GlobalEvtReaderCallbacks_SetOmpReleaseLockCallback( evt_callbacks, print_omp_release_lock );
//  OTF2_GlobalEvtReaderCallbacks_SetOmpTaskCreateCallback( evt_callbacks, print_omp_task_create );
//  OTF2_GlobalEvtReaderCallbacks_SetOmpTaskSwitchCallback( evt_callbacks, print_omp_task_switch );
//  OTF2_GlobalEvtReaderCallbacks_SetOmpTaskCompleteCallback( evt_callbacks, print_omp_task_complete );

//  OTF2_GlobalEvtReaderCallbacks_SetThreadForkCallback( evt_callbacks, print_thread_fork );
//  OTF2_GlobalEvtReaderCallbacks_SetThreadJoinCallback( evt_callbacks, print_thread_join );
//  OTF2_GlobalEvtReaderCallbacks_SetThreadTeamBeginCallback( evt_callbacks, print_thread_team_begin );
//  OTF2_GlobalEvtReaderCallbacks_SetThreadTeamEndCallback( evt_callbacks, print_thread_team_end );
//  OTF2_GlobalEvtReaderCallbacks_SetThreadAcquireLockCallback( evt_callbacks, print_thread_acquire_lock );
//  OTF2_GlobalEvtReaderCallbacks_SetThreadReleaseLockCallback( evt_callbacks, print_thread_release_lock );
//  OTF2_GlobalEvtReaderCallbacks_SetThreadTaskCreateCallback( evt_callbacks, print_thread_task_create );
//  OTF2_GlobalEvtReaderCallbacks_SetThreadTaskSwitchCallback( evt_callbacks, print_thread_task_switch );
//  OTF2_GlobalEvtReaderCallbacks_SetThreadTaskCompleteCallback( evt_callbacks, print_thread_task_complete );
//  OTF2_GlobalEvtReaderCallbacks_SetThreadCreateCallback( evt_callbacks, print_thread_create );
//  OTF2_GlobalEvtReaderCallbacks_SetThreadBeginCallback( evt_callbacks, print_thread_begin );
//  OTF2_GlobalEvtReaderCallbacks_SetThreadWaitCallback( evt_callbacks, print_thread_wait );
//  OTF2_GlobalEvtReaderCallbacks_SetThreadEndCallback( evt_callbacks, print_thread_end );

//  OTF2_GlobalEvtReaderCallbacks_SetIoCreateHandleCallback( evt_callbacks, print_io_create_handle );
//  OTF2_GlobalEvtReaderCallbacks_SetIoDestroyHandleCallback( evt_callbacks, print_io_destroy_handle );
//  OTF2_GlobalEvtReaderCallbacks_SetIoDuplicateHandleCallback( evt_callbacks, print_io_duplicate_handle );
//  OTF2_GlobalEvtReaderCallbacks_SetIoSeekCallback( evt_callbacks, print_io_seek );
//  OTF2_GlobalEvtReaderCallbacks_SetIoChangeStatusFlagsCallback( evt_callbacks, print_io_change_status_flags );
//  OTF2_GlobalEvtReaderCallbacks_SetIoDeleteFileCallback( evt_callbacks, print_io_delete_file );
//  OTF2_GlobalEvtReaderCallbacks_SetIoOperationBeginCallback( evt_callbacks, print_io_operation_begin );
//  OTF2_GlobalEvtReaderCallbacks_SetIoOperationTestCallback( evt_callbacks, print_io_operation_test );
//  OTF2_GlobalEvtReaderCallbacks_SetIoOperationIssuedCallback( evt_callbacks, print_io_operation_issued );
//  OTF2_GlobalEvtReaderCallbacks_SetIoOperationCompleteCallback( evt_callbacks, print_io_operation_complete );
//  OTF2_GlobalEvtReaderCallbacks_SetIoOperationCancelledCallback( evt_callbacks, print_io_operation_cancelled );
//  OTF2_GlobalEvtReaderCallbacks_SetIoAcquireLockCallback( evt_callbacks, print_io_acquire_lock );
//  OTF2_GlobalEvtReaderCallbacks_SetIoReleaseLockCallback( evt_callbacks, print_io_release_lock );
//  OTF2_GlobalEvtReaderCallbacks_SetIoTryLockCallback( evt_callbacks, print_io_try_lock );
//  OTF2_GlobalEvtReaderCallbacks_SetProgramBeginCallback( evt_callbacks, print_program_begin );
//  OTF2_GlobalEvtReaderCallbacks_SetProgramEndCallback( evt_callbacks, print_program_end );
//  OTF2_GlobalEvtReaderCallbacks_SetNonBlockingCollectiveRequestCallback( evt_callbacks, print_non_blocking_collective_request );
//  OTF2_GlobalEvtReaderCallbacks_SetNonBlockingCollectiveCompleteCallback( evt_callbacks, print_non_blocking_collective_complete );
//  OTF2_GlobalEvtReaderCallbacks_SetCommCreateCallback( evt_callbacks, print_comm_create );
//  OTF2_GlobalEvtReaderCallbacks_SetCommDestroyCallback( evt_callbacks, print_comm_destroy );
  
  return evt_callbacks;
}

int main(int argc, char** argv) {
  if(argc<2) {
    fprintf(stderr, "usage: %s trace_file\n", argv[0]);
    return EXIT_FAILURE;
  }

  const char* trace_file = argv[1];
  OTF2_Reader* reader = OTF2_Reader_Open(trace_file);

  _set_callbacks(reader);

  /* Define definition callbacks. */
  OTF2_GlobalDefReaderCallbacks* def_callbacks = otf2_print_define_global_def_callbacks(reader);

  /* Get number of locations from the anchor file. */
  uint64_t num_locations = 0;
  OTF2_ErrorCode status = OTF2_SUCCESS;
  status = OTF2_Reader_GetNumberOfLocations( reader, &num_locations );
  check_status(status, "OTF2_Reader_GetNumberOfLocations failed. Number of locations: %" PRIu64,
	       num_locations );

  struct otf2_print_user_data user_data;
  memset(&user_data, 0, sizeof(user_data));
  /* Read global definitions. */
  uint64_t              definitions_read  = 0;
  OTF2_GlobalDefReader* global_def_reader = OTF2_Reader_GetGlobalDefReader( reader );
  check_pointer(global_def_reader, "Create global definition reader handle: failed\n");

  status = OTF2_Reader_RegisterGlobalDefCallbacks( reader, global_def_reader,
						   def_callbacks,
						   &user_data );
  check_status( status, "Register global definition callbacks." );

  status = OTF2_Reader_ReadGlobalDefinitions( reader, global_def_reader,
					      OTF2_UNDEFINED_UINT64,
					      &definitions_read );
  check_status( status, "Read global definitions. Number of definitions: %" PRIu64,
		definitions_read );

  OTF2_Reader_CloseGlobalDefReader( reader,
				    global_def_reader );


  OTF2_DefReaderCallbacks* local_def_callbacks = OTF2_DefReaderCallbacks_New();
  check_pointer( def_callbacks, "Create global definition callback handle." );

  for ( size_t i = 0; i < user_data.nb_locations; i++ ) {
    uint64_t* location_item = &user_data.locations[i];
    status = OTF2_Reader_SelectLocation( reader, *location_item );
    check_status( status, "Select location to read." );
  }

  status = OTF2_Reader_OpenDefFiles( reader );
  check_status( status, "Open local definition files for reading." );

  status = OTF2_Reader_OpenEvtFiles( reader );
  check_status( status, "Open event files for reading." );

    for ( size_t i = 0; i < user_data.nb_locations; i++ )
    {
        uint64_t* location_item      = &user_data.locations[i];
        uint64_t  locationIdentifier = *location_item;

	OTF2_EvtReader* evt_reader = OTF2_Reader_GetEvtReader( reader,
                                                                   locationIdentifier );
	check_pointer( evt_reader, "Create local event reader for location %" PRIu64 ".",
		       locationIdentifier );

	OTF2_DefReader* def_reader = OTF2_Reader_GetDefReader( reader,
							       locationIdentifier );
	/* a local def file is not mandatory */
	if ( def_reader )
	  {
	    status = OTF2_Reader_RegisterDefCallbacks( reader,
						       def_reader,
						       local_def_callbacks,
						       &locationIdentifier );
	    check_status( status, "Register local definition callbacks." );

	    uint64_t definitions_read = 0;
	    status = OTF2_SUCCESS;
	    do
	      {
		uint64_t def_reads = 0;
		status = OTF2_Reader_ReadAllLocalDefinitions( reader,
							      def_reader,
							      &def_reads );
		definitions_read += def_reads;

		/* continue reading, if we have a duplicate mapping table */
		if ( OTF2_ERROR_DUPLICATE_MAPPING_TABLE != status )
		  {
		    break;
		  }
	      }
	    while ( true );
	    check_status( status,
			  "Read %" PRIu64 " definitions for location %" PRIu64,
			  definitions_read,
			  locationIdentifier );

	    /* Close def reader, it is no longer useful and occupies memory */
	    status = OTF2_Reader_CloseDefReader( reader, def_reader );
	    check_status( status, "Close local definition reader." );
	  }
    }

    OTF2_DefReaderCallbacks_Delete( local_def_callbacks );
    status = OTF2_Reader_CloseDefFiles( reader );
    check_status( status, "Close local definition files for reading." );

    OTF2_Reader_Close( reader );


    /* ___ Read Event Records ____________________________________________________*/


    /* Add a nice table header. */
    printf( "=== Events =====================================================================\n" );

    /* Define event callbacks. */
    OTF2_GlobalEvtReaderCallbacks* evt_callbacks = otf2_print_create_global_evt_callbacks(reader);

    /* Get global event reader. */
    OTF2_GlobalEvtReader* global_evt_reader = OTF2_Reader_GetGlobalEvtReader( reader );
    check_pointer( global_evt_reader, "Create global event reader." );


    /* Register the above defined callbacks to the global event reader. */
    status = OTF2_Reader_RegisterGlobalEvtCallbacks( reader,
						     global_evt_reader,
						     evt_callbacks,
						     &user_data );
    check_status( status, "Register global event callbacks." );
    OTF2_GlobalEvtReaderCallbacks_Delete( evt_callbacks );
    /* Read until events are all read. */
    uint64_t otf2_STEP = 1;
    int otf2_EVENT_COLUMN_WIDTH=20;
    uint64_t events_read = otf2_STEP;

    printf( "%-*s %15s %20s  Attributes\n",
	    otf2_EVENT_COLUMN_WIDTH, "Event", "Location", "Timestamp" );
    printf( "--------------------------------------------------------------------------------\n" );
    while ( events_read > 0 )  {
        status = OTF2_Reader_ReadGlobalEvents( reader,
                                               global_evt_reader,
                                               1,
                                               &events_read );
        check_status( status, "Read %" PRIu64 " events.", events_read );
    }
    status = OTF2_Reader_CloseGlobalEvtReader( reader,
                                               global_evt_reader );
    check_status( status, "Close global definition reader." );
    status = OTF2_Reader_CloseEvtFiles( reader );
    check_status( status, "Close event files for reading." );
    

    return EXIT_SUCCESS;
}

/* -*-
   mode: c;
   c-file-style: "k&r";
   c-basic-offset 2;
   tab-width 2 ;
   indent-tabs-mode nil
   -*- */

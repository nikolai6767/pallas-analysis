/*
 * Copyright (C) Telecom SudParis
 * See LICENSE in top-level directory.
 */
#include <algorithm>
#include <iostream>
#include <string>
#include "pallas/pallas.h"
#include "pallas/pallas_archive.h"
#include "pallas/pallas_read.h"
#include "pallas/pallas_storage.h"
#include "pallas/pallas_write.h"
#include  <fstream>


void pop_data(pallas::Event* e, void* data, size_t data_size, byte*& cursor) {
  if (cursor == nullptr) {
    /* initialize the cursor to the begining of event data */
    cursor = &e->event_data[0];
  }

  uintptr_t last_event_byte = ((uintptr_t)e) + e->event_size;
  uintptr_t last_read_byte = ((uintptr_t)cursor) + data_size;
  pallas_assert(last_read_byte <= last_event_byte);

  memcpy(data, cursor, data_size);
  cursor += data_size;
}




/** Explore all the events of all the threads sorted by timestamp*/
 void compute_matrix(pallas::Archive& trace, int**& matrix){
   /* For each thread */
   for (int i = 0; i < trace.nb_threads; i++) {
     /* For each event in this thread */
     pallas::Thread* t = trace.threads[i];
     for (unsigned j = 0; j < t->nb_events; j++) {

       pallas::EventSummary e = t->events[j];
       /* If the event is a MPI_send */
       if (e.event.record == pallas::PALLAS_EVENT_MPI_SEND)
       {
          byte* cursor = nullptr;

          /* Get attributes of the MPI_SEND call */
          uint32_t receiver;
          uint32_t communicator;
          uint32_t msgTag;
          uint64_t msgLength;

          pop_data(&e.event, &receiver, sizeof(receiver), cursor);
          pop_data(&e.event, &communicator, sizeof(communicator), cursor);
          pop_data(&e.event, &msgTag, sizeof(msgTag), cursor);
          pop_data(&e.event, &msgLength, sizeof(msgLength), cursor);

          /* Update comm matrix for the thread and receiver with the message size msgLength that happened nb_occurences times  */
          matrix[i][receiver]+=msgLength*e.nb_occurences;

       }
    }
  }
 }




void save_matrix(std::string& filename, int** matrix, int size){
  std::ofstream outfile;
  outfile.open(filename.c_str());

  /* Write the first line containing all the ranks id */
  for (int i=0; i<size; i++)
  {
    outfile<<i+1;
    if (i<size-1)
      outfile<<",";
  }
  outfile<<std::endl;


  /* Write the content of the matrix rank by rank */
  for (int i = 0; i < size; i++) {
      for (int j = 0; j < size; j++) {
          outfile << matrix[i][j];
          if (j<size-1)
            outfile<< ",";
      }
      outfile<<std::endl;
  }

  outfile.close();
}



void usage(const std::string& prog_name) {
  std::cout<<std::endl<<"Usage: "<<prog_name<<" -t trace_file [OPTION] "<<std::endl;
   std::cout<<"\t-t FILE   --- Path to the Pallas trace file"<<std::endl;
   std::cout<<"\t[OPTION] -n OUTPUT_FILE   --- Name of the output file. Extension will be .mat. By default, the name of the file with be pallas_comm_matrix.mat"<<std::endl;
   std::cout<<"\t[OPTION] -? -h   --- Display this help and exit"<<std::endl;
}



int main(int argc, char** argv) {
  int nb_opts = 0;
  std::string trace_name="";
  std::string output_file=  "pallas_comm_matrix.mat";

  for (int i = 1; i < argc; i++) {
      if (!strcmp(argv[i], "-t")) {
         nb_opts++;
         trace_name =  std::string(argv[nb_opts + 1]);
         nb_opts++;
      }
       if (!strcmp(argv[i], "-n")) {
         nb_opts++;
         output_file =  std::string(argv[nb_opts + 1]);
         nb_opts++;
      }
      else if (!strcmp(argv[i], "-h") || !strcmp(argv[i], "-?")) {
        usage(argv[0]);
        return EXIT_SUCCESS;
    }
    else {
      /* Unknown parameter name. Stop parsing the parameter list and breaking. */
      break;
    }
  }


  if (trace_name == "") {
    usage(argv[0]);
    return EXIT_SUCCESS;
  }

  /* Reading archive */
  pallas::Archive trace = pallas::Archive();
  pallas_read_archive(&trace, const_cast<char *>(trace_name.c_str()));


  /* Create and initialize matrix with dimensions = number of threads */

  int size = trace.nb_threads;
  int** matrix = new int*[size];
  for (int i = 0; i < size; i++) {
    matrix[i] = new int[size];
      for (int j = 0; j < size; j++) {
          matrix[i][j]=0;
      }
  }


  /* Compute the comm matrix (MPI_SEND) between different threads */
  compute_matrix(trace, matrix);


  /* Save the matrix in the file */
  save_matrix(output_file, matrix,size);

  /* Free memory */
  delete[] (matrix);

  return EXIT_SUCCESS;
}

/* -*-
   mode: c;
   c-file-style: "k&r";
   c-basic-offset 2;
   tab-width 2 ;
   indent-tabs-mode nil
   -*- */

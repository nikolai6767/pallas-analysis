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
#include <cinttypes>


// #include "pallas/pallas.h"
// #include "pallas/pallas_archive.h"
// #include "pallas/pallas_hash.h"
// #include "pallas/pallas_parameter_handler.h"
// #include "pallas/pallas_storage.h"
// #include "pallas/pallas_timestamp.h"
// #include "pallas/pallas_write.h"

static bool verbose = true;




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




/**
 * Compare the timestamps of the current token on each thread and select the smallest timestamp.
 * @returns Tuple containing the ThreadId and a TokenOccurence.
 *          You are responsible for the memory of the TokenOccurence.
 */
static std::tuple<pallas::ThreadId, pallas::TokenOccurence> getNextToken(std::vector<pallas::ThreadReader>& threadReaders) {
  // Find the earliest threadReader
  pallas::ThreadReader* earliestReader = nullptr;
  for (auto& reader : threadReaders) {
    // Check if reader has finished reading its trace
    if (reader.current_frame < 0)
      continue;
    if (!earliestReader || earliestReader->referential_timestamp > reader.referential_timestamp) {
      earliestReader = &reader;
    }
  }

  // If no reader was available
  if (!earliestReader) {
    return {PALLAS_THREAD_ID_INVALID, {nullptr, nullptr}};
  }

  // Grab the interesting information
  auto& token = earliestReader->getCurToken();
  auto* occurrence = earliestReader->getOccurence(token, earliestReader->tokenCount[token]);
  auto threadId = earliestReader->thread_trace->id;

  // Update the reader
  earliestReader->updateReadCurToken();
  if (token.type == pallas::TypeEvent)
    earliestReader->moveToNextToken();

  return {threadId, {&token, occurrence}};
}





/* Treat one event */
static void read_event(const pallas::Thread* thread,
                        const pallas::Token token,
                        const pallas::EventOccurence* e,
                       int**& matrix) {

  byte* cursor = nullptr;
  if (e->event->record == pallas::PALLAS_EVENT_MPI_SEND) {

      /* Get attributes of the MPI_SEND calls */
      uint32_t receiver;
      uint32_t communicator;
      uint32_t msgTag;
      uint64_t msgLength;

      pop_data(e->event, &receiver, sizeof(receiver), cursor);
      pop_data(e->event, &communicator, sizeof(communicator), cursor);
      pop_data(e->event, &msgTag, sizeof(msgTag), cursor);
      pop_data(e->event, &msgLength, sizeof(msgLength), cursor);

      /* Increment comm matrix for the Thread thread and the thread receiver receiver with the message size msgLength  */
      matrix[thread->id][receiver]+=msgLength;

      if (verbose) {
        thread->printToken(token);
        printf("MPI_SEND(dest=%d, comm=%x, tag=%x, len=%" PRIu64 "", receiver, communicator, msgTag, msgLength);
        std::cout << "\t";
     }
  }

  if (verbose)
    std::cout << std::endl;
}















/* Treat one token */
static void read_token(const pallas::Thread* thread,
                        const pallas::Token* t,
                        const pallas::Occurence* e,
                        int**& matrix
                      ) {
   if (verbose) {
    printf("Reading repeated_token(%d.%d) for thread %s\n", t->type, t->id, thread->getName());
   }

  // Prints the repeated_token we first started with
  switch (t->type) {
    case pallas::TypeInvalid:
      pallas_error("Type is invalid\n");
      break;
    case pallas::TypeEvent:
      read_event(thread, *t, &e->event_occurence,matrix);
      break;
    default:
      break;
  }
}




/** Explore all the events of all the threads sorted by timestamp*/
 void compute_matrix(pallas::Archive& trace, int**& matrix){

  auto readers = std::vector<pallas::ThreadReader>();
  int reader_options = pallas::ThreadReaderOptions::None;

  for (int i = 0; i < trace.nb_threads; i++) {
    readers.emplace_back(&trace, trace.threads[i]->id, reader_options);
  }
  pallas::ThreadId threadId;
  pallas::TokenOccurence tokenOccurence;
  std::tie(threadId, tokenOccurence) = getNextToken(readers);
  while (threadId != PALLAS_THREAD_ID_INVALID) {
    auto currentReader = std::find_if(readers.begin(), readers.end(), [&threadId](const pallas::ThreadReader& reader) {
      return reader.thread_trace->id == threadId;
    });
    read_token(currentReader->thread_trace, tokenOccurence.token, tokenOccurence.occurence,matrix);
    // If you read the doc, you'll know that the memory of tokenOccurence is ours to manage
    std::tie(threadId, tokenOccurence) = getNextToken(readers);
  }
}




void save_matrix(std::string& filename, int** matrix){
    std::cout<<"Currently writing the save matrix function"<<std::endl;

    // remove warning temporarily
    std::cout<<"!!! Coming soon : function to save "<<filename<<" trace name !!!"<<std::endl;
    matrix[0][0]=0;
    //TODO
}


void usage(const std::string& prog_name) {
  std::cout<<std::endl<<"Usage: "<<prog_name<<" -t trace_file [OPTION] "<<std::endl;
   std::cout<<"\t-t FILE   --- Path to the Pallas trace file"<<std::endl;
   std::cout<<"\t[OPTION] -n OUTPUT_FILE   --- Name of the output file. Extension will be .mat. By default, the name of the file with be pallas_comm_matrix.mat"<<std::endl;
   std::cout<<"\t[OPTION]-? -h   --- Display this help and exit"<<std::endl;
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
  save_matrix(output_file, matrix);

  delete matrix;

  return EXIT_SUCCESS;
}

/* -*-
   mode: c;
   c-file-style: "k&r";
   c-basic-offset 2;
   tab-width 2 ;
   indent-tabs-mode nil
   -*- */

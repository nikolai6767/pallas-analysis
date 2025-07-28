#include <iostream>
#include <limits>
#include <string>
#include <iomanip>
#include "pallas/pallas.h"
#include "pallas/pallas_archive.h"
#include "pallas/pallas_log.h"
#include "pallas/pallas_read.h"
#include "pallas/pallas_storage.h"
#include <time.h>
#include <fstream>
#include <float.h>
#include "pallas/pallas_dbg.h"
#include <sys/wait.h>



pallas_timestamp_t get_timestamp(const pallas::EventOccurence e){ 
    return(e.timestamp);
}

static void get_thread_timestamp(const pallas::Thread* thread, const pallas::Token token, const pallas::EventOccurence e) {
    get_timestamp(e);  
}



struct thread_data {
  std::vector<std::string> callstack{};
  std::vector<pallas_duration_t> callstack_duration{};
  std::vector<pallas_timestamp_t> callstack_timestamp{};
  pallas_timestamp_t last_timestamp;
};

bool isReadingOver(const std::vector<pallas::ThreadReader>& readers) {
  for (const auto& reader : readers) {
    if (!reader.isEndOfTrace()) {
      return false;
    }
  }
  return true;

}





void printTrace(pallas::GlobalArchive& trace1) {

  std::map<pallas::ThreadReader*, struct thread_data> threads_data;

  auto readers = std::vector<pallas::ThreadReader>();

  auto thread_list = trace1.getThreadList();


  for (auto * thread: thread_list) {
      if (thread == nullptr)  continue;
      readers.emplace_back(thread->archive, thread->id, PALLAS_READ_FLAG_UNROLL_ALL);
      threads_data[&readers.back()] = {};
  }

  while (!isReadingOver(readers)) {
    pallas::ThreadReader* min_reader = &readers[0];
    pallas_timestamp_t min_timestamp = std::numeric_limits<unsigned long>::max();

        
    for (auto & reader : readers) {
      if (!reader.isEndOfTrace() && reader.currentState.currentFrame->referential_timestamp < min_timestamp) {
        min_reader = &reader;
        min_timestamp = reader.currentState.currentFrame->referential_timestamp;
      }
    
    }

    auto token = min_reader->pollCurToken();


    if (token.type == pallas::TypeEvent) {
        auto res = min_reader->getEventOccurence(token, min_reader->currentState.currentFrame->tokenCount[token]);
        pallas_timestamp_t t1 = get_timestamp(res);
    }
  }
    //   auto test = min_reader->getNextToken().isValid();

  }












int main(const int argc, char* argv[]) {

    if (argc != 2){
        std::cerr << "Usage: " << argv[0] << " <trace1.pallas> <trace2.pallas>" << std::endl;
        return EXIT_FAILURE;
    }

    auto trace1_name = argv[1];
    auto trace1 = pallas_open_trace(trace1_name);

    if (trace1==nullptr)
        return EXIT_FAILURE;

    // auto trace2_name = argv[1];
    // auto trace2 = pallas_open_trace(trace2_name);

    // if (trace2==nullptr)
    //     return EXIT_FAILURE;

    printTrace(*trace1);


    delete trace1;
    // delete trace2;
    
    return EXIT_SUCCESS;
}
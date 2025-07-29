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
#include <malloc.h>

#define EPSILON 1e-12

bool details = true;

int thread_to_print = -1;

void write_csv_details(const char* filename, pallas_timestamp_t t){
    std::ofstream file(std::string(filename) + ".csv", std::ios::app);
    file << t << "\n";
}

pallas_timestamp_t get_timestamp(const pallas::EventOccurence e){ 
    return(e.timestamp);
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

void getTraceTimepstamps(char* name) {

  auto trac = pallas_open_trace(name);
  auto trace = *trac;

  std::map<pallas::ThreadReader*, struct thread_data> threads_data;

  auto readers = std::vector<pallas::ThreadReader>();

  auto thread_list = trace.getThreadList();

  for (auto * thread: thread_list) {
      if (thread == nullptr)  continue;
      if(!(thread_to_print < 0 || thread->id == thread_to_print)) continue;
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
        auto* copy = strdup(name);
        auto* slash = strrchr(copy, '/');
        *slash = '\0';
        write_csv_details(copy, t1);
        free(copy);

    }
    if (! min_reader->getNextToken().isValid()) {
      pallas_assert(min_reader->isEndOfTrace());
    }
  }
  delete trac;
}


double mean(const char* trace){
  FILE* file = fopen(trace, "r");

  double n = 0.0;
  double y, sum_y = 0.0;

  if (!file){
    perror("[mean_fopen]");
    return EXIT_FAILURE;
  }
  while(fscanf(file, "%lf", &y) == 1){
    sum_y += y;
    n++;
  }

  fclose(file);

  double mean_y = sum_y / n ;
    if (details){
      std::cout.precision(12);
      std::cout << "------------------------------------------------------------------" << std::endl;
      std::cout << std::right << std::fixed << "Details: ";
      std::cout << " n = " << n << "," << "mean_y = " << mean_y << std::endl;
    }
    return mean_y;
}


/**
 * Test R2
 * R^2 = 1 - (SS_res / SS_tot) where:
 * SS_res = sum_1_n(y_i - x_i)^2
 * SS_tot = sum_1_n(y_i - mean(y))^2
 */
double CompareTimestamps(const char* trace1, const char* trace2){
    FILE* file1 = fopen(trace1, "r");
    FILE* file2 = fopen(trace2, "r");
    if (!file1 || !file2) {
        perror("[fopen]");
        return EXIT_FAILURE;
    }

    double y, val2;
    double ss_res, ss_tot = 0.0;
    double mean_y = mean(trace1);

    while ((fscanf(file1, "%lf", &y)==1) && (fscanf(file2, "%lf", &val2)==1)) {
        ss_res += (y - val2) * (y - val2);
        ss_tot += (y - mean_y) * (y - mean_y);
    }

    fclose(file1);
    fclose(file2);

    if (ss_tot < EPSILON){
      fprintf(stderr, "\nss_tot nul\n");
      return -1.0;
    }
    if (details){
      std::cout.precision(0);
      std::cout << "------------------------------------------------------------------" << std::endl;
      std::cout << std::right << std::fixed << "Details: ";
      std::cout << " ss_res = " << ss_res << "," << " ss_tot = " << ss_tot << std::endl;
    }
    double R2 = 1.0 - ss_res / ss_tot;
    return R2;
}

auto get_name_w_csv(char* c){
  auto* copy = strdup(c);
  auto* slash = strrchr(copy, '/');
  if (slash != nullptr)
    *slash = '\0';
  std::string copy_s = std::string(copy) + ".csv";
  free(copy);
  return copy_s;
}

int main(const int argc, char* argv[]) {

    if (argc != 3){
        std::cerr << "Usage: " << argv[0] << " <trace1.pallas> <trace2.pallas> " << std::endl;
        return EXIT_FAILURE;
    }

    int status1, status2;
    auto trace_csv_1 = get_name_w_csv(argv[1]);
    auto trace_csv_2 = get_name_w_csv(argv[2]);

    std::ofstream(trace_csv_1, std::ios::trunc);
    std::ofstream(trace_csv_2, std::ios::trunc);

    pid_t pid1 = fork();

    if (pid1 == 0){
      getTraceTimepstamps(argv[1]);
      exit(EXIT_SUCCESS);
    }

    pid_t pid2 = fork();
    if (pid2 == 0){
      getTraceTimepstamps(argv[2]);
      exit(EXIT_SUCCESS);
    }
    waitpid(pid1, &status1, 0);
    waitpid(pid2, &status2, 0);

    if (!WIFEXITED(status1) || !WIFEXITED(status2)) {
      std::cerr << "One of the child processes failed." << std::endl;
      return EXIT_FAILURE;
    }


    double res = CompareTimestamps(trace_csv_1.c_str(), trace_csv_2.c_str());

    std::cout.precision(12);

    std::cout << "------------------------------------------------------------------" << std::endl;
    std::cout << std::right << std::setw(30) << std::fixed << "Result: R^2 = ";
    std::cout << std::right << std::setw(12) << std::fixed << res << std::endl;
    std::cout << "------------------------------------------------------------------" << std::endl;


    return EXIT_SUCCESS;
}
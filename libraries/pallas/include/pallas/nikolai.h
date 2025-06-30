#ifndef NIKOLAI

#define NIKOLAI


typedef struct {
	double total_d;
	double min_d;
	double max_d;
	int count;
} Duration;

void duration_init(Duration* d) {
	d->total_d = 0.0;
	d->min_d = DBL_MAX;
	d->max_d = 0.0;
	d-> count = 0;
}

void update_duration(Duration* d, struct timespec t1, struct timespec t2){
	long time = (t2.tv_sec - t1.tv_sec) * 1e9 + (t2.tv_nsec - t1.tv_nsec);
	d->total_d += time;
	if (time < d ->min_d) d->min_d = time;
	if (time > d -> max_d) d-> max_d = time;
	d->count++;
}

void duration_write_csv(const char* filename, const Duration* d) {
    std::ofstream file(std::string(filename) + ".csv");
    file << "calls,total_us,min_us,max_us,average_us\n";
    file << d->count << "," << d->total_d << "," << d->min_d << "," << d->max_d << ",";
    file << (d->count ? d->total_d / d->count : 0) << "\n";
}

enum FunctionIndex {
  PRINT_TIMESTAMP,
  PRINT_TIMESTAMP_HEADER,
  PRINT_DURATION,
  PRINT_DURATION_HEADER,
  PRINT_EVENT,
  PRINT_FLAME,
  PRINT_CSV,
  PRINT_CSV_BULK,
  PRINT_TRACE,
  GET_CURRENT_INDEX,
  PRINT_THREAD_STRUCTURE,
  PRINT_STRUCTURE,
  NB_FUNCTIONS
};


Duration durations[NB_FUNCTIONS];


void duration_write_all_csv(const char* filename) {
  std::ofstream file(std::string(filename) + ".csv");
  file << "function,calls,total,min,max,average\n";

  const char* function_names[NB_FUNCTIONS] = {
  "PRINT_TIMESTAMP",
  "PRINT_TIMESTAMP_HEADER",
  "PRINT_DURATION",
  "PRINT_DURATION_HEADER",
  "PRINT_EVENT",
  "PRINT_FLAME",
  "PRINT_CSV",
  "PRINT_CSV_BULK",
  "PRINT_TRACE",
  "GET_CURRENT_INDEX",
  "PRINT_THREAD_STRUCTURE",
  "PRINT_STRUCTURE"
  };

  for (int i = 0; i < NB_FUNCTIONS; ++i) {
    const Duration& d = durations[i];
    double avg = d.count ? d.total_d / d.count : 0.0;
    file << function_names[i] << "," << d.count << "," << d.total_d << "," << d.min_d << "," << d.max_d << "," << avg << "\n";
  }
}

#endif

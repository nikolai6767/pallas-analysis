

typedef struct {
	double total_d;
	double min_d;
	double max_d;
	int count;
} Duration;

void duration_init(Duration* d);

void update_duration(Duration* d, struct timespec t1, struct timespec t2);

void duration_write_csv(const char* filename, const Duration* d);

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


void duration_write_all_csv(const char* filename);